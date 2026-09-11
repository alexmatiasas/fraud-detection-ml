from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from fdml.features.factory import (
    create_pipeline,
    load_data,
    load_data_config,
    merge_tables,
)
from fdml.features.factory import (
    load_fe_config as load_features_config,
)
from fdml.models.config import (
    load_evaluation_config,
    load_mlflow_config,
    load_train_config,
    resolve_mlflow_tracking,
)
from fdml.models.evaluate.reporter import (
    ConsoleReporter,
    DVCLiveReporter,
    JSONFileReporter,
    LoggingReporter,
    MLflowReporter,
    log_dataset_lineage,
)
from fdml.models.evaluate.runner import evaluate
from fdml.models.split import StratifiedSplitter, TemporalSplitter
from fdml.models.train.model_builder import model_builder_registry
from fdml.utils.logging import (
    ensure_logging,
    get_phase_timings,
    setup_logging,
    step,
)

logger = logging.getLogger(__name__)

_TARGET = "isFraud"


class TrainResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    # Raw (pre-feature-engineering) validation fold — the contract the full
    # pipeline (features + model) is served with, so signature and
    # input_example must be inferred from it, not from the featurized frame.
    X_val_raw: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    cfg: Any
    params: dict[str, Any]
    feature_names: list[str]


def _get_splitter(split_cfg: Any) -> TemporalSplitter | StratifiedSplitter:
    if split_cfg.strategy == "temporal":
        return TemporalSplitter(
            time_col=split_cfg.time_col,
            test_size=split_cfg.test_size,
            embargo_seconds=split_cfg.embargo_seconds,
        )
    if split_cfg.strategy == "random":
        return StratifiedSplitter(test_size=split_cfg.test_size)
    msg = f"Unknown split strategy: {split_cfg.strategy}"
    raise ValueError(msg)


def fit_transform_steps(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Run ``fit_transform`` step by step so each step's cost is visible."""
    Xt = X.copy()
    for name, transformer in pipeline.steps:
        start = time.perf_counter()
        if hasattr(transformer, "fit_transform"):
            Xt = transformer.fit_transform(Xt, y)
        else:
            Xt = transformer.fit(Xt, y).transform(Xt)
        elapsed = time.perf_counter() - start
        logger.info(
            "    step %-12s → %s rows × %s cols (%.2fs)",
            name,
            f"{Xt.shape[0]:,}",
            Xt.shape[1],
            elapsed,
        )
    return Xt


def transform_steps(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run ``transform`` step by step over already-fitted transformers."""
    Xt = X.copy()
    for name, transformer in pipeline.steps:
        Xt = transformer.transform(Xt)
        logger.info("    step %-12s → %s cols", name, Xt.shape[1])
    return Xt


def load_split(
    cfg: Any, mlflow_cfg: Any = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """PHASES 1-2: load the merged data and produce the train/val split."""

    data_cfg = load_data_config()

    with step("PHASE 1: Load data"):
        train_df, identity_df = load_data(data_cfg)
        df = merge_tables(train_df, identity_df)
        logger.info("  Merged: %s rows x %s cols", f"{df.shape[0]:,}", df.shape[1])
        logger.info(
            "  Target: %.2f%% fraud (%.0f positives)",
            df[_TARGET].mean() * 100,
            df[_TARGET].sum(),
        )

    with step("PHASE 2: Train / validation split"):
        splitter = _get_splitter(cfg.split)
        train_idx, val_idx = next(splitter.split(df, df[_TARGET].to_numpy()))

        X = df.drop(columns=[_TARGET])
        y = df[_TARGET]

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logger.info(
            "  Train: %s rows (fraud %.2f%%)", f"{len(X_train):,}", y_train.mean() * 100
        )
        logger.info(
            "  Val:   %s rows (fraud %.2f%%)", f"{len(X_val):,}", y_val.mean() * 100
        )

        if cfg.split.strategy == "temporal" and cfg.split.time_col in X.columns:
            train_tr = (
                int(X_train[cfg.split.time_col].min()),
                int(X_train[cfg.split.time_col].max()),
            )
            val_tr = (
                int(X_val[cfg.split.time_col].min()),
                int(X_val[cfg.split.time_col].max()),
            )
            logger.info(
                "  Time range — train: [%d, %d]  val: [%d, %d]",
                train_tr[0],
                train_tr[1],
                val_tr[0],
                val_tr[1],
            )

        cap = cfg.data.max_train_rows
        if cap and len(X_train) > cap:
            X_train, y_train = _cap_train_fold(cfg, X_train, y_train, cap=cap)

        if mlflow_cfg is not None and mlflow_cfg.datasets.enabled:
            log_dataset_lineage(X_train, y_train, X_val, y_val)

    return X_train, X_val, y_train, y_val


def _cap_train_fold(
    cfg: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cap: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep the earliest ``cap`` rows of the training fold (by time when possible)."""
    if cfg.split.strategy == "temporal" and cfg.split.time_col in X_train.columns:
        X_train = X_train.sort_values(cfg.split.time_col).head(cap)
    else:
        X_train = X_train.head(cap)
    y_train = y_train.loc[X_train.index]
    logger.info(
        "  data.max_train_rows: train capped to %s rows (earliest by time)",
        f"{len(X_train):,}",
    )
    return X_train, y_train


def featurize(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    features_cfg: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, list[str]]:
    """PHASE 3: run the feature pipeline over an already-split dataset."""

    with step("PHASE 3: Feature engineering"):
        pipeline, _ = create_pipeline(features_cfg)
        X_train_fe = fit_transform_steps(pipeline, X_train, y_train)
        X_val_fe = transform_steps(pipeline, X_val)

        logger.info(
            "  Pipeline: %d steps → %s train, %s val",
            len(pipeline.steps),
            f"{X_train_fe.shape[1]} cols",
            f"{X_val_fe.shape[1]} cols",
        )
        v_before, v_after = X_train.shape[1], X_train_fe.shape[1]
        logger.info(
            "  Feature delta: %+d (from %d to %d columns)",
            v_after - v_before,
            v_before,
            v_after,
        )

        def _fmt_dtypes(df: pd.DataFrame) -> str:
            return ", ".join(
                f"{k}: {v}"
                for k, v in df.dtypes.apply(lambda x: x.name).value_counts().items()
            )

        logger.info("  dtypes: %s", _fmt_dtypes(X_train_fe))

    return X_train_fe, X_val_fe, pipeline, X_train_fe.columns.tolist()


def _sample_eval_set(
    X_val: pd.DataFrame, y_val: pd.Series, max_rows: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Fixed, stratified subsample of the validation fold for early stopping.

    Uses a constant random_state (not the training seed) so every seed of an
    experiment early-stops on the same validation rows, keeping per-iteration
    curves comparable across runs. Final metrics are still computed on the
    full validation fold in PHASE 5.
    """
    if max_rows <= 0 or len(y_val) <= max_rows:
        return X_val, y_val
    from sklearn.model_selection import train_test_split

    X_es, _rest_x, y_es, _rest_y = train_test_split(
        X_val, y_val, train_size=max_rows, stratify=y_val, random_state=0
    )
    logger.info(
        "  Early stopping eval set: %d rows (of %d val)",
        len(y_es),
        len(y_val),
    )
    return cast(pd.DataFrame, X_es), cast(pd.Series, y_es)


def _prepare_data(cfg: Any, mlflow_cfg: Any = None) -> tuple:
    """PHASES 1-3: load, split, feature engineering."""

    X_train, X_val, y_train, y_val = load_split(cfg, mlflow_cfg)
    X_train_fe, X_val_fe, pipeline, feature_names = featurize(
        X_train, X_val, y_train, load_features_config()
    )
    X_val_es, y_val_es = _sample_eval_set(
        X_val_fe, y_val, cfg.early_stopping.eval_max_rows
    )

    return (
        X_train_fe,
        X_val_fe,
        X_val,
        y_train,
        y_val,
        X_val_es,
        y_val_es,
        pipeline,
        feature_names,
    )


def _fit_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Any,
    callbacks: list | None = None,
) -> Any:
    """Fit ``model`` with an evaluation set and native early stopping.

    LightGBM 4.x removed the ``early_stopping_rounds`` estimator attribute, so
    early stopping is applied through the native ``lgb.early_stopping`` /
    ``xgb.callback.EarlyStopping`` callbacks instead of estimator parameters.
    """
    use_es = cfg.early_stopping.enabled
    if use_es:
        logger.info(
            "  Early stopping: %d rounds on %s",
            cfg.early_stopping.rounds,
            cfg.early_stopping.eval_metric,
        )

    import lightgbm as lgb  # noqa: PLC0415
    import xgboost as xgb  # noqa: PLC0415

    eval_set = [(X_val, y_val)] if use_es else None
    if eval_set is None:
        model.fit(X_train, y_train)
        return model

    # Optional per-iteration TRAIN curve: a fixed random sample of the training
    # rows is added as a second eval dataset so overfitting/underfitting is
    # visible (train/auc vs val/auc at each iteration). 0 = off.
    train_eval: tuple[pd.DataFrame, pd.Series] | None = None
    train_rows = cfg.early_stopping.train_eval_max_rows
    if train_rows:
        n_train = len(X_train)
        if n_train > train_rows:
            idx = np.sort(
                np.random.default_rng(cfg.seed).choice(
                    n_train, train_rows, replace=False
                )
            )
        else:
            idx = np.arange(n_train)
        train_eval = (X_train.iloc[idx], y_train.iloc[idx])
        logger.info(
            "  Train eval: %s rows sampled for the per-iteration train curve",
            f"{len(idx):,}",
        )

    if isinstance(model, lgb.LGBMClassifier):
        eval_names = ["validation"]
        if train_eval:
            eval_set = eval_set + [train_eval]
            eval_names.append("train")
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=cfg.early_stopping.eval_metric,
            callbacks=[
                lgb.early_stopping(cfg.early_stopping.rounds, first_metric_only=True),
                *(callbacks or []),
            ],
        )
    elif isinstance(model, xgb.XGBClassifier):
        # XGBoost >= 3.x sklearn API: early stopping, eval_metric and callbacks
        # are estimator constructor params, not fit() kwargs. Callbacks must be
        # TrainingCallback instances, so convert the LightGBM-style ones.
        # XGBoost early-stops on the LAST eval dataset, so validation goes last;
        # the callback name_map keeps train/validation distinguishable.
        if train_eval:
            eval_set = [train_eval, (X_val, y_val)]
            name_map = {"validation_0": "train", "validation_1": "validation"}
        else:
            name_map = {"validation_0": "validation"}
        xgb_kwargs: dict[str, Any] = {
            "eval_metric": cfg.early_stopping.eval_metric,
            "early_stopping_rounds": cfg.early_stopping.rounds,
        }
        if callbacks:
            xgb_kwargs["callbacks"] = [
                cb.as_xgboost(name_map) for cb in callbacks if hasattr(cb, "as_xgboost")
            ]
        model.set_params(**xgb_kwargs)
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def train(
    cfg: Any = None,
    cli_args: list[str] | None = None,
    callbacks: list | None = None,
    hpo_strategy: Any = None,
    mlflow_cfg: Any = None,
) -> TrainResult:
    ensure_logging()

    if cfg is None:
        cfg = load_train_config(cli_args=cli_args)

    (
        X_train_fe,
        X_val_fe,
        X_val_raw,
        y_train,
        y_val,
        X_val_es,
        y_val_es,
        pipeline,
        feature_names,
    ) = _prepare_data(cfg, mlflow_cfg)

    with step("PHASE 4: Model training"):
        builder = model_builder_registry.get(cfg.model.name)
        logger.info("Model: %s", cfg.model.name)

        params = {**cfg.model.params.model_dump(), "random_state": cfg.seed}

        if hpo_strategy is not None:
            logger.info("  Running HPO before final training ...")
            best_params = hpo_strategy.search(
                X_train_fe,
                y_train,
                X_val_fe,
                y_val,
                builder=builder,
                base_params=params,
                es_rounds=cfg.early_stopping.rounds,
                es_metric=cfg.early_stopping.eval_metric,
                seed=cfg.seed,
            )
            logger.info("  Best HPO params: %s", builder.format_params(best_params))
            params = best_params

        logger.info("  params: %s", builder.format_params(params))

        model = builder.build(params)
        model = _fit_model(
            model, X_train_fe, y_train, X_val_es, y_val_es, cfg, callbacks
        )

        logger.info("  ✓ Training complete")

    return TrainResult(
        model=model,
        pipeline=pipeline,
        X_train=X_train_fe,
        X_val=X_val_fe,
        X_val_raw=X_val_raw,
        y_train=y_train,
        y_val=y_val,
        cfg=cfg,
        params=params,
        feature_names=feature_names,
    )


def _passes_quality_gate(mlflow_cfg: Any, auc: float, average_precision: float) -> bool:
    """True when the run clears the registry's minimum-metric thresholds."""
    gate = mlflow_cfg.registry
    failures: list[str] = []
    if gate.min_auc > 0 and auc < gate.min_auc:
        failures.append(f"AUC {auc:.4f} < {gate.min_auc}")
    if (
        gate.min_average_precision > 0
        and average_precision < gate.min_average_precision
    ):
        failures.append(f"AP {average_precision:.4f} < {gate.min_average_precision}")
    if failures:
        logger.warning("  Registry: quality gate REJECTED — %s", "; ".join(failures))
        if mlflow.active_run() is not None:
            mlflow.set_tag("validation_status", "rejected")
        return False
    return True


def _register_model(
    mlflow_cfg: Any,
    auc: float,
    run_id: str,
    model_uri: str | None = None,
    average_precision: float = 0.0,
    cfg_model_name: str = "lightgbm",
) -> None:
    from mlflow import MlflowClient

    if not _passes_quality_gate(mlflow_cfg, auc, average_precision):
        return

    client = MlflowClient()
    # Derive registry model name from algorithm name
    _REGISTRY_NAMES = {
        "lightgbm": "fraud-detection-lgbm",
        "xgboost": "fraud-detection-xgboost",
        "random_forest": "fraud-detection-rf",
    }
    model_name = _REGISTRY_NAMES.get(cfg_model_name, mlflow_cfg.registry.model_name)
    registry_tags = dict(mlflow_cfg.registry.tags)

    # MLflow 3 logs models as LoggedModels outside the run's artifacts; use
    # the models:/m-<id> URI returned by log_model(). The legacy
    # runs:/{run_id}/model pointer only works when the server resolves it to
    # that LoggedModel — unreliable on DagsHub (v50/v53 registered empty
    # schemas this way).
    if not model_uri:
        logger.warning(
            "  Registry: no logged-model URI (log_model disabled or failed)"
            " — skipping registration"
        )
        return

    try:
        client.create_registered_model(
            model_name, description=mlflow_cfg.registry.description
        )
        logger.info("  Registry: created model '%s'", model_name)
    except Exception:
        pass

    try:
        _model_label = cfg_model_name.replace("_", " ").title()
        version_desc = (
            f"{_model_label} baseline — AUC={auc:.4f}, AP={average_precision:.4f}"
        )
        version_tags = {
            **registry_tags,
            "validation_status": "approved",
            "validation_auc": f"{auc:.4f}",
            "validation_ap": f"{average_precision:.4f}",
        }
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=version_desc,
            tags=version_tags,
        )
        version = mv.version
        mlflow.log_param("registered_model_version", version)
        mlflow.set_tag("registered_model_version", version)
        mlflow.set_tag("validation_status", "approved")
        logger.info("  Registry: created version %s (AUC=%.4f)", version, auc)

        try:
            champion_mv = client.get_model_version_by_alias(model_name, "champion")
            champion_run_id = champion_mv.run_id
            champion_auc = 0.0
            if champion_run_id is not None:
                champion_run = client.get_run(champion_run_id)
                champion_auc = champion_run.data.metrics.get("val/roc_auc", 0.0)

            if auc > champion_auc:
                client.set_registered_model_alias(model_name, "champion", version)
                logger.info(
                    "  Registry: alias 'champion' → v%s (AUC=%.4f > %.4f)",
                    version,
                    auc,
                    champion_auc,
                )
            else:
                client.set_registered_model_alias(model_name, "challenger", version)
                logger.info(
                    "  Registry: alias 'challenger' → v%s (AUC=%.4f ≤ champion %.4f)",
                    version,
                    auc,
                    champion_auc,
                )
        except Exception:
            client.set_registered_model_alias(model_name, "champion", version)
            logger.info(
                "  Registry: alias 'champion' → version %s (first model)", version
            )
    except Exception as exc:
        logger.warning("  Registry: failed to register model: %s", exc)


def _run_native_evaluation(
    mlflow_cfg: Any,
    model_info: Any,
    X_val_raw: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Native ``mlflow.models.evaluate`` on the logged full pipeline.

    Runs over RAW transactions (the pipeline's serving contract), computes the
    standard classifier metric set, interactive ROC/PR/confusion-matrix
    artifacts and optionally a SHAP explainer, and links everything to both
    the run and the LoggedModel. Failures degrade to a warning — the custom
    evaluate runner has already produced the authoritative metrics.
    """
    cfg = mlflow_cfg.native_evaluate
    if not cfg.enabled or model_info is None:
        return

    df = X_val_raw.copy()
    # pandas Categorical breaks the default evaluator ("Cannot interpret
    # CategoricalDtype ... as a data type"); the served pipeline accepts
    # object columns equally well (CategoryEncoder handles both).
    cat_cols = df.select_dtypes(include="category").columns
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].astype(object)
    df["isFraud"] = y_val.to_numpy()
    if cfg.max_rows > 0 and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=0)
    logger.info(
        "  Native evaluation: %s rows (explainer=%s)",
        f"{len(df):,}",
        cfg.log_explainer,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Hint: Inferred schema contains integer column"
            )
            mlflow.models.evaluate(
                model_info.model_uri,
                df,
                targets="isFraud",
                model_type="classifier",
                evaluator_config={"log_explainer": cfg.log_explainer},
            )
            logger.info("  ✓ Native evaluation logged")
    except Exception as exc:
        logger.warning("  Native evaluation failed: %s", exc)


def _build_callbacks(cfg: Any) -> list | None:
    if not cfg.training_callbacks.log_per_iteration:
        return None
    from fdml.models.train.callbacks import IterationCallback

    return [
        IterationCallback(
            log_mlflow=True,
            log_dvclive=cfg.training_callbacks.dvclive,
            log_console=True,
            mlflow_every=cfg.training_callbacks.mlflow_every,
        )
    ]


def _log_configs_and_env(cfg: Any) -> None:
    for config_path in [
        "configs/train.yaml",
        "configs/features.yaml",
        "configs/evaluate.yaml",
        "configs/mlflow.yaml",
    ]:
        if Path(config_path).exists():
            mlflow.log_artifact(config_path, "configs")

    # Resolved config (after CLI overrides) — makes the run reproducible from
    # MLflow alone, since the raw yaml files don't reflect `key=value` overrides.
    import json
    import tempfile

    try:
        resolved = json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False
        ) as f:
            f.write(resolved)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, "configs")
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Config snapshot failed: %s", exc)
    for dep_path in ["pyproject.toml", "uv.lock"]:
        if Path(dep_path).exists():
            mlflow.log_artifact(dep_path, "env")


def _log_git_tags() -> None:
    for tag_cmd, tag_name in [
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "git_branch"),
        (["git", "rev-parse", "--short", "HEAD"], "git_commit"),
    ]:
        try:
            result = subprocess.run(
                tag_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            mlflow.set_tag(tag_name, result.stdout.strip())
        except Exception:
            pass
    try:
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
            != ""
        )
        mlflow.set_tag("git_dirty", "yes" if dirty else "no")
    except Exception:
        pass


def features_fingerprint(features_cfg: Any) -> str:
    """Canonical sha1 of the resolved feature configuration.

    Distinguishes runs that used different feature sets even though the
    ``features.yaml`` file on disk changed between runs. Two runs share a
    ``features_hash`` iff their feature pipeline is identical.
    """
    import json

    payload = json.dumps(
        features_cfg.model_dump(mode="json"), sort_keys=True, default=str
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _register_abort_handler() -> None:
    """Tag the active MLflow run ``status=aborted`` on Ctrl-C / SIGTERM.

    The tag lets ``compare_runs.py`` exclude interrupted runs instead of
    showing them as metric-less noise. Runs that are killed hard (SIGKILL)
    cannot be tagged — they are simply dropped by the completeness filter.
    """
    import signal

    def _mark_aborted(signum, frame):  # noqa: ARG001
        try:
            if mlflow.active_run() is not None:
                mlflow.set_tag("status", "aborted")
                logger.warning("  Interrupted — run tagged status=aborted")
        except Exception:
            pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _mark_aborted)
    signal.signal(signal.SIGTERM, _mark_aborted)


def main() -> None:
    cfg = load_train_config(cli_args=sys.argv[1:])
    tag = cfg.mlflow.experiment_tag
    run_name = f"{cfg.model.name}_s{cfg.seed}" + (f"_{tag}" if tag else "")
    log_path = setup_logging(log_path=f"train_{run_name}.log")

    if cfg.ablation.enabled:
        from fdml.models.train.ablation import run_ablation, summarize

        results = run_ablation(cfg, max_train_rows=cfg.ablation.max_train_rows)
        report_path = Path(cfg.ablation.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summarize(results).to_csv(report_path, index=False)
        logger.info("Ablation report saved to %s", report_path)
        logger.info("\n%s", summarize(results).to_string(index=False))
        return

    eval_cfg = load_evaluation_config()
    mlflow_cfg = resolve_mlflow_tracking(load_mlflow_config())

    mlflow.set_tracking_uri(mlflow_cfg.tracking.tracking_uri)
    try:
        mlflow.set_experiment(mlflow_cfg.tracking.experiment_name)
    except Exception as exc:
        logger.warning(
            "  MLflow: experiment setup failed (%s), falling back to local SQLite", exc
        )
        mlflow_cfg.tracking.tracking_uri = "sqlite:///mlruns.db"
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(mlflow_cfg.tracking.experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        _register_abort_handler()
        _log_git_tags()

        if cfg.mlflow.experiment_tag:
            mlflow.set_tag("experiment", cfg.mlflow.experiment_tag)
            mlflow.log_param("experiment", cfg.mlflow.experiment_tag)

        features_hash = features_fingerprint(load_features_config())
        mlflow.set_tag("features_hash", features_hash)
        mlflow.log_param("features_hash", features_hash)

        hpo_strategy = None
        if cfg.optuna.enabled:
            from fdml.models.train.hpo import OptunaHPO

            hpo_strategy = OptunaHPO(
                n_trials=cfg.optuna.n_trials,
                timeout_seconds=cfg.optuna.timeout_seconds,
                direction=cfg.optuna.direction,
                study_name=cfg.optuna.study_name,
                save_best=cfg.optuna.save_best_params,
            )

        callbacks = _build_callbacks(cfg)
        train_start = time.perf_counter()
        result = train(
            cfg,
            callbacks=callbacks,
            hpo_strategy=hpo_strategy,
            mlflow_cfg=mlflow_cfg,
        )
        train_elapsed_s = time.perf_counter() - train_start

        with step("PHASE 5: Evaluation + MLflow tracking"):
            mlflow.log_params(result.params)
            mlflow.log_params({"seed": cfg.seed})
            mlflow.log_metric("training_elapsed_s", round(train_elapsed_s, 1))

            for phase_name, elapsed in get_phase_timings().items():
                match = re.search(r"PHASE (\d+)", phase_name)
                key = (
                    f"phase_{match.group(1)}_seconds"
                    if match
                    else f"phase_{phase_name}_seconds"
                )
                mlflow.log_metric(key, round(elapsed, 1))

            if (
                cfg.split.strategy == "temporal"
                and cfg.split.time_col in result.X_train.columns
            ):
                mlflow.log_params(
                    {
                        "split_embargo_seconds": cfg.split.embargo_seconds,
                        "train_dt_min": int(result.X_train[cfg.split.time_col].min()),
                        "train_dt_max": int(result.X_train[cfg.split.time_col].max()),
                        "val_dt_min": int(result.X_val[cfg.split.time_col].min()),
                        "val_dt_max": int(result.X_val[cfg.split.time_col].max()),
                    }
                )

            best_iter = getattr(result.model, "best_iteration_", None)
            if best_iter is None:
                best_iter = getattr(result.model, "best_iteration", None)
            if best_iter is not None:
                mlflow.log_metric("val/best_iteration", int(best_iter))
            model = result.model
            y_proba = model.predict_proba(result.X_val)[:, 1]
            y_pred = model.predict(result.X_val)

            train_proba = model.predict_proba(result.X_train)[:, 1]
            mlflow.log_metric(
                "train/roc_auc",
                round(float(roc_auc_score(result.y_train, train_proba)), 5),
            )

            reporters = [
                ConsoleReporter(),
                JSONFileReporter(eval_cfg.report.path),
                LoggingReporter(),
                DVCLiveReporter(
                    eval_cfg=eval_cfg,
                    y_true=result.y_val.to_numpy(),
                    y_proba=y_proba,
                    model_name=cfg.model.name,
                    split_strategy=cfg.split.strategy,
                ),
                MLflowReporter(
                    mlflow_cfg=mlflow_cfg,
                    eval_cfg=eval_cfg,
                    model_name=cfg.model.name,
                    split_strategy=cfg.split.strategy,
                ),
            ]

            report = evaluate(
                model_name=cfg.model.name,
                y_true=result.y_val.to_numpy(),
                y_proba=y_proba,
                model=model,
                pipeline=result.pipeline,
                X_train=result.X_train,
                y_train=result.y_train,
                X_val=result.X_val,
                feature_names=result.feature_names,
                split_strategy=cfg.split.strategy,
                eval_cfg=eval_cfg,
                reporters=reporters,
            )

        with step("PHASE 6: Save artifacts + log model"):
            full_pipeline = Pipeline([("features", result.pipeline), ("model", model)])
            artifact_dir = Path(cfg.model.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            pipeline_path = artifact_dir / "pipeline.joblib"
            joblib.dump(full_pipeline, pipeline_path)
            size_mb = pipeline_path.stat().st_size / (1024 * 1024)
            logger.info("  Pipeline saved: %s (%.1f MB)", pipeline_path, size_mb)

            proba_path = artifact_dir / "val_proba.npy"
            yval_path = artifact_dir / "y_val.npy"
            np.save(proba_path, y_proba)
            np.save(yval_path, result.y_val.values.astype("int32"))
            mlflow.log_artifact(str(proba_path))
            mlflow.log_artifact(str(yval_path))
            logger.info(
                "  Saved val predictions for cross-run comparison (seed bagging)"
            )

            _log_configs_and_env(cfg)

            mlflow.log_artifact(str(log_path))

            if Path("dvc.lock").exists():
                lock_hash = hashlib.md5(Path("dvc.lock").read_bytes()).hexdigest()
                mlflow.log_param("dataset_hash", lock_hash)
                mlflow.set_tag("dataset_hash", lock_hash)

            model_info = None
            if mlflow_cfg.log_model:
                # The full pipeline (features + model) is served with RAW
                # transactions (see api loader.predict_proba), so signature and
                # input_example must describe the pre-FE frame, not X_val_fe.
                # pandas CategoricalDtype breaks MLflow schema serialization —
                # cast to object to match the serving contract (CategoryEncoder
                # handles both object and category).
                sig_raw = result.X_val_raw.copy()
                cat_cols = sig_raw.select_dtypes(include="category").columns
                if len(cat_cols):
                    sig_raw[cat_cols] = sig_raw[cat_cols].astype("object")
                sig_input = sig_raw.head(50)
                sig_output = y_pred[: len(sig_input)].astype("int64")
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Hint: Inferred schema contains integer column",
                    )
                    logging.getLogger("mlflow").setLevel(logging.ERROR)
                    signature = infer_signature(sig_input, sig_output)
                    model_info = log_model(
                        full_pipeline,
                        name="model",
                        signature=signature,
                        input_example=sig_raw.head(5),
                    )
                logger.info("  ✓ Model logged to MLflow")

            _run_native_evaluation(
                mlflow_cfg, model_info, result.X_val_raw, result.y_val
            )

            if mlflow_cfg.registry.enabled:
                _register_model(
                    mlflow_cfg,
                    auc=report.roc_auc,
                    run_id=run_id,
                    model_uri=model_info.model_uri if model_info else None,
                    average_precision=report.average_precision,
                    cfg_model_name=cfg.model.name,
                )

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
