from __future__ import annotations

import hashlib
import logging
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from pydantic import BaseModel, ConfigDict
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
from fdml.models.categoricals import (
    encode_categoricals as _encode_categoricals,
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
)
from fdml.models.evaluate.runner import evaluate
from fdml.models.split import StratifiedSplitter, TemporalSplitter
from fdml.models.train.model_builder import model_builder_registry
from fdml.utils.logging import ensure_logging, setup_logging, step

logger = logging.getLogger(__name__)

_TARGET = "isFraud"


class TrainResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    cfg: Any
    params: dict[str, Any]
    feature_names: list[str]


def _get_splitter(split_cfg: Any) -> TemporalSplitter | StratifiedSplitter:
    if split_cfg.strategy == "temporal":
        return TemporalSplitter(
            time_col=split_cfg.time_col, test_size=split_cfg.test_size
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


def load_split(cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
        train_idx, val_idx = next(splitter.split(df, df[_TARGET]))

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

    return X_train, X_val, y_train, y_val


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

        logger.info("  dtypes (pre-encode): %s", _fmt_dtypes(X_train_fe))
        X_train_fe, X_val_fe = _encode_categoricals(X_train_fe, X_val_fe)
        logger.info("  Final shape: train=%s, val=%s", X_train_fe.shape, X_val_fe.shape)
        logger.info("  dtypes: %s", _fmt_dtypes(X_train_fe))

    return X_train_fe, X_val_fe, pipeline, X_train_fe.columns.tolist()


def _prepare_data(cfg: Any) -> tuple:
    """PHASES 1-3: load, split, feature engineering."""

    X_train, X_val, y_train, y_val = load_split(cfg)
    X_train_fe, X_val_fe, pipeline, feature_names = featurize(
        X_train, X_val, y_train, load_features_config()
    )

    return X_train_fe, X_val_fe, y_train, y_val, pipeline, feature_names


def train(
    cfg: Any = None,
    cli_args: list[str] | None = None,
    callbacks: list | None = None,
    hpo_strategy: Any = None,
) -> TrainResult:
    ensure_logging()

    if cfg is None:
        cfg = load_train_config(cli_args=cli_args)

    X_train_fe, X_val_fe, y_train, y_val, pipeline, feature_names = _prepare_data(cfg)

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
        use_es = cfg.early_stopping.enabled and hasattr(model, "early_stopping_rounds")

        if use_es:
            logger.info(
                "  Early stopping: %d rounds on %s",
                cfg.early_stopping.rounds,
                cfg.early_stopping.eval_metric,
            )

        import lightgbm as lgb  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415

        eval_set = [(X_val_fe, y_val)] if use_es else None

        if eval_set is not None:
            if isinstance(model, lgb.LGBMClassifier):
                model.fit(
                    X_train_fe,
                    y_train,
                    eval_set=eval_set,
                    eval_names=["validation"],
                    eval_metric=cfg.early_stopping.eval_metric,
                    callbacks=callbacks or None,
                )
            elif isinstance(model, xgb.XGBClassifier):
                model.fit(
                    X_train_fe,
                    y_train,
                    eval_set=eval_set,
                    verbose=False,
                    callbacks=callbacks or None,
                )
            else:
                model.fit(X_train_fe, y_train)
        else:
            model.fit(X_train_fe, y_train)

        logger.info("  ✓ Training complete")

    return TrainResult(
        model=model,
        pipeline=pipeline,
        X_train=X_train_fe,
        X_val=X_val_fe,
        y_train=y_train,
        y_val=y_val,
        cfg=cfg,
        params=params,
        feature_names=feature_names,
    )


def _register_model(mlflow_cfg: Any, auc: float, run_id: str) -> None:
    from mlflow import MlflowClient

    client = MlflowClient()
    model_name = mlflow_cfg.registry.model_name

    try:
        client.create_registered_model(model_name)
        logger.info("  Registry: created model '%s'", model_name)
    except Exception:
        pass

    model_uri = f"runs:/{run_id}/model"
    try:
        mv = client.create_model_version(model_name, model_uri, run_id)
        version = mv.version
        logger.info("  Registry: created version %s (AUC=%.4f)", version, auc)

        try:
            champion_mv = client.get_model_version_by_alias(model_name, "champion")
            champion_run = client.get_run(champion_mv.run_id)
            champion_auc = champion_run.data.metrics.get("roc_auc", 0.0)

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


def _build_callbacks(cfg: Any) -> list | None:
    if not cfg.training_callbacks.log_per_iteration:
        return None
    from fdml.models.train.callbacks import IterationCallback

    return [
        IterationCallback(
            log_mlflow=True,
            log_dvclive=cfg.training_callbacks.dvclive,
            log_console=True,
        )
    ]


def _log_configs_and_env() -> None:
    for config_path in [
        "configs/train.yaml",
        "configs/features.yaml",
        "configs/evaluate.yaml",
        "configs/mlflow.yaml",
    ]:
        if Path(config_path).exists():
            mlflow.log_artifact(config_path, "configs")
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


def main() -> None:
    cfg = load_train_config()
    run_name = f"{cfg.model.name}_s{cfg.seed}"
    log_path = setup_logging(log_path=f"train_{run_name}.log")

    if cfg.ablation.enabled:
        from fdml.models.train.ablation import run_ablation

        results = run_ablation(cfg, max_train_rows=cfg.ablation.max_train_rows)
        report_path = Path(cfg.ablation.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(report_path, index=False)
        logger.info("Ablation report saved to %s", report_path)
        logger.info("\n%s", results.to_string(index=False))
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

        _log_git_tags()

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
        result = train(cfg, callbacks=callbacks, hpo_strategy=hpo_strategy)
        train_elapsed_s = time.perf_counter() - train_start

        with step("PHASE 5: Evaluation + MLflow tracking"):
            mlflow.log_params(result.params)
            mlflow.log_params(
                {"seed": cfg.seed, "training_elapsed_s": round(train_elapsed_s, 1)}
            )

            if (
                cfg.split.strategy == "temporal"
                and cfg.split.time_col in result.X_train.columns
            ):
                mlflow.log_params(
                    {
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
                mlflow.log_param("best_iteration", int(best_iter))
            model = result.model
            y_proba = model.predict_proba(result.X_val)[:, 1]
            y_pred = model.predict(result.X_val)

            reporters = [
                ConsoleReporter(),
                JSONFileReporter(eval_cfg.report.path),
                LoggingReporter(),
                DVCLiveReporter(
                    eval_cfg=eval_cfg,
                    y_true=result.y_val.values,
                    y_proba=y_proba,
                    model_name=cfg.model.name,
                    split_strategy=cfg.split.strategy,
                ),
                MLflowReporter(
                    mlflow_cfg=mlflow_cfg,
                    eval_cfg=eval_cfg,
                    X_train=result.X_train,
                    y_train=result.y_train,
                    model_name=cfg.model.name,
                    split_strategy=cfg.split.strategy,
                ),
            ]

            report = evaluate(
                model_name=cfg.model.name,
                y_true=result.y_val.values,
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

            _log_configs_and_env()

            mlflow.log_artifact(str(log_path))

            if Path("dvc.lock").exists():
                lock_hash = hashlib.md5(Path("dvc.lock").read_bytes()).hexdigest()
                mlflow.log_param("dataset_hash", lock_hash)
                mlflow.set_tag("dataset_hash", lock_hash)

            if mlflow_cfg.log_model:
                sig_input = result.X_val.astype(
                    {c: "float64" for c in result.X_val.select_dtypes("int").columns}
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Hint: Inferred schema contains integer column",
                    )
                    logging.getLogger("mlflow").setLevel(logging.ERROR)
                    signature = infer_signature(sig_input, y_pred.astype("int64"))
                    mlflow.sklearn.log_model(
                        full_pipeline, "model", signature=signature
                    )
                logger.info("  ✓ Model logged to MLflow")

            if mlflow_cfg.registry.enabled:
                _register_model(
                    mlflow_cfg,
                    auc=report.roc_auc,
                    run_id=run_id,
                )

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
