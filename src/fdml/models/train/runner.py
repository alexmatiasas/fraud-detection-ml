from __future__ import annotations

import hashlib
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from fdml.features.factory import load_fe_config as load_features_config
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
from fdml.models.train.artifacts import _run_native_evaluation
from fdml.models.train.data_prep import (
    _cap_train_fold,  # noqa: F401
    _get_splitter,  # noqa: F401
    _prepare_data,
    _sample_eval_set,  # noqa: F401
    featurize,  # noqa: F401
    fit_transform_steps,  # noqa: F401
    load_split,  # noqa: F401
    transform_steps,  # noqa: F401
)
from fdml.models.train.fitting import _build_callbacks, _fit_model
from fdml.models.train.model_builder import model_builder_registry
from fdml.models.train.registry import _passes_quality_gate, _register_model  # noqa: F401
from fdml.models.train.tracking import (
    _log_configs_and_env,
    _log_git_tags,
    _register_abort_handler,
    features_fingerprint,
)
from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.mlflow import MlflowFullConfig
from fdml.schemas.train import TrainConfig
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
    cfg: TrainConfig
    params: dict[str, Any]
    feature_names: list[str]


def train(
    cfg: TrainConfig | None = None,
    cli_args: list[str] | None = None,
    callbacks: list | None = None,
    hpo_strategy: Any | None = None,
    mlflow_cfg: MlflowFullConfig | None = None,
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
        model = _fit_model(model, X_train_fe, y_train, X_val_es, y_val_es, cfg, callbacks)

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


def _evaluate_and_track(
    cfg: TrainConfig,
    eval_cfg: EvaluateConfig,
    mlflow_cfg: MlflowFullConfig,
    result: TrainResult,
    train_elapsed_s: float,
) -> tuple:
    """PHASE 5: metrics + MLflow tracking + reporters. Returns (report, y_proba, y_pred, model)."""
    with step("PHASE 5: Evaluation + MLflow tracking"):
        mlflow.log_params(result.params)
        mlflow.log_params({"seed": cfg.seed})
        mlflow.log_metric("training_elapsed_s", round(train_elapsed_s, 1))

        for phase_name, elapsed in get_phase_timings().items():
            match = re.search(r"PHASE (\d+)", phase_name)
            key = f"phase_{match.group(1)}_seconds" if match else f"phase_{phase_name}_seconds"
            mlflow.log_metric(key, round(elapsed, 1))

        if cfg.split.strategy == "temporal" and cfg.split.time_col in result.X_train.columns:
            mlflow.log_params(
                {
                    "split_embargo_seconds": cfg.split.embargo_seconds,
                    "train_dt_min": int(result.X_train[cfg.split.time_col].min()),
                    "train_dt_max": int(result.X_train[cfg.split.time_col].max()),
                    "val_dt_min": int(result.X_val[cfg.split.time_col].min()),
                    "val_dt_max": int(result.X_val[cfg.split.time_col].max()),
                }
            )

        builder = model_builder_registry.get(cfg.model.name)
        best_iter = builder.get_best_iteration(result.model)
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

    return report, y_proba, y_pred, model


def _log_and_save_artifacts(
    cfg: TrainConfig,
    mlflow_cfg: MlflowFullConfig,
    result: TrainResult,
    run_id: str,
    model: Any,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    report: Any,
    log_path: Path,
) -> None:
    """PHASE 6: persist pipeline, log model, native eval, registry."""
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
        logger.info("  Saved val predictions for cross-run comparison (seed bagging)")

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
                sig_raw[cat_cols] = sig_raw[cat_cols].astype(object)
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

        _run_native_evaluation(mlflow_cfg, model_info, result.X_val_raw, result.y_val)

        if mlflow_cfg.registry.enabled:
            _register_model(
                mlflow_cfg,
                auc=report.roc_auc,
                run_id=run_id,
                model_uri=model_info.model_uri if model_info else None,
                average_precision=report.average_precision,
                cfg_model_name=cfg.model.name,
            )


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
    except Exception as exc:  # noqa: BLE001
        logger.warning("  MLflow: experiment setup failed (%s), falling back to local SQLite", exc)
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

        report, y_proba, y_pred, model = _evaluate_and_track(
            cfg, eval_cfg, mlflow_cfg, result, train_elapsed_s
        )

        _log_and_save_artifacts(
            cfg,
            mlflow_cfg,
            result,
            run_id,
            model,
            y_proba,
            y_pred,
            report,
            log_path,
        )

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
