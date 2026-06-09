from __future__ import annotations

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline

from src.features.factory import (
    create_pipeline,
    load_config as load_features_config,
    load_data,
    load_data_config,
    merge_tables,
)
from src.models.categoricals import encode_categoricals as _encode_categoricals
from src.models.config import (
    load_evaluation_config,
    load_mlflow_config,
    load_train_config,
)
from src.models.evaluate.reporter import (
    ConsoleReporter,
    JSONFileReporter,
    LoggingReporter,
)
from src.models.evaluate.runner import evaluate
from src.models.split import StratifiedSplitter, TemporalSplitter
from src.models.train.model_builder import model_builder_registry

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


def train(
    cfg: Any = None,
    cli_args: list[str] | None = None,
) -> TrainResult:
    if cfg is None:
        cfg = load_train_config(cli_args=cli_args)

    features_cfg = load_features_config()
    data_cfg = load_data_config()

    logger.info("=" * 56)
    logger.info("  PHASE 1: Load data")
    logger.info("=" * 56)
    train_df, identity_df = load_data(data_cfg)
    df = merge_tables(train_df, identity_df)
    logger.info("  Merged: %s rows x %s cols", f"{df.shape[0]:,}", df.shape[1])
    logger.info(
        "  Target: %.2f%% fraud (%.0f positives)",
        df[_TARGET].mean() * 100,
        df[_TARGET].sum(),
    )

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 2: Train / validation split")
    logger.info("=" * 56)
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

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 3: Feature engineering")
    logger.info("=" * 56)
    pipeline, _ = create_pipeline(features_cfg)
    X_train_fe = pipeline.fit_transform(X_train, y_train)
    X_val_fe = pipeline.transform(X_val)

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

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 4: Model training")
    logger.info("=" * 56)

    builder = model_builder_registry.get(cfg.model.name)
    logger.info("Model: %s", cfg.model.name)
    logger.info("  params: %s", builder.format_params(cfg.model.params.model_dump()))

    model = builder.build(cfg.model.params.model_dump())
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
            )
        elif isinstance(model, xgb.XGBClassifier):
            model.fit(X_train_fe, y_train, eval_set=eval_set, verbose=False)
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
        feature_names=X_train_fe.columns.tolist(),
    )


def main() -> None:
    cfg = load_train_config()
    eval_cfg = load_evaluation_config()
    mlflow_cfg = load_mlflow_config()

    result = train(cfg)

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 5: Evaluation")
    logger.info("=" * 56)
    model = result.model
    y_proba = model.predict_proba(result.X_val)[:, 1]
    y_pred = model.predict(result.X_val)

    # Reporters for evaluation output
    eval_reporters = [
        ConsoleReporter(),
        JSONFileReporter(eval_cfg.report.path),
        LoggingReporter(),
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
        reporters=eval_reporters,
    )

    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 6: Save artifacts")
    logger.info("=" * 56)
    full_pipeline = Pipeline([("features", result.pipeline), ("model", model)])
    artifact_dir = Path(cfg.model.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = artifact_dir / "pipeline.joblib"
    joblib.dump(full_pipeline, pipeline_path)
    size_mb = pipeline_path.stat().st_size / (1024 * 1024)
    logger.info("  Pipeline saved: %s (%.1f MB)", pipeline_path, size_mb)

    mlflow.set_experiment(mlflow_cfg.tracking.experiment_name)
    with mlflow.start_run(run_name=mlflow_cfg.tracking.run_name):
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "split_strategy": cfg.split.strategy,
                "split_test_size": cfg.split.test_size,
            }
        )
        for key, value in cfg.model.params.model_dump().items():
            mlflow.log_param(f"model_{key}", value)
        mlflow.log_metric("roc_auc", report.roc_auc)
        mlflow.log_metric("average_precision", report.average_precision)
        mlflow.log_metric("f1", report.f1)
        mlflow.log_metric("best_threshold", report.best_threshold)

        if Path("dvc.lock").exists():
            lock_hash = hashlib.md5(Path("dvc.lock").read_bytes()).hexdigest()
            mlflow.log_param("dataset_hash", lock_hash)

        if mlflow_cfg.log_model:
            sig_input = result.X_val.astype(
                {c: "float64" for c in result.X_val.select_dtypes("int").columns}
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Hint: Inferred schema contains integer column"
                )
                logging.getLogger("mlflow").setLevel(logging.ERROR)
                signature = infer_signature(sig_input, y_pred.astype("int64"))
                mlflow.sklearn.log_model(full_pipeline, "model", signature=signature)
            logger.info("  ✓ Model logged to MLflow")

    logger.info("")
    logger.info("Done.")


if __name__ == "__main__":
    main()
