from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features.factory import (
    create_pipeline,
    load_config as load_features_config,
    load_data,
    load_data_config,
    merge_tables,
)
from src.models.config import load_train_config
from src.models.split import StratifiedSplitter, TemporalSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_TARGET = "isFraud"
_RANDOM_STATE = 42


def _get_splitter(split_cfg):
    if split_cfg.strategy == "temporal":
        t = split_cfg.time_col
        s = split_cfg.test_size
        logger.info("Split: temporal (time_col=%s, test_size=%.0f%%)", t, s * 100)
        return TemporalSplitter(time_col=t, test_size=s)
    if split_cfg.strategy == "random":
        s = split_cfg.test_size
        logger.info("Split: stratified random (test_size=%.0f%%)", s * 100)
        return StratifiedSplitter(test_size=s)
    msg = f"Unknown split strategy: {split_cfg.strategy}"
    raise ValueError(msg)


def _build_model(model_cfg):
    name = model_cfg.name
    params = model_cfg.params.model_dump()
    logger.info("Model: %s", name)

    if name == "lightgbm":
        for _k in ("n_jobs", "random_state"):
            params.pop(_k, None)
        logger.info(
            "  params: leaves=%d depth=%d lr=%.3f subsample=%.1f colsample=%.1f "
            "alpha=%.4f lambda=%.1f scale_pos=%.1f",
            params.get("num_leaves", 127),
            params.get("max_depth", 8),
            params.get("learning_rate", 0.05),
            params.get("subsample", 0.8),
            params.get("colsample_bytree", 0.8),
            params.get("reg_alpha", 0.1),
            params.get("reg_lambda", 1.0),
            params.get("scale_pos_weight", 27.6),
        )
        return lgb.LGBMClassifier(
            **params, n_jobs=-1, verbose=-1, random_state=_RANDOM_STATE
        )
    if name == "xgboost":
        logger.info(
            "  params: n_est=%d depth=%d lr=%.3f subsample=%.1f colsample=%.1f "
            "scale_pos=%.1f",
            params.get("n_estimators", 1000),
            params.get("max_depth", 8),
            params.get("learning_rate", 0.05),
            params.get("subsample", 0.8),
            params.get("colsample_bytree", 0.8),
            params.get("scale_pos_weight", 27.6),
        )
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 1000),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 8),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=params.get("scale_pos_weight", 27.6),
            verbosity=0,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "random_forest":
        logger.info(
            "  params: n_est=%d depth=%d min_samples_leaf=%d max_samples=%.1f",
            params.get("n_estimators", 1000),
            params.get("max_depth", 8),
            params.get("min_child_samples", 100),
            params.get("subsample", 0.8),
        )
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 1000),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_child_samples", 100),
            max_samples=params.get("subsample", 0.8),
            max_features=params.get("colsample_bytree", 0.8),
            class_weight="balanced_subsample",
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )

    msg = f"Unknown model: {name}"
    raise ValueError(msg)


def _encode_categoricals(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return X_train, X_val

    logger.info("Encoding %d categorical columns → int32 ordinal codes", len(cat_cols))
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in cat_cols:
        train_vals = X_train[col].astype(str)
        categories = sorted(train_vals.unique())
        cat_to_code = {c: i for i, c in enumerate(categories)}
        X_train[col] = train_vals.map(cat_to_code).astype("int32")

        val_str = X_val[col].astype(str)
        X_val[col] = val_str.map(cat_to_code).fillna(-1).astype("int32")

    return X_train, X_val


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }


def main() -> None:
    cfg = load_train_config()
    features_cfg = load_features_config()
    data_cfg = load_data_config()

    # ═══════════════════════════════════════════════════════════════════════
    #  1. Load & merge
    # ═══════════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════════
    #  2. Split — BEFORE feature engineering
    # ═══════════════════════════════════════════════════════════════════════
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
        "  Train: %s rows (fraud %.2f%%)",
        f"{len(X_train):,}",
        y_train.mean() * 100,
    )
    logger.info(
        "  Val:   %s rows (fraud %.2f%%)",
        f"{len(X_val):,}",
        y_val.mean() * 100,
    )

    if cfg.split.strategy == "temporal" and cfg.split.time_col in X.columns:
        train_time_range = (
            int(X_train[cfg.split.time_col].min()),
            int(X_train[cfg.split.time_col].max()),
        )
        val_time_range = (
            int(X_val[cfg.split.time_col].min()),
            int(X_val[cfg.split.time_col].max()),
        )
        logger.info(
            "  Time range — train: [%d, %d]  val: [%d, %d]",
            train_time_range[0],
            train_time_range[1],
            val_time_range[0],
            val_time_range[1],
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  3. Feature engineering — fit only on training set
    # ═══════════════════════════════════════════════════════════════════════
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
    v_before = X_train.shape[1]
    v_after = X_train_fe.shape[1]
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

    # ═══════════════════════════════════════════════════════════════════════
    #  4. Train model
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 4: Model training")
    logger.info("=" * 56)
    model = _build_model(cfg.model)
    eval_set = None
    if cfg.early_stopping.enabled and hasattr(model, "early_stopping_rounds"):
        logger.info(
            "  Early stopping: %d rounds on %s",
            cfg.early_stopping.rounds,
            cfg.early_stopping.eval_metric,
        )
        eval_set = [(X_val_fe, y_val)]

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
            model.fit(
                X_train_fe,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        else:
            model.fit(X_train_fe, y_train)
    else:
        model.fit(X_train_fe, y_train)

    logger.info("  ✓ Training complete")

    # ═══════════════════════════════════════════════════════════════════════
    #  5. Evaluate
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 5: Evaluation")
    logger.info("=" * 56)
    y_proba = model.predict_proba(X_val_fe)[:, 1]
    y_pred = model.predict(X_val_fe)
    metrics = _compute_metrics(y_val, y_proba)
    logger.info("  ROC AUC:         %.4f", metrics["roc_auc"])
    logger.info("  Avg Precision:   %.4f", metrics["average_precision"])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if importances is not None and len(importances) > 0:
            top_k = min(10, len(importances))
            top_idx = np.argsort(importances)[::-1][:top_k]
            logger.info("  Top-%d features (importance = split gain):", top_k)
            for i, idx in enumerate(top_idx):
                col = X_train_fe.columns[idx]
                logger.info("    %2d. %s (%.4f)", i + 1, col, importances[idx])

    # ═══════════════════════════════════════════════════════════════════════
    #  6. Save pipeline locally
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 56)
    logger.info("  PHASE 6: Save artifacts")
    logger.info("=" * 56)
    full_pipeline = Pipeline([("features", pipeline), ("model", model)])
    artifact_dir = Path(cfg.model.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = artifact_dir / "pipeline.joblib"

    import joblib

    joblib.dump(full_pipeline, pipeline_path)
    size_mb = pipeline_path.stat().st_size / (1024 * 1024)
    logger.info("  Pipeline saved: %s (%.1f MB)", pipeline_path, size_mb)

    # ═══════════════════════════════════════════════════════════════════════
    #  7. MLflow
    # ═══════════════════════════════════════════════════════════════════════
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(
            {
                "model_name": cfg.model.name,
                "split_strategy": cfg.split.strategy,
                "split_test_size": cfg.split.test_size,
            }
        )
        for key, value in cfg.model.params.model_dump().items():
            mlflow.log_param(f"model_{key}", value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        if Path("dvc.lock").exists():
            import hashlib

            lock_hash = hashlib.md5(Path("dvc.lock").read_bytes()).hexdigest()
            mlflow.log_param("dataset_hash", lock_hash)

        if cfg.mlflow.log_model:
            sig_input = X_val_fe.astype(
                {c: "float64" for c in X_val_fe.select_dtypes("int").columns}
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
