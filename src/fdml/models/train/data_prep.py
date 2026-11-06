from __future__ import annotations

import logging
import time
from typing import cast

import pandas as pd
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
from fdml.models.evaluate.reporter import log_dataset_lineage
from fdml.models.split import StratifiedSplitter, TemporalSplitter
from fdml.schemas.features import FeaturesConfig
from fdml.schemas.mlflow import MlflowFullConfig
from fdml.schemas.train import SplitCfg, TrainConfig
from fdml.utils.logging import step

logger = logging.getLogger(__name__)

_TARGET = "isFraud"


def _get_splitter(split_cfg: SplitCfg) -> TemporalSplitter | StratifiedSplitter:
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


def fit_transform_steps(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
    cfg: TrainConfig, mlflow_cfg: MlflowFullConfig | None = None
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
        logger.info("  Train: %s rows (fraud %.2f%%)", f"{len(X_train):,}", y_train.mean() * 100)
        logger.info("  Val:   %s rows (fraud %.2f%%)", f"{len(X_val):,}", y_val.mean() * 100)

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
    cfg: TrainConfig,
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
    features_cfg: FeaturesConfig,
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
                f"{k}: {v}" for k, v in df.dtypes.apply(lambda x: x.name).value_counts().items()
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


def _prepare_data(cfg: TrainConfig, mlflow_cfg: MlflowFullConfig | None = None) -> tuple:
    """PHASES 1-3: load, split, feature engineering."""

    X_train, X_val, y_train, y_val = load_split(cfg, mlflow_cfg)
    X_train_fe, X_val_fe, pipeline, feature_names = featurize(
        X_train, X_val, y_train, load_features_config()
    )
    X_val_es, y_val_es = _sample_eval_set(X_val_fe, y_val, cfg.early_stopping.eval_max_rows)

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
