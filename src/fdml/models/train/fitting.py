from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fdml.models.train.model_builder import model_builder_registry
from fdml.schemas.train import TrainConfig

logger = logging.getLogger(__name__)


def _fit_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: TrainConfig,
    callbacks: list | None = None,
) -> Any:
    """Fit ``model`` with an evaluation set and native early stopping.

    LightGBM 4.x removed the ``early_stopping_rounds`` estimator attribute, so
    early stopping is applied through the native ``lgb.early_stopping`` /
    ``xgb.callback.EarlyStopping`` callbacks instead of estimator parameters.

    Delegates to ``ModelBuilder.fit`` — the per-framework early-stopping
    wiring lives there, keeping this function a thin orchestrator of the
    eval-set / train-curve policy.
    """
    use_es = cfg.early_stopping.enabled
    if use_es:
        logger.info(
            "  Early stopping: %d rounds on %s",
            cfg.early_stopping.rounds,
            cfg.early_stopping.eval_metric,
        )

    if not use_es:
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
                np.random.default_rng(cfg.seed).choice(n_train, train_rows, replace=False)
            )
        else:
            idx = np.arange(n_train)
        train_eval = (X_train.iloc[idx], y_train.iloc[idx])
        logger.info(
            "  Train eval: %s rows sampled for the per-iteration train curve",
            f"{len(idx):,}",
        )

    builder = model_builder_registry.get(cfg.model.name)
    is_xgb = isinstance(model, xgb.XGBClassifier)

    eval_set: list[tuple[pd.DataFrame, pd.Series]] = [(X_val, y_val)]
    eval_names = ["validation"]
    xgb_name_map: dict[str, str] | None = None
    if train_eval:
        if is_xgb:
            # XGBoost early-stops on the LAST eval dataset, so validation goes
            # last; the name_map keeps train/validation distinguishable.
            eval_set = [train_eval, (X_val, y_val)]
            xgb_name_map = {"validation_0": "train", "validation_1": "validation"}
        else:
            eval_set = [*eval_set, train_eval]
            eval_names.append("train")
    else:
        xgb_name_map = {"validation_0": "validation"} if is_xgb else None

    builder.fit(
        model,
        X_train,
        y_train,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_metric=cfg.early_stopping.eval_metric,
        callbacks=callbacks,
        es_rounds=cfg.early_stopping.rounds,
        **({"xgb_name_map": xgb_name_map} if is_xgb else {}),
    )
    return model


def _build_callbacks(cfg: TrainConfig) -> list | None:
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
