from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


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
                np.random.default_rng(cfg.seed).choice(n_train, train_rows, replace=False)
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
            eval_set = [*eval_set, train_eval]
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
        logger.warning(
            "  %s does not support early stopping — training without it",
            type(model).__name__,
        )
        model.fit(X_train, y_train)
    return model


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
