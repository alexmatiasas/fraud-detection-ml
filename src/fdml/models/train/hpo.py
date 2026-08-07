from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
import xgboost as xgb

from fdml.models.train.model_builder import ModelBuilder

logger = logging.getLogger(__name__)


class HPOStrategy(ABC):
    """Strategy for hyperparameter optimization.

    Implementations define how the search space is explored and
    how the best configuration is selected.
    """

    @abstractmethod
    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        builder: ModelBuilder,
        base_params: dict[str, Any],
        es_rounds: int,
        es_metric: str,
        seed: int,
    ) -> dict[str, Any]:
        """Run the search and return the best hyperparameters found."""


class OptunaHPO(HPOStrategy):
    """Optuna-based hyperparameter optimisation with per-trial MLflow logging.

    Each trial is logged as a nested MLflow run under the currently active run.
    The best parameters found are persisted to ``models/best_params_{model}.yaml``.

    Parameters
    ----------
    n_trials : int
        Number of optimisation trials.
    timeout_seconds : int or None
        Time limit for the entire study.
    direction : str
        ``"maximize"`` or ``"minimize"``.
    study_name : str
        Name for the Optuna study (persisted to SQLite DB).
    save_best : bool
        Whether to persist the best parameters to disk.
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout_seconds: int | None = 3600,
        direction: str = "maximize",
        study_name: str = "fraud_study",
        save_best: bool = True,
    ):
        self._n_trials = n_trials
        self._timeout = timeout_seconds
        self._direction = direction
        self._study_name = study_name
        self._save_best = save_best

    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        builder: ModelBuilder,
        base_params: dict[str, Any],
        es_rounds: int,
        es_metric: str,
        seed: int,
    ) -> dict[str, Any]:
        study = optuna.create_study(
            direction=self._direction,
            study_name=self._study_name,
        )

        from optuna_integration import MLflowCallback as _MLflowCallback

        mlflow_cb = _MLflowCallback(
            metric_name="validation_auc",
            mlflow_kwargs={"nested": True},
        )

        logger.info(
            "  Optuna: %s trials, timeout=%s, study='%s'",
            self._n_trials,
            f"{self._timeout}s" if self._timeout else "none",
            self._study_name,
        )

        try:
            study.optimize(
                lambda trial: _objective(
                    trial,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    builder=builder,
                    base_params=base_params,
                    es_rounds=es_rounds,
                    es_metric=es_metric,
                    seed=seed,
                ),
                n_trials=self._n_trials,
                timeout=self._timeout,
                callbacks=[mlflow_cb],
            )
        except Exception as exc:
            logger.warning("  Optuna: study failed with %s", exc)
            return base_params

        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_metrics(
            {"optuna_best_value": best_value, "optuna_n_trials": study.trials.__len__()}
        )

        logger.info("  Optuna: best_value=%.6f, params=%s", best_value, best_params)

        if self._save_best:
            _save_best_params(builder.name(), best_params, best_value)

        return {**base_params, **best_params}


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    builder: ModelBuilder,
    base_params: dict[str, Any],
    es_rounds: int,
    es_metric: str,
    seed: int,
) -> float:
    """Objective function for one Optuna trial."""

    search_space = builder.get_search_space(trial)
    params = {**base_params, **search_space}

    model = builder.build(params)

    eval_set = [(X_val, y_val)]

    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=["validation"],
            eval_metric=es_metric,
            callbacks=[lgb.early_stopping(es_rounds, first_metric_only=True)],
        )
    elif isinstance(model, xgb.XGBClassifier):
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_val, y_proba))


def _save_best_params(model_name: str, params: dict[str, Any], value: float) -> None:
    import json

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"best_params_{model_name}.json"
    payload = {
        "model": model_name,
        "best_value": value,
        "params": {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in params.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("  Best params saved to %s", path)
    mlflow.log_artifact(str(path))
