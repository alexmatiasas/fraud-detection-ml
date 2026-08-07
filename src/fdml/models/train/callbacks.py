from __future__ import annotations

import logging
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


class IterationCallback:
    """Logs per-iteration validation metrics to MLflow and/or DVCLive.

    Compatible with both LightGBM and XGBoost callback APIs.
    LightGBM env.evaluation_result_list: (dataset, metric, value, higher_better)
    XGBoost  env.evaluation_result_list: (dataset, metric, value)
    """

    def __init__(self, log_mlflow: bool = True, log_dvclive: bool = False):
        self._log_mlflow = log_mlflow
        self._live = None
        if log_dvclive:
            from dvclive import Live

            self._live = Live(dvcyaml=False, report="notebook")

    def __call__(self, env: Any) -> None:
        if not hasattr(env, "evaluation_result_list") or not env.evaluation_result_list:
            return

        iteration = env.iteration

        for item in env.evaluation_result_list:
            if len(item) == 4:
                dataset_name, metric_name, value, _ = item
            elif len(item) == 3:
                dataset_name, metric_name, value = item
            else:
                continue

            key = f"{dataset_name}_{metric_name}"

            if self._log_mlflow and mlflow.active_run() is not None:
                try:
                    mlflow.log_metric(key, value, step=iteration)
                except Exception:
                    pass

            if self._live is not None:
                try:
                    self._live.log_metric(key, value)
                    self._live.next_step()
                except Exception:
                    pass

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.make_summary()
            except Exception:
                pass
