from __future__ import annotations

import logging
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

_HIGHER_IS_BETTER = {
    "auc",
    "average_precision",
    "accuracy",
    "f1",
    "precision",
    "recall",
}


def _higher_is_better(metric_name: str, default: bool = True) -> bool:
    if metric_name in _HIGHER_IS_BETTER:
        return True
    if metric_name in {"logloss", "log_loss", "error", "mse", "mae", "rmse"}:
        return False
    return default


class IterationCallback:
    """Logs per-iteration validation metrics to MLflow, DVCLive and/or console.

    Compatible with both LightGBM and XGBoost callback APIs.
    LightGBM env.evaluation_result_list: (dataset, metric, value, higher_better)
    XGBoost  env.evaluation_result_list: (dataset, metric, value)

    Args:
        log_mlflow: Log metrics to the active MLflow run.
        log_dvclive: Log metrics to DVCLive.
        log_console: Print a progress line every ``console_every`` iterations
            plus a line each time a metric reaches a new best value.
        console_every: Print cadence when ``log_console`` is enabled.
    """

    def __init__(
        self,
        log_mlflow: bool = True,
        log_dvclive: bool = False,
        log_console: bool = False,
        console_every: int = 50,
    ):
        self._log_mlflow = log_mlflow
        self._log_console = log_console
        self._console_every = max(1, console_every)
        self._best: dict[str, tuple[float, int]] = {}
        self._live = None
        if log_dvclive:
            from dvclive import Live

            self._live = Live(dvcyaml=False, report="notebook")

    def __call__(self, env: Any) -> None:
        if not hasattr(env, "evaluation_result_list") or not env.evaluation_result_list:
            return

        iteration = env.iteration
        parsed: list[tuple[str, str, float, bool]] = []

        for item in env.evaluation_result_list:
            if len(item) == 4:
                dataset_name, metric_name, value, higher = item
            elif len(item) == 3:
                dataset_name, metric_name, value = item
                higher = _higher_is_better(metric_name)
            else:
                continue

            parsed.append((dataset_name, metric_name, float(value), bool(higher)))
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

        if self._log_console:
            self._log_console_line(iteration, parsed)

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.make_summary()
            except Exception:
                pass

    def _log_console_line(
        self, iteration: int, parsed: list[tuple[str, str, float, bool]]
    ) -> None:
        parts: list[str] = []
        for dataset_name, metric_name, value, higher in parsed:
            key = f"{dataset_name}_{metric_name}"
            best_value, best_iter = self._best.get(key, (None, None))
            is_best = best_value is None or (
                value > best_value if higher else value < best_value
            )
            if is_best:
                self._best[key] = (value, iteration)
                best_value, best_iter = value, iteration

            if iteration % self._console_every == 0 or is_best:
                parts.append(
                    f"{metric_name}={value:.4f} (best {best_value:.4f} @ {best_iter})"
                )

        if parts:
            logger.info("  iter %5d | %s", iteration, " | ".join(parts))
