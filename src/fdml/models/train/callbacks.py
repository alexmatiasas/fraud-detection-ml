from __future__ import annotations

import logging
from typing import Any

import mlflow
from xgboost.callback import TrainingCallback

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


def _metric_key(dataset_name: str, metric_name: str) -> str:
    """MLflow metric name for a (dataset, metric) pair.

    Validation and training metrics get the conventional ``val/`` / ``train/``
    prefixes so they group together in the MLflow UI; other datasets keep
    ``dataset_metric``.
    """
    if dataset_name.lower() in {"validation", "valid", "val"}:
        return f"val/{metric_name}"
    if dataset_name.lower() in {"train", "training"}:
        return f"train/{metric_name}"
    return f"{dataset_name}_{metric_name}"


def _normalize_dataset_name(data_name: str) -> str:
    """Map xgboost's auto-generated ``validation_0`` to ``validation``."""
    if len(data_name) >= 2 and data_name[-2] == "_" and data_name[-1].isdigit():
        return data_name[:-2]
    return data_name


class _MetricLogger:
    """Shared per-iteration logging for LightGBM and XGBoost callbacks.

    Args:
        log_mlflow: Log metrics to the active MLflow run.
        log_dvclive: Log metrics to DVCLive.
        log_console: Print a progress line every ``console_every`` iterations
            plus a line each time a metric reaches a new best value.
        console_every: Print cadence when ``log_console`` is enabled.
        mlflow_every: Send one MLflow metric per this many iterations. Remote
            log_metric calls (DagsHub) cost ~1.5s each, so throttling keeps
            experiments fast without losing the curve shape.
    """

    def __init__(
        self,
        log_mlflow: bool = True,
        log_dvclive: bool = False,
        log_console: bool = False,
        console_every: int = 50,
        mlflow_every: int = 1,
    ):
        self._log_mlflow = log_mlflow
        self._log_console = log_console
        self._console_every = max(1, console_every)
        self._mlflow_every = max(1, mlflow_every)
        self._best: dict[str, tuple[float, int]] = {}
        self._live = None
        if log_dvclive:
            from dvclive.live import Live

            self._live = Live(dvcyaml=False, report="notebook")

    def _log_metric(
        self, dataset_name: str, metric_name: str, value: float, step: int
    ) -> None:
        key = _metric_key(dataset_name, metric_name)

        if (
            self._log_mlflow
            and step % self._mlflow_every == 0
            and mlflow.active_run() is not None
        ):
            try:
                mlflow.log_metric(key, value, step=step)
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

    def _log_console_line(
        self, iteration: int, parsed: list[tuple[str, str, float, bool]]
    ) -> None:
        multiple = len(parsed) > 1
        parts: list[str] = []
        for dataset_name, metric_name, value, higher in parsed:
            key = _metric_key(dataset_name, metric_name)
            best_value, best_iter = self._best.get(key, (None, None))
            is_best = best_value is None or (
                value > best_value if higher else value < best_value
            )
            if is_best:
                self._best[key] = (value, iteration)
                best_value, best_iter = value, iteration

            if iteration % self._console_every == 0 or is_best:
                label = f"{dataset_name}:{metric_name}" if multiple else metric_name
                parts.append(
                    f"{label}={value:.4f} (best {best_value:.4f} @ {best_iter})"
                )

        if parts:
            logger.info("  iter %5d | %s", iteration, " | ".join(parts))


class IterationCallback(_MetricLogger):
    """Per-iteration metric logging for LightGBM (plain callable protocol).

    ``__call__(env)`` reads ``env.evaluation_result_list`` where each item is
    ``(dataset, metric, value)`` or ``(dataset, metric, value, higher_better)``.
    """

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
            self._log_metric(dataset_name, metric_name, float(value), iteration)

        if self._log_console:
            self._log_console_line(iteration, parsed)

    def as_xgboost(
        self, name_map: dict[str, str] | None = None
    ) -> XGBoostIterationCallback:
        """Return an equivalent callback for the XGBoost >= 3.x protocol.

        ``name_map`` maps XGBoost's auto-generated eval names (``validation_0``,
        ``validation_1``, ...) to their logical dataset names (``validation``,
        ``train``). Required when more than one eval set is passed, otherwise
        every dataset would normalize to ``validation``.
        """
        return XGBoostIterationCallback(
            log_mlflow=self._log_mlflow,
            log_dvclive=self._live is not None,
            log_console=self._log_console,
            console_every=self._console_every,
            mlflow_every=self._mlflow_every,
            name_map=name_map,
        )


class XGBoostIterationCallback(_MetricLogger, TrainingCallback):
    """Per-iteration metric logging for XGBoost >= 3.x.

    ``after_iteration(model, epoch, evals_log)`` reads the ``evals_log`` dict
    (``data_name -> metric -> [value per epoch]``). Epochs are 0-indexed in
    XGBoost; MLflow steps are 1-indexed for consistency with LightGBM.
    """

    def __init__(
        self,
        log_mlflow: bool = True,
        log_dvclive: bool = False,
        log_console: bool = False,
        console_every: int = 50,
        mlflow_every: int = 1,
        name_map: dict[str, str] | None = None,
    ):
        _MetricLogger.__init__(
            self,
            log_mlflow=log_mlflow,
            log_dvclive=log_dvclive,
            log_console=log_console,
            console_every=console_every,
            mlflow_every=mlflow_every,
        )
        TrainingCallback.__init__(self)
        self._name_map = name_map

    def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
        parsed: list[tuple[str, str, float, bool]] = []
        for data_name, metrics in evals_log.items():
            dataset_name = (
                self._name_map[data_name]
                if self._name_map and data_name in self._name_map
                else _normalize_dataset_name(data_name)
            )
            for metric_name, values in metrics.items():
                if not values:
                    continue
                value = float(values[epoch])
                parsed.append((dataset_name, metric_name, value, True))
                self._log_metric(dataset_name, metric_name, value, epoch + 1)

        if self._log_console:
            self._log_console_line(epoch + 1, parsed)
        return False
