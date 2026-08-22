"""Shared evaluation-metric helpers for the API routers."""

from __future__ import annotations

from typing import Any

DEFAULT_THRESHOLD = 0.5

#: Keys surfaced from the offline evaluation report (``report.json``).
METRIC_KEYS = (
    "model_name",
    "split_strategy",
    "n_features",
    "n_train",
    "n_val",
    "fraud_rate",
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "ci_lower",
    "ci_upper",
    "best_threshold",
    "best_f1",
    "brier",
    "f_beta",
    "cost_best_threshold",
    "expected_cost",
    "recall_at_k",
)


def metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Subset of the report's metric keys, skipping anything absent."""
    return {key: report[key] for key in METRIC_KEYS if key in report}
