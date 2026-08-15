"""Offline evaluation metrics from models/report.json."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from fdml.api.dependencies import get_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX

metrics_router = APIRouter(prefix=f"{API_PREFIX}/metrics", tags=["metrics"])

_METRIC_KEYS = (
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


@metrics_router.get("/", response_model=dict[str, Any])
@limiter.limit("120/minute")
def metrics(
    request: Request, loader: ModelLoader = Depends(get_loader)
) -> dict[str, Any]:
    """Evaluation metrics from the offline report (models/report.json)."""
    return {key: loader.report[key] for key in _METRIC_KEYS if key in loader.report}
