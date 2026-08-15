"""Model metadata and (protected) reload endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fdml.api.dependencies import get_loader, get_model_loader, verify_api_key
from fdml.api.internal.loader import ModelLoader
from fdml.api.schemas.model import ModelInfo

model_router = APIRouter(prefix="/model", tags=["model"])

_METRIC_KEYS = (
    "roc_auc",
    "average_precision",
    "f1",
    "precision",
    "recall",
    "best_f1",
    "best_threshold",
    "brier",
    "expected_cost",
    "fraud_rate",
)


@model_router.get("/info", response_model=ModelInfo)
def info(loader: ModelLoader = Depends(get_loader)) -> ModelInfo:
    """Current model version, source, and evaluation metrics."""
    report = loader.report
    return ModelInfo(
        model_name=report.get("model_name"),
        source=loader.source,
        run_id=loader.run_id,
        version=loader.version,
        loaded_at=loader.loaded_at.isoformat() if loader.loaded_at else None,
        n_features=report.get("n_features"),
        metrics={key: report[key] for key in _METRIC_KEYS if key in report},
    )


@model_router.put("/reload")
def reload(
    loader: ModelLoader = Depends(get_model_loader),
    _: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Reload the model from the configured source (MLflow → local)."""
    loader.load(force=True)
    return {
        "status": "reloaded",
        "source": loader.source or "",
        "loaded_at": loader.loaded_at.isoformat() if loader.loaded_at else "",
    }
