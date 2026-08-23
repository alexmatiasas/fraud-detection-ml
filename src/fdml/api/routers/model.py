"""Model metadata, reload, and switch endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from fdml.api.dependencies import get_loader, get_model_loader, verify_api_key
from fdml.api.internal.loader import ModelLoader
from fdml.api.internal.metrics import metric_summary
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.model import ModelInfo
from fdml.api.schemas.models import ModelSwitchRequest, ModelSwitchResponse

model_router = APIRouter(prefix=f"{API_PREFIX}/model", tags=["model"])


@model_router.get("/info", response_model=ModelInfo)
@limiter.limit("120/minute")
def info(
    request: Request,
    response: Response,
    loader: ModelLoader = Depends(get_loader),
) -> ModelInfo:
    """Current model version, source, and evaluation metrics."""
    response.headers["Cache-Control"] = "public, max-age=120"
    report = loader.report
    return ModelInfo(
        model_name=report.get("model_name"),
        source=loader.source,
        run_id=loader.run_id,
        version=int(loader.version) if loader.version else None,
        loaded_at=loader.loaded_at.isoformat() if loader.loaded_at else None,
        n_features=report.get("n_features"),
        metrics=metric_summary(report),
    )


@model_router.put("/reload")
@limiter.limit("10/minute")
def reload(
    request: Request,
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


@model_router.put("/switch", response_model=ModelSwitchResponse)
@limiter.limit("10/minute")
def switch(
    request: Request,
    req: ModelSwitchRequest,
    loader: ModelLoader = Depends(get_model_loader),
    _: str = Depends(verify_api_key),
) -> ModelSwitchResponse:
    """Hot-swap the served model to a specific registry version."""
    loader.load_version(req.version)
    return ModelSwitchResponse(
        model_name=loader._mlflow_model,
        version=req.version,
        source=loader.source or "",
        loaded_at=loader.loaded_at.isoformat() if loader.loaded_at else "",
        details=metric_summary(loader.report),
    )
