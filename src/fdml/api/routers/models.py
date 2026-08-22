"""Registered-model listing and per-model detail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from fdml.api.dependencies import get_multi_loader
from fdml.api.internal.registry import MultiModelLoader, RegisteredModel
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.models import (
    FeatureImportance,
    ModelList,
    RegisteredModelInfo,
)

models_router = APIRouter(prefix=f"{API_PREFIX}/models", tags=["model"])

TOP_FEATURES = 20


def _to_info(multi: MultiModelLoader, info: RegisteredModel) -> RegisteredModelInfo:
    top_features = [
        FeatureImportance(feature=item["feature"], importance=float(item["importance"]))
        for item in info.report.get("top_features", [])[:TOP_FEATURES]
    ]
    return RegisteredModelInfo(
        name=info.name,
        version=info.version,
        status=info.status,
        run_id=info.run_id,
        aliases=info.aliases,
        active=multi.is_default(info.name),
        loaded=multi.is_loaded_model(info.name),
        n_features=info.n_features,
        best_threshold=info.best_threshold,
        metrics=info.metric_summary(),
        top_features=top_features or None,
        error=info.error,
    )


@models_router.get("/", response_model=ModelList)
@limiter.limit("120/minute")
def list_models(
    request: Request, multi: MultiModelLoader = Depends(get_multi_loader)
) -> ModelList:
    """All registered models with their evaluation metrics (leaderboard)."""
    return ModelList(
        default=multi.default_model,
        models=[_to_info(multi, info) for info in multi.iter_models()],
    )


@models_router.get("/{name}", response_model=RegisteredModelInfo)
@limiter.limit("120/minute")
def model_detail(
    request: Request,
    name: str,
    multi: MultiModelLoader = Depends(get_multi_loader),
) -> RegisteredModelInfo:
    """Detail for a single registered model."""
    info = multi.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not registered")
    return _to_info(multi, info)
