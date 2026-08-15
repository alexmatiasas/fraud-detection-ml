"""Model registry listing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from fdml.api.dependencies import get_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.models import ModelVersionInfo

models_router = APIRouter(prefix=f"{API_PREFIX}/models", tags=["model"])


@models_router.get("/", response_model=list[ModelVersionInfo])
@limiter.limit("120/minute")
def list_models(
    request: Request, loader: ModelLoader = Depends(get_loader)
) -> list[ModelVersionInfo]:
    """List all versions of the registered model in the MLflow registry."""
    return [ModelVersionInfo(**row) for row in loader.list_versions()]
