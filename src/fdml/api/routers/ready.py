"""Readiness probe — the model is loaded and predictions are possible."""

from fastapi import APIRouter, Depends

from fdml.api.dependencies import get_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.health import ReadyStatus

ready_router = APIRouter(prefix=f"{API_PREFIX}/ready", tags=["ready"])


@ready_router.get("/", response_model=ReadyStatus)
def ready(loader: ModelLoader = Depends(get_loader)) -> ReadyStatus:
    """Readiness probe — 200 when the model is loaded, 503 otherwise."""
    return ReadyStatus(status="ready", model_loaded=loader.is_loaded)
