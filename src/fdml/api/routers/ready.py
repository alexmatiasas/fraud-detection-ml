"""Readiness probe — the model is loaded and predictions are possible."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from fdml.api.dependencies import get_model_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.health import ReadyStatus

ready_router = APIRouter(prefix=f"{API_PREFIX}/ready", tags=["ready"])


@ready_router.get("/", response_model=ReadyStatus)
def ready(
    loader: ModelLoader = Depends(get_model_loader),
) -> ReadyStatus | JSONResponse:
    """Readiness probe — 200 when the model is loaded, 503 otherwise.

    Cloud Run can use this as a startup/readiness probe: an instance whose
    model failed to load (e.g. missing DAGSHUB secrets) returns 503 and never
    receives traffic. ``get_model_loader`` (not ``get_loader``) is used so the
    body keeps the ``model_loaded`` field instead of failing the dependency.
    """
    if not loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content=ReadyStatus(status="not_ready", model_loaded=False).model_dump(),
        )
    return ReadyStatus(status="ready", model_loaded=True)
