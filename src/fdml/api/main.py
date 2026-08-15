"""FastAPI application entrypoint for the fraud detection service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fdml.api.dependencies import get_model_loader
from fdml.api.internal.loader import ModelLoadError
from fdml.api.metadata import DESCRIPTION, SUMMARY, TITLE, VERSION, tags_metadata
from fdml.api.routers.health import health_router
from fdml.api.routers.metrics import metrics_router
from fdml.api.routers.model import model_router
from fdml.api.routers.predict import predict_router
from fdml.api.routers.ready import ready_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and demo sample once at startup."""
    loader = get_model_loader()
    try:
        loader.load()
    except ModelLoadError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    loader.load_sample()
    yield


app = FastAPI(
    title=TITLE,
    summary=SUMMARY,
    description=DESCRIPTION,
    version=VERSION,
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

app.include_router(predict_router)
app.include_router(health_router)
app.include_router(ready_router)
app.include_router(model_router)
app.include_router(metrics_router)


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    """503 with a clear message when no model is reachable."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def read_root() -> dict[str, str]:
    """Service landing page pointing at the interactive docs."""
    return {"service": TITLE, "docs": "/docs"}
