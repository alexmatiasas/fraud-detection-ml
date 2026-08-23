"""FastAPI application entrypoint for the fraud detection service."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import sentry_sdk
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from fdml.api.context import request_id as request_id_ctx
from fdml.api.dependencies import get_model_loader, get_multi_loader
from fdml.api.internal.loader import ModelLoadError
from fdml.api.internal.registry import ModelRegistryError
from fdml.api.limiter import limiter, rate_limit_exceeded_handler
from fdml.api.metadata import DESCRIPTION, SUMMARY, TITLE, VERSION, tags_metadata
from fdml.api.routers.features import features_router
from fdml.api.routers.health import health_router
from fdml.api.routers.metrics import metrics_router
from fdml.api.routers.model import model_router
from fdml.api.routers.models import models_router
from fdml.api.routers.predict import predict_router
from fdml.api.routers.ready import ready_router
from fdml.api.routers.transactions import transactions_router

load_dotenv()
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    send_default_pii=True,
    enable_logs=True,
    profile_session_sample_rate=1.0,
    profile_lifecycle="trace",
    traces_sample_rate=0.1,
)

logger = logging.getLogger(__name__)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# In production: only allow the Vercel frontend origin.
# In development: allow localhost for local testing.
DEFAULT_ORIGINS = (
    ["https://alexmatias.vercel.app"]
    if IS_PRODUCTION
    else ["https://alexmatias.vercel.app", "http://localhost:4321"]
)
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model(s) and demo sample once at startup."""
    loader = get_model_loader()
    try:
        loader.load()
    except ModelLoadError as exc:
        logger.warning("Model not loaded at startup: %s", exc)
    loader.load_sample()

    multi = get_multi_loader()
    try:
        multi.load()
    except ModelRegistryError as exc:
        logger.warning("Model registry not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title=TITLE,
    summary=SUMMARY,
    description=DESCRIPTION,
    version=VERSION,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Structured JSON log line per request (request_id, timestamp, latency, status)."""
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    token = request_id_ctx.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        json.dumps(
            {
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 1),
            }
        )
    )
    response.headers["X-Request-ID"] = rid
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.include_router(predict_router)
app.include_router(health_router)
app.include_router(ready_router)
app.include_router(model_router)
app.include_router(models_router)
app.include_router(metrics_router)
app.include_router(transactions_router)
app.include_router(features_router)


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    """503 with a clear message when no model is reachable."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def read_root() -> dict[str, str]:
    """Service landing page."""
    result: dict[str, str] = {"service": TITLE}
    if not IS_PRODUCTION:
        result["docs"] = "/docs"
    return result
