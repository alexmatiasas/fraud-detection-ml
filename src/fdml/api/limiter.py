"""Shared slowapi rate limiter for the fraud detection service.

Limits are applied per-endpoint via ``@limiter.limit(...)``. Because slowapi
needs the ``request: Request`` in every guarded handler signature, the limiter
lives here so routers import a single instance.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)

# All configured limits use a 1-minute window, so the retry hint is constant.
RETRY_AFTER_SECONDS = "60"


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """429 response with the standard rate-limit headers attached."""
    response = JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )
    response.headers["Retry-After"] = RETRY_AFTER_SECONDS
    return response
