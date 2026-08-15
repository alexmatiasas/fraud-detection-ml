"""Liveness probe — the process is up and accepting requests."""

from fastapi import APIRouter

from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.health import HealthStatus

health_router = APIRouter(prefix=f"{API_PREFIX}/health", tags=["health"])


@health_router.get("/", response_model=HealthStatus)
def health() -> HealthStatus:
    """Liveness probe — confirms the API process is running."""
    return HealthStatus(status="ok")
