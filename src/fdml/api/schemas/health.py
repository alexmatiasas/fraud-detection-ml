"""Health / readiness probe models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str = Field(default="ok", description="Liveness status")


class ReadyStatus(BaseModel):
    status: str = Field(..., description="ready when the model is loaded")
    model_loaded: bool = Field(..., description="Whether a model is available")
