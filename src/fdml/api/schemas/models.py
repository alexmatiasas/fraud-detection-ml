"""Model registry version schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelVersionInfo(BaseModel):
    """A single version of the registered model in the MLflow registry."""

    version: int = Field(..., description="Model registry version number")
    status: str = Field(..., description="Registry status (e.g. READY, ARCHIVED)")
    run_id: str | None = Field(
        default=None, description="MLflow run that produced this version"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Registry aliases (champion, challenger)"
    )
    active: bool = Field(
        default=False, description="True when this is the version currently served"
    )


class ModelSwitchRequest(BaseModel):
    """Select a registered model version to serve."""

    version: int = Field(..., ge=1, description="Version number to switch to")

    def __hash__(self) -> int:
        return hash(self.version)


class ModelSwitchResponse(BaseModel):
    """Confirmation of a model switch."""

    status: str = Field(default="switched", description="Always 'switched' on success")
    model_name: str = Field(..., description="Registered model name")
    version: int = Field(..., description="Newly active version")
    source: str = Field(..., description="Loading source (always 'mlflow')")
    loaded_at: str = Field(..., description="ISO timestamp of the switch")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Evaluation report of the active model"
    )
