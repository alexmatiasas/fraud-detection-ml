"""Model metadata response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model_name: str | None = Field(
        default=None, description="Algorithm name (e.g. lightgbm)"
    )
    source: str | None = Field(
        default=None, description="Loading source: mlflow | local"
    )
    run_id: str | None = Field(default=None, description="MLflow run id")
    version: int | None = Field(
        default=None, description="MLflow model registry version"
    )
    loaded_at: str | None = Field(
        default=None, description="ISO timestamp of the last successful load"
    )
    n_features: int | None = Field(
        default=None, description="Number of features the model was trained on"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Key evaluation metrics from models/report.json",
    )
