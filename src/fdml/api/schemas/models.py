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


class FeatureImportance(BaseModel):
    """One feature-importance entry from the evaluation report."""

    feature: str = Field(..., description="Feature-engineered column name")
    importance: float = Field(..., description="Importance score (model-specific)")


class RegisteredModelInfo(BaseModel):
    """A single registered model in the MLflow registry (leaderboard entry)."""

    name: str = Field(..., description="Registered model name")
    version: int | None = Field(
        default=None, description="Resolved registry version (champion or newest READY)"
    )
    status: str | None = Field(default=None, description="Registry status")
    run_id: str | None = Field(default=None, description="MLflow run that produced it")
    aliases: list[str] = Field(
        default_factory=list, description="Registry aliases (champion, challenger)"
    )
    active: bool = Field(
        default=False, description="True when this is the default served model"
    )
    loaded: bool = Field(
        default=False, description="True when the pipeline is loaded in memory"
    )
    n_features: int | None = Field(
        default=None, description="Number of features the model was trained on"
    )
    best_threshold: float = Field(
        default=0.5, description="Decision threshold (default 0.5 when unknown)"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Key evaluation metrics from report.json"
    )
    top_features: list[FeatureImportance] | None = Field(
        default=None,
        description="Most important features (by training importance), top 20",
    )
    error: str | None = Field(
        default=None, description="Load error when the model cannot be served"
    )


class ModelList(BaseModel):
    """All registered models, plus which one is the default."""

    default: str | None = Field(
        default=None, description="Model used by the default predict endpoint"
    )
    models: list[RegisteredModelInfo] = Field(
        default_factory=list, description="Registered models"
    )


class CompareRequest(BaseModel):
    """Score one transaction across a selection of registered models."""

    transaction_id: int = Field(
        ..., description="TransactionID present in the demo sample"
    )
    models: list[str] | None = Field(
        default=None, description="Model names to compare (default: all registered)"
    )


class CompareItem(BaseModel):
    """Per-model result within a comparison."""

    model: str = Field(..., description="Registered model name")
    probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="P(isFraud=1), null on error"
    )
    is_fraud: bool | None = Field(
        default=None, description="True when probability >= threshold, null on error"
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Decision threshold"
    )
    version: int | None = Field(
        default=None, description="Registry version used for the prediction"
    )
    run_id: str | None = Field(default=None, description="MLflow run that produced it")
    error: str | None = Field(
        default=None, description="Message when the model could not be scored"
    )


class CompareResponse(BaseModel):
    """Predictions from several models for the same transaction."""

    transaction_id: int = Field(..., description="Requested TransactionID")
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="The raw transaction row from the demo sample",
    )
    predictions: list[CompareItem] = Field(
        default_factory=list, description="One scored entry per requested model"
    )
