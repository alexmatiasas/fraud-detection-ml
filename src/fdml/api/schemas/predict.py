"""API request/response models for the fraud detection service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Lookup a transaction from the demo sample and score it.

    The model is scored on the full raw row (all 434 merged columns)
    retrieved from ``data/samples/val_sample.parquet`` so the feature
    pipeline rebuilds the exact 341-feature space it was trained on.
    """

    transaction_id: int = Field(
        ..., description="TransactionID present in the demo sample"
    )


class PredictionResponse(BaseModel):
    transaction_id: int = Field(..., description="Requested TransactionID")
    is_fraud: bool = Field(..., description="True when probability >= best_threshold")
    probability: float = Field(..., ge=0.0, le=1.0, description="P(isFraud=1)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision threshold")
    model_version: str | None = Field(
        default=None, description="MLflow run/version that produced the prediction"
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The raw transaction row from the demo sample, including the "
            "ground-truth isFraud label when present (for prediction vs "
            "actual comparison)"
        ),
    )
