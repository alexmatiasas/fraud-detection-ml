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
    overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional raw-feature overrides applied before scoring, e.g. "
            '{"TransactionAmt": 500.0, "ProductCD": "W"}. Values are '
            "cast to the column dtype; unknown features return 422."
        ),
    )
    include_shap: bool = Field(
        default=False,
        description="When true, include a per-prediction SHAP explanation",
    )


class ShapContribution(BaseModel):
    feature: str = Field(..., description="Feature-engineered column name")
    value: float | str | bool | None = Field(
        default=None, description="Current feature value for the scored row"
    )
    shap: float = Field(..., description="SHAP contribution of this feature")


class Explanation(BaseModel):
    base_value: float = Field(..., description="Expected model output (log-odds)")
    n_features: int = Field(..., description="Width of the feature-engineered space")
    top_features: list[ShapContribution] = Field(
        default_factory=list, description="Top contributors by |SHAP|, descending"
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
            "actual comparison). Reflects any overrides applied."
        ),
    )
    overrides_applied: dict[str, Any] | None = Field(
        default=None, description="Normalized overrides that changed the scored row"
    )
    explanation: Explanation | None = Field(
        default=None, description="SHAP explanation, when requested and available"
    )


class TransactionSummary(BaseModel):
    transaction_id: int = Field(..., description="TransactionID in the demo sample")
    amount: float | None = Field(default=None, description="TransactionAmt")
    product_cd: str | None = Field(default=None, description="ProductCD")
    is_fraud: bool | None = Field(
        default=None, description="Ground-truth fraud label from the sample"
    )


class TransactionList(BaseModel):
    total: int = Field(..., description="Total transactions in the demo sample")
    transactions: list[TransactionSummary] = Field(
        default_factory=list, description="One summary per transaction on the page"
    )
