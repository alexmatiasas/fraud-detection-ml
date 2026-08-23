"""Schema for the feature metadata endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureInfo(BaseModel):
    """Metadata for a single raw feature available for overrides."""

    name: str = Field(..., description="Raw column name in the demo sample")
    dtype: str = Field(..., description="Pandas dtype string (e.g. float32, category)")
    importance: float = Field(
        default=0.0, description="LightGBM split importance (0 if not in top features)"
    )
    min_value: float | None = Field(
        default=None, description="Minimum value in the sample (numeric features)"
    )
    max_value: float | None = Field(
        default=None, description="Maximum value in the sample (numeric features)"
    )
    values: list[str] | None = Field(
        default=None, description="Sorted unique categories (categorical features)"
    )


class FeatureList(BaseModel):
    """List of all raw features with metadata for the override UI."""

    features: list[FeatureInfo] = Field(
        default_factory=list, description="Features ordered by importance descending"
    )
    total: int = Field(..., description="Total number of overridable features")
