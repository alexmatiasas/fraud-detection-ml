"""Feature metadata for the \"what-if\" override UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from fdml.api.dependencies import get_model_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.features import FeatureInfo, FeatureList

features_router = APIRouter(prefix=f"{API_PREFIX}/features", tags=["model"])


@features_router.get("/", response_model=FeatureList)
@limiter.limit("30/minute")
def list_features(
    request: Request,
    loader: ModelLoader = Depends(get_model_loader),
) -> FeatureList:
    """List all raw features available for overrides.

    Returns feature metadata (name, dtype, importance, range/values) needed
    by the frontend to build the \"what-if\" exploration UI.  Features are
    ordered by importance descending.
    """
    if not loader.is_loaded:
        return FeatureList(features=[], total=0)

    sample = loader.sample
    report = loader.report
    top_features = {
        f["feature"]: f["importance"] for f in report.get("top_features", [])
    }

    features: list[FeatureInfo] = []
    if sample is not None:
        for col in sample.columns:
            if col in ("TransactionID", "isFraud"):
                continue
            series = sample[col]
            dtype_str = str(series.dtype)

            importance = top_features.get(col, 0.0)

            if hasattr(series, "cat") and series.cat.categories is not None:
                values = sorted(str(v) for v in series.cat.categories)
                features.append(
                    FeatureInfo(
                        name=col,
                        dtype=dtype_str,
                        importance=round(importance, 6),
                        values=values,
                    )
                )
            else:
                min_val = float(series.min()) if series.notna().any() else None
                max_val = float(series.max()) if series.notna().any() else None
                features.append(
                    FeatureInfo(
                        name=col,
                        dtype=dtype_str,
                        importance=round(importance, 6),
                        min_value=min_val,
                        max_value=max_val,
                    )
                )

    features.sort(key=lambda f: f.importance, reverse=True)
    return FeatureList(features=features, total=len(features))
