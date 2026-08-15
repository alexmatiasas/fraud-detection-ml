"""Fraud prediction by TransactionID lookup against the demo sample."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from fdml.api.dependencies import get_loader
from fdml.api.internal.loader import ModelLoader, to_model_input
from fdml.api.schemas.predict import PredictionRequest, PredictionResponse

predict_router = APIRouter(prefix="/predict", tags=["predict"])


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON responses."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _raw_dict(row: pd.DataFrame) -> dict[str, Any]:
    """Serializable raw row, omitting NaN (missing identity) fields."""
    first = row.iloc[0]
    return {key: _to_jsonable(value) for key, value in first.items() if pd.notna(value)}


def _score(loader: ModelLoader, transaction_id: int) -> PredictionResponse:
    row = loader.lookup(transaction_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found in demo sample",
        )

    X_raw = to_model_input(row)
    probability = float(loader.predict_proba(X_raw)[0])
    threshold = float(loader.report.get("best_threshold", 0.5))

    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=probability >= threshold,
        probability=probability,
        threshold=threshold,
        model_version=loader.model_version,
        raw=_raw_dict(row),
    )


@predict_router.post("/", response_model=PredictionResponse)
def predict(
    request: PredictionRequest, loader: ModelLoader = Depends(get_loader)
) -> PredictionResponse:
    """Score a single transaction by TransactionID.

    The full raw row is pulled from the demo sample and scored through the
    complete feature pipeline so the model sees its full 341-feature space.
    """
    return _score(loader, request.transaction_id)


@predict_router.post("/batch", response_model=list[PredictionResponse])
def batch_predict(
    requests: list[PredictionRequest], loader: ModelLoader = Depends(get_loader)
) -> list[PredictionResponse]:
    """Score multiple transactions in one call."""
    return [_score(loader, req.transaction_id) for req in requests]
