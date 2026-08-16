"""Fraud prediction by TransactionID lookup against the demo sample."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Request

from fdml.api.dependencies import get_loader, get_multi_loader
from fdml.api.internal.loader import ModelLoader, to_model_input
from fdml.api.internal.registry import ModelRegistryError, MultiModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.models import CompareItem, CompareRequest, CompareResponse
from fdml.api.schemas.predict import PredictionRequest, PredictionResponse

predict_router = APIRouter(prefix=f"{API_PREFIX}/predict", tags=["predict"])

BATCH_MAX = 50


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
@limiter.limit("60/minute")
def predict(
    request: Request,
    req: PredictionRequest,
    loader: ModelLoader = Depends(get_loader),
) -> PredictionResponse:
    """Score a single transaction by TransactionID.

    The full raw row is pulled from the demo sample and scored through the
    complete feature pipeline so the model sees its full 341-feature space.
    """
    return _score(loader, req.transaction_id)


@predict_router.post("/batch", response_model=list[PredictionResponse])
@limiter.limit("20/minute")
def batch_predict(
    request: Request,
    requests: list[PredictionRequest] = Body(..., max_length=BATCH_MAX),
    loader: ModelLoader = Depends(get_loader),
) -> list[PredictionResponse]:
    """Score multiple transactions in one call (max 50)."""
    return [_score(loader, req.transaction_id) for req in requests]


def _row_or_404(loader: ModelLoader, transaction_id: int) -> pd.DataFrame:
    row = loader.lookup(transaction_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found in demo sample",
        )
    return row


def _score_named(
    multi: MultiModelLoader,
    name: str,
    transaction_id: int,
    row: pd.DataFrame,
) -> PredictionResponse:
    info = multi.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not registered")
    try:
        probability = float(multi.predict_proba(name, to_model_input(row))[0])
    except (ModelRegistryError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    threshold = info.best_threshold
    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=probability >= threshold,
        probability=probability,
        threshold=threshold,
        model_version=f"{name}:{info.version}" if info.version else name,
        raw=_raw_dict(row),
    )


def _compare_item(multi: MultiModelLoader, name: str, row: pd.DataFrame) -> CompareItem:
    info = multi.get(name)
    if info is None:
        return CompareItem(model=name, error="not registered")
    try:
        probability = float(multi.predict_proba(name, to_model_input(row))[0])
    except (ModelRegistryError, KeyError) as exc:
        return CompareItem(model=name, error=str(exc))
    threshold = info.best_threshold
    return CompareItem(
        model=name,
        probability=probability,
        is_fraud=probability >= threshold,
        threshold=threshold,
        version=info.version,
        run_id=info.run_id,
    )


@predict_router.post("/compare", response_model=CompareResponse)
@limiter.limit("30/minute")
def compare_models(
    request: Request,
    req: CompareRequest,
    loader: ModelLoader = Depends(get_loader),
    multi: MultiModelLoader = Depends(get_multi_loader),
) -> CompareResponse:
    """Score one transaction across all (or selected) registered models."""
    row = _row_or_404(loader, req.transaction_id)
    names = req.models or multi.model_names()
    return CompareResponse(
        transaction_id=req.transaction_id,
        raw=_raw_dict(row),
        predictions=[_compare_item(multi, name, row) for name in names],
    )


@predict_router.post("/{name}", response_model=PredictionResponse)
@limiter.limit("60/minute")
def predict_with_model(
    request: Request,
    name: str,
    req: PredictionRequest,
    loader: ModelLoader = Depends(get_loader),
    multi: MultiModelLoader = Depends(get_multi_loader),
) -> PredictionResponse:
    """Score a single transaction with a specific registered model."""
    row = _row_or_404(loader, req.transaction_id)
    return _score_named(multi, name, req.transaction_id, row)
