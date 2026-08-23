"""Fraud prediction by TransactionID lookup against the demo sample.

Supports optional raw-feature overrides ("what if the amount were 500?")
and per-prediction SHAP explanations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException, Request

from fdml.api.context import request_id as request_id_ctx
from fdml.api.dependencies import get_loader, get_multi_loader
from fdml.api.internal.explain import OverrideError, apply_overrides
from fdml.api.internal.loader import ModelLoader, to_model_input
from fdml.api.internal.registry import ModelRegistryError, MultiModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.models import CompareItem, CompareRequest, CompareResponse
from fdml.api.schemas.predict import (
    Explanation,
    PredictionRequest,
    PredictionResponse,
)

predict_router = APIRouter(prefix=f"{API_PREFIX}/predict", tags=["predict"])
audit_logger = logging.getLogger("fdml.api.audit")

BATCH_MAX = 50


def _to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python for JSON responses."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _audit_log(
    transaction_id: int,
    probability: float,
    is_fraud: bool,
    model_version: str | None,
    overrides_used: bool,
    shap_computed: bool,
) -> None:
    """Structured audit line for every scored prediction."""
    audit_logger.info(
        json.dumps(
            {
                "request_id": request_id_ctx.get(),
                "transaction_id": transaction_id,
                "probability": round(probability, 6),
                "is_fraud": is_fraud,
                "model_version": model_version,
                "overrides_used": overrides_used,
                "shap_computed": shap_computed,
            }
        )
    )


def _raw_dict(row: pd.DataFrame) -> dict[str, Any]:
    """Serializable raw row, omitting NaN (missing identity) fields."""
    first = row.iloc[0]
    return {key: _to_jsonable(value) for key, value in first.items() if pd.notna(value)}


def _score(
    loader: ModelLoader,
    transaction_id: int,
    overrides: dict[str, Any] | None = None,
    top_k: int = 20,
) -> PredictionResponse:
    row = loader.lookup(transaction_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {transaction_id} not found in demo sample",
        )

    try:
        row, overrides_applied = apply_overrides(row, overrides)
    except OverrideError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    X_raw = to_model_input(row)
    probability = float(loader.predict_proba(X_raw)[0])
    threshold = float(loader.report.get("best_threshold", 0.5))
    explanation = _explanation(loader.explain(X_raw, top_k=top_k))

    is_fraud = probability >= threshold
    confidence = abs(probability - threshold)
    risk_level = _risk_level(confidence)

    response = PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        probability=probability,
        threshold=threshold,
        risk_level=risk_level,
        confidence=confidence,
        is_above_threshold=is_fraud,
        model_version=loader.model_version,
        raw=_raw_dict(row),
        overrides_applied=overrides_applied or None,
        explanation=explanation,
    )
    _audit_log(
        transaction_id,
        probability,
        response.is_fraud,
        loader.model_version,
        overrides_used=bool(overrides_applied),
        shap_computed=explanation is not None,
    )
    return response


def _risk_level(confidence: float) -> str:
    """Distance-based risk classification."""
    if confidence < 0.15:
        return "critical"
    if confidence < 0.30:
        return "high"
    if confidence < 0.45:
        return "medium"
    return "low"


def _explanation(raw: dict[str, Any] | None) -> Explanation | None:
    if raw is None:
        return None
    return Explanation(**raw)


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
    Optional ``overrides`` mutate raw features before scoring.  A SHAP
    explanation is always included when the model supports it.
    """
    return _score(
        loader,
        req.transaction_id,
        overrides=req.overrides,
    )


@predict_router.post("/batch", response_model=list[PredictionResponse])
@limiter.limit("20/minute")
def batch_predict(
    request: Request,
    requests: list[PredictionRequest] = Body(..., max_length=BATCH_MAX),
    loader: ModelLoader = Depends(get_loader),
) -> list[PredictionResponse]:
    """Score multiple transactions in one call (max 50).

    Overrides and SHAP are not computed for batch calls; use the single
    endpoint for interactive "what if" exploration.
    """
    for req in requests:
        if req.overrides:
            raise HTTPException(
                status_code=422,
                detail="Overrides are not supported in batch; use POST /v1/predict/ for what-if",
            )
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
    overrides: dict[str, Any] | None = None,
    top_k: int = 20,
) -> PredictionResponse:
    info = multi.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not registered")

    try:
        row, overrides_applied = apply_overrides(row, overrides)
    except OverrideError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        probability = float(multi.predict_proba(name, to_model_input(row))[0])
    except (ModelRegistryError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    threshold = info.best_threshold
    explanation = _explanation(multi.explain(name, to_model_input(row), top_k=top_k))
    model_version = f"{name}:{info.version}" if info.version else name

    is_fraud = probability >= threshold
    confidence = abs(probability - threshold)
    risk_level = _risk_level(confidence)

    response = PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        probability=probability,
        threshold=threshold,
        risk_level=risk_level,
        confidence=confidence,
        is_above_threshold=is_fraud,
        model_version=model_version,
        raw=_raw_dict(row),
        overrides_applied=overrides_applied or None,
        explanation=explanation,
    )
    _audit_log(
        transaction_id,
        probability,
        response.is_fraud,
        model_version,
        overrides_used=bool(overrides_applied),
        shap_computed=explanation is not None,
    )
    return response


def _compare_item(multi: MultiModelLoader, name: str, row: pd.DataFrame) -> CompareItem:
    info = multi.get(name)
    if info is None:
        return CompareItem(model=name, error="not registered")
    try:
        probability = float(multi.predict_proba(name, to_model_input(row))[0])
    except (ModelRegistryError, KeyError) as exc:
        return CompareItem(model=name, error=str(exc))
    threshold = info.best_threshold
    is_fraud = probability >= threshold
    confidence = abs(probability - threshold)
    return CompareItem(
        model=name,
        probability=probability,
        is_fraud=is_fraud,
        threshold=threshold,
        risk_level=_risk_level(confidence),
        confidence=confidence,
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
    """Score a single transaction with a specific registered model.

    Supports the same ``overrides`` option as the default predict endpoint.
    A SHAP explanation is always included when the model supports it.
    """
    row = _row_or_404(loader, req.transaction_id)
    return _score_named(
        multi,
        name,
        req.transaction_id,
        row,
        overrides=req.overrides,
    )
