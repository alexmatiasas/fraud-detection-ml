"""Demo-sample transaction listing for the selection UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response

from fdml.api.dependencies import get_model_loader
from fdml.api.internal.loader import ModelLoader
from fdml.api.limiter import limiter
from fdml.api.metadata import API_PREFIX
from fdml.api.schemas.predict import TransactionList, TransactionSummary

transactions_router = APIRouter(prefix=f"{API_PREFIX}/transactions", tags=["predict"])


@transactions_router.get("/", response_model=TransactionList)
@limiter.limit("60/minute")
def list_transactions(
    request: Request,
    response: Response,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    loader: ModelLoader = Depends(get_model_loader),
) -> TransactionList:
    """Available demo-sample transactions (ID, amount, product, fraud label).

    The predict endpoints require a TransactionID; this endpoint is how the
    front discovers which ones exist without opening the dataset.
    """
    response.headers["Cache-Control"] = "public, max-age=300"
    total, records = loader.list_transactions(limit=limit, offset=offset)
    response.headers["X-Total-Count"] = str(total)
    return TransactionList(
        total=total, transactions=[TransactionSummary(**record) for record in records]
    )
