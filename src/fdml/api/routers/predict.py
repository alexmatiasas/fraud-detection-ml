from fastapi import APIRouter
from pydantic import BaseModel

predict_router = APIRouter(prefix="/predict", tags=["predict"])


class Transaction(BaseModel):
    transaction_id: int


class Prediction(BaseModel):
    isFraud: bool | None = None
    fraud_probability: float = 0.0
    # shap_values: Explainer | None = None


@predict_router.post("/")
def predict(transaction: Transaction) -> Prediction:
    """transaction -> prediction

    Args:
        transaction (Transaction): _description_

    Returns:
        Prediction: _description_
    """


@predict_router.post("/batch/")
def batch_predict(transactions: list[Transaction]) -> list[Prediction]:
    """list of transactions -> list of predictions

    Args:
        transaction (list[Transaction]): _description_
    """
