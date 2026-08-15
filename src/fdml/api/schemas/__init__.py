"""API request/response models."""

from fdml.api.schemas.health import HealthStatus, ReadyStatus
from fdml.api.schemas.model import ModelInfo
from fdml.api.schemas.models import (
    ModelSwitchRequest,
    ModelSwitchResponse,
    ModelVersionInfo,
)
from fdml.api.schemas.predict import PredictionRequest, PredictionResponse

__all__ = [
    "HealthStatus",
    "ReadyStatus",
    "ModelInfo",
    "ModelSwitchRequest",
    "ModelSwitchResponse",
    "ModelVersionInfo",
    "PredictionRequest",
    "PredictionResponse",
]
