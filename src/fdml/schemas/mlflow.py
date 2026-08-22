from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


BackendType = Literal["local", "dagshub"]


class TrackingCfg(BaseModel):
    experiment_name: str = Field(
        default="fraud-detection", description="MLflow experiment name"
    )
    run_name: str = Field(default="lgbm_baseline", description="MLflow run name")
    tracking_uri: str = Field(
        default="sqlite:///mlruns.db",
        description="MLflow tracking URI (sqlite:///mlruns.db or DagsHub remote)",
    )
    backend: BackendType = Field(
        default="local",
        description="Backend type — auto-detected if dagshub credentials present",
    )


class RegistryCfg(BaseModel):
    enabled: bool = Field(default=True, description="Enable Model Registry")
    model_name: str = Field(
        default="fraud-detection-lgbm",
        description="Registered model name in MLflow Model Registry",
    )
    alias: Literal["champion", "challenger", "candidate"] = Field(
        default="champion",
        description="Model alias for registry (champion = production)",
    )


class DatasetCfg(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Log dataset information with mlflow.log_input",
    )
    context: str = Field(
        default="training",
        description="Dataset context label (training, validation, testing)",
    )


class MlflowFullConfig(BaseModel):
    tracking: TrackingCfg = Field(
        default_factory=TrackingCfg, description="Tracking server configuration"
    )
    registry: RegistryCfg = Field(
        default_factory=RegistryCfg, description="Model Registry configuration"
    )
    datasets: DatasetCfg = Field(
        default_factory=DatasetCfg, description="Dataset lineage configuration"
    )
    log_model: bool = Field(default=True, description="Log model artifact to MLflow")
    log_feature_importance: bool = Field(
        default=True, description="Log feature importance plot"
    )
