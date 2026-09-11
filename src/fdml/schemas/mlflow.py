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
    min_auc: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Quality gate: minimum val ROC AUC to register a version "
            "(0 = disabled). Runs below the threshold get tag "
            "validation_status=rejected and are NOT registered"
        ),
    )
    min_average_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quality gate: minimum val average precision (0 = disabled)",
    )
    tags: dict[str, str] = Field(
        default_factory=lambda: {
            "framework": "lightgbm",
            "dataset": "ieee-cis-fraud",
            "task": "binary_classification",
        },
        description="Metadata tags attached to every registered model version",
    )
    description: str = Field(
        default=(
            "LightGBM binary classifier for IEEE-CIS fraud detection. "
            "Full pipeline (feature engineering + model) serves raw "
            "transactions."
        ),
        description="Description for the registered model",
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


class NativeEvaluateCfg(BaseModel):
    """Native ``mlflow.models.evaluate`` on the logged pipeline.

    Complements the custom evaluate runner: MLflow computes standard
    classifier metrics, interactive ROC/PR/confusion-matrix artifacts and an
    optional SHAP explainer, all linked to the LoggedModel.
    """

    enabled: bool = Field(
        default=True,
        description="Run mlflow.models.evaluate on the logged model",
    )
    log_explainer: bool = Field(
        default=True,
        description="Log a SHAP explainer as a run artifact (serving-ready)",
    )
    max_rows: int = Field(
        default=20000,
        ge=0,
        description=(
            "Row cap for the native evaluation set (random sample, fixed "
            "seed) — keeps prediction + explainer cheap on big folds. "
            "0 = full validation fold"
        ),
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
    native_evaluate: NativeEvaluateCfg = Field(
        default_factory=NativeEvaluateCfg,
        description="Native mlflow.models.evaluate configuration",
    )
    log_model: bool = Field(default=True, description="Log model artifact to MLflow")
    log_feature_importance: bool = Field(
        default=True, description="Log feature importance plot"
    )
