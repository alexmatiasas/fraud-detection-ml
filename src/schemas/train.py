from typing import Literal, Optional

from pydantic import BaseModel, Field


class DataPathsCfg(BaseModel):
    raw_dir: str = Field(default="data/raw", description="Raw data directory")
    processed_dir: str = Field(
        default="data/processed", description="Processed Parquet directory"
    )
    features_dir: str = Field(
        default="data/features", description="Feature Feather directory"
    )
    train_transaction: str = Field(
        default="data/raw/train_transaction.parquet",
        description="Training transaction file",
    )
    train_identity: str = Field(
        default="data/raw/train_identity.parquet", description="Training identity file"
    )
    test_transaction: str = Field(
        default="data/raw/test_transaction.parquet", description="Test transaction file"
    )
    test_identity: str = Field(
        default="data/raw/test_identity.parquet", description="Test identity file"
    )


SplitStrategy = Literal["temporal", "random"]


class SplitCfg(BaseModel):
    strategy: SplitStrategy = Field(
        default="temporal",
        description="Split strategy — temporal is preferred for fraud",
    )
    time_col: str = Field(
        default="TransactionDT", description="Column used for temporal split"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of data held out for validation",
    )


ModelName = Literal["lightgbm", "xgboost", "logistic_regression"]


class LightGBMParams(BaseModel):
    n_estimators: int = Field(
        default=1000, ge=1, description="Number of boosting rounds"
    )
    learning_rate: float = Field(
        default=0.05, gt=0.0, description="Boosting learning rate"
    )
    max_depth: int = Field(
        default=8, ge=-1, description="Maximum tree depth (-1 = unlimited)"
    )
    num_leaves: int = Field(default=127, ge=1, description="Maximum tree leaves")
    min_child_samples: int = Field(
        default=100, ge=1, description="Minimum data per child node"
    )
    subsample: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Row sampling ratio per tree"
    )
    colsample_bytree: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Column sampling ratio per tree"
    )
    reg_alpha: float = Field(default=0.1, ge=0.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0.0, description="L2 regularization")
    scale_pos_weight: float = Field(
        default=27.6,
        gt=0.0,
        description="Class weight for positive class (~(1-0.035)/0.035)",
    )
    random_state: int = Field(default=42, description="PRNG seed")
    n_jobs: int = Field(default=-1, description="Parallel threads (-1 = all cores)")


class ModelCfg(BaseModel):
    name: ModelName = Field(default="lightgbm", description="Model algorithm")
    artifact_dir: str = Field(
        default="models/", description="Directory for model artifacts"
    )
    params: LightGBMParams = Field(
        default_factory=LightGBMParams, description="Model hyperparameters"
    )


OptunaMetric = Literal["roc_auc", "average_precision"]


class OptunaCfg(BaseModel):
    enabled: bool = Field(
        default=False, description="Enable Optuna hyperparameter optimisation"
    )
    n_trials: int = Field(default=50, ge=1, description="Number of HPO trials")
    timeout_seconds: Optional[int] = Field(
        default=3600, ge=1, description="HPO timeout (None = unlimited)"
    )
    metric: OptunaMetric = Field(default="roc_auc", description="Metric to optimise")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimisation direction"
    )
    study_name: str = Field(
        default="fraud_lgbm_v1", description="Optuna study name (persisted to DB)"
    )


EvalMetric = Literal["auc", "average_precision"]


class EarlyStoppingCfg(BaseModel):
    enabled: bool = Field(default=True, description="Enable early stopping")
    rounds: int = Field(default=100, ge=1, description="Patience rounds")
    eval_metric: EvalMetric = Field(
        default="auc", description="Early stopping evaluation metric"
    )


EvalMetricName = Literal[
    "roc_auc", "average_precision", "f1_score", "precision", "recall"
]


class EvaluationCfg(BaseModel):
    metrics: list[EvalMetricName] = Field(
        default_factory=lambda: [
            "roc_auc",
            "average_precision",
            "f1_score",
            "precision",
            "recall",
        ],
        description="Metrics computed at evaluation time",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold for binary classification",
    )


class MlflowCfg(BaseModel):
    experiment_name: str = Field(
        default="fraud-detection", description="MLflow experiment name"
    )
    run_name: str = Field(default="lgbm_baseline", description="MLflow run name")
    tracking_uri: str = Field(
        default="mlruns/",
        description="MLflow tracking URI (local dir or DagsHub remote)",
    )
    log_model: bool = Field(default=True, description="Log model artifact to MLflow")
    log_feature_importance: bool = Field(
        default=True, description="Log feature importance plot"
    )
    log_shap: bool = Field(default=True, description="Log SHAP summary plot")


class ShapCfg(BaseModel):
    enabled: bool = Field(
        default=True, description="Compute SHAP values at evaluation time"
    )
    max_display: int = Field(
        default=20, ge=1, description="Top-N features in summary plot"
    )
    sample_size: int = Field(
        default=1000,
        ge=1,
        description="Sample size for SHAP computation (full dataset is expensive)",
    )


class TrainConfig(BaseModel):
    seed: int = Field(default=42, description="Global PRNG seed")
    data: DataPathsCfg = Field(
        default_factory=DataPathsCfg, description="Data file paths"
    )
    split: SplitCfg = Field(
        default_factory=SplitCfg, description="Train/validation split configuration"
    )
    model: ModelCfg = Field(default_factory=ModelCfg, description="Model configuration")
    optuna: OptunaCfg = Field(
        default_factory=OptunaCfg, description="Optuna HPO configuration"
    )
    early_stopping: EarlyStoppingCfg = Field(
        default_factory=EarlyStoppingCfg, description="Early stopping configuration"
    )
    evaluation: EvaluationCfg = Field(
        default_factory=EvaluationCfg, description="Evaluation metrics configuration"
    )
    mlflow: MlflowCfg = Field(
        default_factory=MlflowCfg, description="MLflow tracking configuration"
    )
    shap: ShapCfg = Field(
        default_factory=ShapCfg, description="SHAP explainability configuration"
    )
