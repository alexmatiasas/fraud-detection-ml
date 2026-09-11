from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    max_train_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "Cap the training fold to the earliest N rows by time "
            "(validation stays untouched). 0 = use all rows"
        ),
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
    embargo_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Time gap (in time_col units) dropped between train and validation "
            "to prevent leakage from aggregate features; the Kaggle train/test "
            "windows are ~30 days apart"
        ),
    )


ModelName = Literal["lightgbm", "xgboost", "random_forest"]


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

    model_config = ConfigDict(extra="allow")


class XGBoostParams(BaseModel):
    n_estimators: int = Field(
        default=500, ge=1, description="Number of boosting rounds"
    )
    learning_rate: float = Field(
        default=0.05, gt=0.0, description="Boosting learning rate"
    )
    max_depth: int = Field(default=8, ge=0, description="Maximum tree depth")
    min_child_weight: float = Field(
        default=5.0, ge=0.0, description="Minimum sum of instance weight"
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
        default=27.6, gt=0.0, description="Class weight for positive class"
    )
    random_state: int = Field(default=42, description="PRNG seed")
    n_jobs: int = Field(default=-1, description="Parallel threads (-1 = all cores)")

    model_config = ConfigDict(extra="allow")


class RandomForestParams(BaseModel):
    n_estimators: int = Field(
        default=300, ge=1, description="Number of trees in the forest"
    )
    max_depth: int = Field(default=8, ge=0, description="Maximum tree depth")
    min_child_samples: int = Field(
        default=50, ge=1, description="Minimum samples per leaf (min_samples_leaf)"
    )
    subsample: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Fraction of samples for each tree (max_samples)",
    )
    colsample_bytree: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Fraction of features per tree (max_features)",
    )
    random_state: int = Field(default=42, description="PRNG seed")
    n_jobs: int = Field(default=-1, description="Parallel threads (-1 = all cores)")

    model_config = ConfigDict(extra="allow")


ModelParams = Annotated[
    Union[LightGBMParams, XGBoostParams, RandomForestParams],
    Field(discriminator=None),
]


class ModelCfg(BaseModel):
    name: ModelName = Field(default="lightgbm", description="Model algorithm")
    artifact_dir: str = Field(
        default="models/", description="Directory for model artifacts"
    )
    params: ModelParams = Field(
        default_factory=LightGBMParams, description="Model hyperparameters"
    )

    @classmethod
    def _resolve_params(cls, v: Any, info: Any) -> Any:
        """Select the correct params class based on model.name."""
        if isinstance(v, dict):
            name = info.data.get("name", "lightgbm") if info else "lightgbm"
            params_cls = {
                "lightgbm": LightGBMParams,
                "xgboost": XGBoostParams,
                "random_forest": RandomForestParams,
            }.get(name, LightGBMParams)
            return params_cls.model_validate(v)
        return v

    model_config = {"json_schema_extra": {"examples": []}}

    @model_validator(mode="before")
    @classmethod
    def _select_params_class(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data:
            name = data.get("name", "lightgbm")
            params = data["params"]
            if isinstance(params, dict):
                params_cls = {
                    "lightgbm": LightGBMParams,
                    "xgboost": XGBoostParams,
                    "random_forest": RandomForestParams,
                }.get(name, LightGBMParams)
                data["params"] = params_cls.model_validate(params)
        return data


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
    save_best_params: bool = Field(
        default=True, description="Save best HPO params to models/best_params_*.json"
    )


EvalMetric = Literal["auc", "average_precision"]


class EarlyStoppingCfg(BaseModel):
    enabled: bool = Field(default=True, description="Enable early stopping")
    rounds: int = Field(default=100, ge=1, description="Patience rounds")
    eval_metric: EvalMetric = Field(
        default="auc", description="Early stopping evaluation metric"
    )
    eval_max_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "Max rows of the validation fold used for early stopping and the "
            "per-iteration metric curve (0 = full fold). A fixed stratified "
            "subsample makes each iteration much cheaper; final metrics are "
            "still computed on the full validation fold"
        ),
    )
    train_eval_max_rows: int = Field(
        default=50000,
        ge=0,
        description=(
            "Max rows of the training set used for the per-iteration TRAIN "
            "curve (train/auc), so overfitting vs validation is visible (0 = "
            "off). A fixed random sample keeps each iteration cheap; the model "
            "still trains on the full training set"
        ),
    )


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


class MlflowCfg(BaseModel):
    experiment_name: str = Field(
        default="fraud-detection", description="MLflow experiment name"
    )
    run_name: str = Field(default="lgbm_baseline", description="MLflow run name")
    log_model: bool = Field(default=True, description="Log model artifact to MLflow")
    experiment_tag: str = Field(
        default="",
        description=(
            "Experiment group label (e.g. 'A_baseline', 'B_reg'). Empty = no tag. "
            "Used to filter/compare runs across seeds and variants"
        ),
    )


class TrainingCallbacksCfg(BaseModel):
    log_per_iteration: bool = Field(
        default=True,
        description="Log per-iteration validation metrics to MLflow + DVCLive",
    )
    mlflow_every: int = Field(
        default=1,
        ge=1,
        description=(
            "Send one MLflow metric per this many iterations. Each remote "
            "log_metric call to DagsHub costs ~1.5s, so 10 keeps the curve "
            "while cutting experiment wall time ~10x. Console logging is "
            "unaffected"
        ),
    )
    dvclive: bool = Field(
        default=False,
        description="Log per-iteration metrics via DVCLive (separate from eval dvclive)",
    )


class AblationCfg(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Run the feature ablation study instead of a single fit",
    )
    max_train_rows: int = Field(
        default=0,
        ge=0,
        description="Cap train rows per variant to keep the study fast (0 = all)",
    )
    seeds: list[int] = Field(
        default_factory=lambda: [40, 41, 42],
        description=(
            "Seeds to run per variant (one MLflow run each). Empty falls back "
            "to the global TrainConfig.seed"
        ),
    )
    report_path: str = Field(
        default="models/feature_ablation.csv",
        description="Where to save the per-variant results table",
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
    mlflow: MlflowCfg = Field(
        default_factory=MlflowCfg, description="Minimal MLflow config"
    )
    shap: ShapCfg = Field(
        default_factory=ShapCfg, description="SHAP explainability configuration"
    )
    training_callbacks: TrainingCallbacksCfg = Field(
        default_factory=TrainingCallbacksCfg,
        description="Per-iteration training callbacks",
    )
    ablation: AblationCfg = Field(
        default_factory=AblationCfg,
        description="Feature ablation study configuration",
    )
