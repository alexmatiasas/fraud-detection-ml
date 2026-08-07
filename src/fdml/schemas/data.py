from pathlib import Path

from pydantic import BaseModel, Field

# TODO: refactor dirs with Path class using the @model_validator
# or @field_validator


class FilesCfg(BaseModel):
    train_transaction: str = Field(
        default="train_transaction.csv",
        description="Filename for training transaction CSV",
    )
    train_identity: str = Field(
        default="train_identity.csv", description="Filename for training identity CSV"
    )
    test_transaction: str = Field(
        default="test_transaction.csv", description="Filename for test transaction CSV"
    )
    test_identity: str = Field(
        default="test_identity.csv", description="Filename for test identity CSV"
    )
    sample_submission: str = Field(
        default="sample_submission.csv",
        description="Filename for sample submission CSV",
    )


class ValidationCfg(BaseModel):
    target_values: list[int] = Field(
        default=[0, 1], description="Expected unique values in target column"
    )
    transaction_id_unique: bool = Field(
        default=True, description="TransactionID must be unique", examples=[True, False]
    )
    transaction_amt_positive: bool = Field(
        default=True, description="TransactionAmt must be positive"
    )


class DataConfig(BaseModel):
    base_dir: Path = Field(default=Path("data"), description="Root data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw CSV directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed Parquet directory"
    )
    features_dir: Path = Field(
        default=Path("data/features"), description="Feature Feather directory"
    )
    files: FilesCfg = Field(description="Input/output filenames")
    join_key: str = Field(
        default="TransactionID",
        description="Column used to merge transaction and identity tables",
    )
    target: str = Field(default="isFraud", description="Target column name")
    validation: ValidationCfg = Field(description="Data quality validation rules")
