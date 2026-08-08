from typing import Literal

from pydantic import BaseModel, Field


AggFunc = Literal["mean", "std", "max", "count", "min", "sum"]
DateTimeFeature = Literal["hour_of_day", "day_of_week", "day_of_month"]


class VestaCfg(BaseModel):
    include: bool = Field(description="Whether to include Vesta V features")
    variance_threshold: float = Field(
        ge=0.0,
        description="Variance floor — V columns below this are dropped",
    )
    correlation_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Pearson r threshold — one of each correlated pair is dropped",
    )


class CountCorrFilterCfg(BaseModel):
    include: bool = Field(
        default=True,
        description="Whether to correlation-filter C count features",
    )
    variance_threshold: float = Field(
        ge=0.0,
        default=0.01,
        description="Variance floor — C columns below this are dropped",
    )
    correlation_threshold: float = Field(
        ge=0.0,
        le=1.0,
        default=0.99,
        description="Pearson r threshold — one of each correlated pair is dropped",
    )


class TransactionCfg(BaseModel):
    target: str = Field(default="isFraud", description="Target column name")
    numerical: list[str] = Field(
        default=["TransactionAmt", "TransactionDT", "dist1", "dist2"],
        description="Direct numeric features selected from EDA",
    )
    categorical: list[str] = Field(
        default=[
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "M7",
            "M8",
            "M9",
        ],
        description="Low-cardinality categorical features",
    )
    high_cardinality: list[str] = Field(
        default=["card1", "card2", "card3", "card5", "addr1", "addr2"],
        description="High-cardinality features (frequency-encoded)",
    )
    count_features: list[str] = Field(
        default=[
            "C1",
            "C2",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
        ],
        description="C-prefix count features",
    )
    delta_features: list[str] = Field(
        default=["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"],
        description="D-prefix delta features (D6–D9, D12–D14 dropped per EDA 5.6)",
    )
    vesta_features: VestaCfg | None = Field(
        default=None,
        description="Vesta engineered V features config",
    )
    count_corr_filter: CountCorrFilterCfg | None = Field(
        default=None,
        description="Correlation filter for C count features",
    )


class IdentityCfg(BaseModel):
    categorical: list[str] = Field(
        default=["DeviceType", "DeviceInfo"],
        description="Identity categorical columns",
    )
    id_features: list[str] = Field(
        default=[f"id_{i:02d}" for i in range(1, 39)],
        description="id_xx identity columns",
    )


class CardAggregationCfg(BaseModel):
    group_by: list[str] = Field(
        description="Columns to group by (e.g. card1 through card5)",
    )
    aggregations: dict[str, list[AggFunc]] = Field(
        description="Per-column aggregation functions",
    )


class EngineeredCfg(BaseModel):
    datetime: list[DateTimeFeature] = Field(
        default=["hour_of_day", "day_of_week", "day_of_month"],
        description="Temporal features extracted from TransactionDT",
    )
    card_aggregations: CardAggregationCfg | None = Field(
        default=None,
        description="Group-by aggregations on card columns",
    )
    frequency_encoding: list[str] = Field(
        default=[
            "P_emaildomain",
            "R_emaildomain",
            "DeviceInfo",
            "card1",
            "card2",
            "card3",
            "card5",
            "addr1",
            "addr2",
        ],
        description="Columns to frequency-encode",
    )
    device_os: bool = Field(
        default=True,
        description="Extract OS from DeviceInfo (heuristic)",
    )
    device_brand: bool = Field(
        default=True,
        description="Extract brand from DeviceInfo (heuristic)",
    )
    cyclical: bool = Field(
        default=True,
        description="Cyclic sin/cos encoding of the datetime features",
    )
    has_identity: bool = Field(
        default=True,
        description="Binary flag for presence of an identity record",
    )
    log_amount: bool = Field(
        default=True,
        description="log1p transform of TransactionAmt",
    )
    is_round_amount: bool = Field(
        default=True,
        description="Flag for whole-number TransactionAmt",
    )
    email_domain: bool = Field(
        default=True,
        description="Normalize email domains and add payer/recipient match flag",
    )


class MatchFlagsCfg(BaseModel):
    T: int = Field(default=1, description="Mapping for True in binary M flags")
    F: int = Field(default=0, description="Mapping for False in binary M flags")
    nan: int = Field(default=-1, description="Mapping for NaN in binary M flags")


class EncodingCfg(BaseModel):
    low_cardinality_max_values: int = Field(
        default=50,
        ge=1,
        description="Max unique values for low-cardinality treatment",
    )
    high_cardinality_strategy: Literal["frequency"] = Field(
        default="frequency",
        description="Encoding for high-cardinality columns",
    )
    match_flags_encoding: MatchFlagsCfg = Field(
        default_factory=MatchFlagsCfg,
        description="T/F/NaN mapping for M-prefix flags",
    )


class ImputationCfg(BaseModel):
    numerical: int = Field(
        default=-999,
        description="Sentinel for missing numerical values (LightGBM-native)",
    )
    categorical: str = Field(
        default="missing",
        description="Sentinel for missing categorical values",
    )


class FeaturesConfig(BaseModel):
    transaction: TransactionCfg = Field(
        description="Transaction table feature selection",
    )
    identity: IdentityCfg | None = Field(
        default=None,
        description="Identity table feature selection",
    )
    engineered: EngineeredCfg = Field(
        description="Engineered features created by the pipeline",
    )
    encoding: EncodingCfg = Field(
        default_factory=EncodingCfg,
        description="Encoding strategy configuration",
    )
    imputation: ImputationCfg = Field(
        default_factory=ImputationCfg,
        description="Missing value imputation strategy",
    )
