from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from fdml.features.amount import AmountFeatureExtractor
from fdml.features.card import CardAggregator
from fdml.features.device import DeviceFeatureExtractor
from fdml.features.dtypes import DtypeOptimizer
from fdml.features.email import EmailFeatureExtractor
from fdml.features.flags import MFlagEncoder
from fdml.features.freq import FrequencyEncoder
from fdml.features.id_encoder import IdCodeEncoder
from fdml.features.identity import IdentityFlagExtractor
from fdml.features.imputer import MissingImputer
from fdml.features.selector import FeatureSelector
from fdml.features.time import TimeFeatureExtractor
from fdml.features.vfilter import VFeatureFilter
from fdml.schemas.data import DataConfig
from fdml.schemas.features import FeaturesConfig
from fdml.utils.paths import data_config, features_config


def load_fe_config(config: Mapping[str, Any] = features_config) -> FeaturesConfig:
    """This takes the configuration file [configs/features.yaml](configs/features.yaml) parsed as a only read dictionary
    (:class:`~collections.abc.Mapping` class) loaded and solved with :mod:`~OmegaConfig` and validates against the :class:`~fdml.schemas.features.FeaturesConfig` pydantic model

    Args:
        config (Mapping[str, Any], optional): A configuration dictionary (loaded and solved with :mod:`~OmegaConfig`, check :mod:`~fdml.utils.paths` module). Defaults to features_config.

    Returns:
        FeaturesConfig: Pydantic model validated against the :class:`~fraud_detection.schemas.features.FeaturesConfig` pydantic model.
    """
    return FeaturesConfig.model_validate(config)


def load_data_config(config: Mapping[str, Any] = data_config) -> DataConfig:

    return DataConfig.model_validate(config)


def _build_feature_columns(cfg: FeaturesConfig) -> list[str]:
    """Build the ordered list of expected feature columns from config."""
    cols: list[str] = []
    t = cfg.transaction
    cols.extend(t.numerical)
    cols.extend(t.categorical)
    cols.extend(t.high_cardinality)
    cols.extend(t.count_features)
    cols.extend(t.delta_features)

    if cfg.identity:
        cols.extend(cfg.identity.categorical)
        cols.extend(cfg.identity.id_features)

    eng = cfg.engineered
    cols.extend(eng.datetime)

    if eng.cyclical:
        for feature in eng.datetime:
            cols.append(f"{feature}_sin")
            cols.append(f"{feature}_cos")

    if eng.has_identity:
        cols.append("has_identity")

    if eng.log_amount:
        cols.append("TransactionAmt_log")
    if eng.is_round_amount:
        cols.append("is_round_amount")

    if eng.email_domain:
        cols.append("p_r_domain_match")

    if eng.device_os:
        cols.append("device_os")
    if eng.device_brand:
        cols.append("device_brand")

    for freq_col in eng.frequency_encoding:
        cols.append(f"{freq_col}_freq")

    if eng.card_aggregations:
        for col, stats in eng.card_aggregations.aggregations.items():
            for stat in stats:
                cols.append(f"card_{stat}_{col.lower()}")

    return cols


def create_pipeline(cfg: FeaturesConfig) -> tuple[Pipeline, list[str]]:
    """Build the sklearn Pipeline and the expected feature columns.

    Args:
        cfg: Validated feature configuration.

    Returns:
        Tuple of ``(pipeline, feature_columns)`` where *feature_columns* is
        the ordered list of columns the pipeline should produce.
    """
    eng = cfg.engineered
    t_cfg = cfg.transaction

    steps: list[tuple[str, Any]] = []

    if eng.has_identity:
        steps.append(("identity", IdentityFlagExtractor()))

    if eng.email_domain:
        steps.append(("email", EmailFeatureExtractor()))

    freq_cols = eng.frequency_encoding
    if freq_cols:
        steps.append(("freq", FrequencyEncoder(columns=list(freq_cols))))

    # freq must run before device: it encodes DeviceInfo which device drops.
    steps.append(
        (
            "device",
            DeviceFeatureExtractor(
                use_os=eng.device_os,
                use_brand=eng.device_brand,
            ),
        )
    )
    steps.append(("mflags", MFlagEncoder()))
    steps.append(("time", TimeFeatureExtractor(use_sin_cos=eng.cyclical)))

    if eng.id_codes_encoding and eng.id_codes:
        steps.append(("idcodes", IdCodeEncoder(columns=list(eng.id_codes))))

    if eng.card_aggregations:
        steps.append(
            (
                "card",
                CardAggregator(
                    group_by=list(eng.card_aggregations.group_by),
                    aggregations=eng.card_aggregations.aggregations,
                ),
            )
        )

    if eng.log_amount or eng.is_round_amount:
        steps.append(
            (
                "amount",
                AmountFeatureExtractor(
                    use_log=eng.log_amount,
                    use_round=eng.is_round_amount,
                ),
            )
        )

    v_cfg = t_cfg.vesta_features
    include_v = v_cfg is not None and v_cfg.include
    if v_cfg is not None and v_cfg.include:
        steps.append(
            (
                "vfilter",
                VFeatureFilter(
                    variance_threshold=v_cfg.variance_threshold,
                    correlation_threshold=v_cfg.correlation_threshold,
                    prefixes=("V",),
                ),
            )
        )

    c_cfg = t_cfg.count_corr_filter
    if c_cfg is not None and c_cfg.include:
        steps.append(
            (
                "vfilter_c",
                VFeatureFilter(
                    variance_threshold=c_cfg.variance_threshold,
                    correlation_threshold=c_cfg.correlation_threshold,
                    prefixes=("C",),
                ),
            )
        )

    steps.append(("imputer", MissingImputer()))
    steps.append(("dtypes", DtypeOptimizer()))

    feature_columns = _build_feature_columns(cfg)
    steps.append(
        (
            "selector",
            FeatureSelector(
                feature_columns=feature_columns,
                target=t_cfg.target,
                vesta_include=include_v,
            ),
        )
    )

    pipeline = Pipeline(steps)
    return pipeline, feature_columns


def load_data(data_cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read training transaction and identity DataFrames from processed Parquet.

    Args:
        data_cfg: Validated data configuration.

    Returns:
        Tuple of ``(transaction_df, identity_df)``.
    """
    processed_dir = Path(data_cfg.processed_dir)
    train = pd.read_parquet(processed_dir / "train_transaction.parquet")
    identity = pd.read_parquet(processed_dir / "train_identity.parquet")
    return train, identity


def merge_tables(
    transaction: pd.DataFrame, identity: pd.DataFrame, join_key: str = "TransactionID"
) -> pd.DataFrame:
    """Merge transaction e identity on TransactionID column

    Args:
        transaction (pd.DataFrame): transaction dataset
        identity (pd.DataFrame): identity datasat
        join_key (str, optional): column to use as merge join key. Defaults to "TransactionID".

    Returns:
        pd.DataFrame: Merged dataset
    """
    return transaction.merge(identity, on=join_key, how="left")
