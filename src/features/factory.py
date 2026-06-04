from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf, DictConfig
from sklearn.pipeline import Pipeline

from src.features.device import DeviceFeatureExtractor
from src.features.flags import MFlagEncoder
from src.features.time import TimeFeatureExtractor
from src.features.freq import FrequencyEncoder
from src.features.card import CardAggregator
from src.features.vfilter import VFeatureFilter
from src.features.imputer import MissingImputer
from src.features.selector import FeatureSelector


def load_config(path: str = "configs/features.yaml") -> DictConfig:
    cfg = OmegaConf.load(path)
    assert cfg.get("transaction"), "Missing 'transaction' in config"
    return cfg


def _build_feature_columns(cfg: DictConfig) -> list[str]:
    """Build the ordered list of expected feature columns from config."""
    cols: list[str] = []
    t = cfg.transaction
    cols.extend(t.get("numerical", []))
    cols.extend(t.get("categorical", []))
    cols.extend(t.get("high_cardinality", []))
    cols.extend(t.get("count_features", []))
    cols.extend(t.get("delta_features", []))

    identity_cfg = cfg.get("identity", {})
    cols.extend(identity_cfg.get("categorical", []))
    cols.extend(identity_cfg.get("id_features", []))

    eng = cfg.engineered
    cols.extend(eng.get("datetime", []))

    cols.append("device_os")
    cols.append("device_brand")

    for freq_col in eng.get("frequency_encoding", []):
        cols.append(f"{freq_col}_freq")

    if "M4" in t.get("categorical", []):
        cols.append("M4")

    card_aggs = eng.get("card_aggregations", {})
    if card_aggs:
        for col, stats in card_aggs.get("aggregations", {}).items():
            for stat in stats:
                cols.append(f"card_{stat}_{col.lower()}")

    return cols


def create_pipeline(cfg: DictConfig) -> tuple[Pipeline, list[str]]:
    """Build the sklearn Pipeline and the expected feature columns.

    Args:
        cfg: OmegaConf config (from ``configs/features.yaml``).

    Returns:
        Tuple of ``(pipeline, feature_columns)`` where *feature_columns* is
        the ordered list of columns the pipeline should produce.
    """
    eng = cfg.engineered
    t_cfg = cfg.transaction
    card_aggs = eng.get("card_aggregations", {})

    steps = [
        ("device", DeviceFeatureExtractor()),
        ("mflags", MFlagEncoder()),
        ("time", TimeFeatureExtractor()),
    ]

    freq_cols = eng.get("frequency_encoding", [])
    if freq_cols:
        steps.append(("freq", FrequencyEncoder(columns=list(freq_cols))))

    if card_aggs:
        steps.append(
            (
                "card",
                CardAggregator(
                    group_by=list(card_aggs.get("group_by", [])),
                    aggregations=dict(card_aggs.get("aggregations", {})),
                ),
            )
        )

    v_cfg = t_cfg.vesta_features
    if v_cfg.include:
        steps.append(
            (
                "vfilter",
                VFeatureFilter(
                    variance_threshold=v_cfg.variance_threshold,
                    correlation_threshold=v_cfg.correlation_threshold,
                ),
            )
        )

    steps.append(("imputer", MissingImputer()))

    feature_columns = _build_feature_columns(cfg)
    target = t_cfg.target
    steps.append(
        (
            "selector",
            FeatureSelector(
                feature_columns=feature_columns,
                target=target,
                vesta_include=v_cfg.include,
            ),
        )
    )

    pipeline = Pipeline(steps)
    return pipeline, feature_columns


def load_data(data_cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read training transaction and identity DataFrames from processed Parquet.

    Args:
        data_cfg: OmegaConf config with ``processed_dir`` key.

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
    return transaction.merge(identity, on=join_key, how="left")
