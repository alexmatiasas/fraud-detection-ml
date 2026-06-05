from pathlib import Path

import pandas as pd

from src.features.factory import (
    create_pipeline,
    load_config,
    load_data,
    load_data_config,
    merge_tables,
)

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    data_cfg = load_data_config()
    processed_dir = Path(data_cfg.processed_dir)
    features_dir = Path(data_cfg.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    identity_path = processed_dir / "train_identity.parquet"
    if identity_path.exists():
        train_df, identity_df = load_data(data_cfg)
        df = merge_tables(train_df, identity_df)
        logger.info(f"Merged: {df.shape}")
    else:
        logger.info("Identity file not found, loading transaction data only")
        df = pd.read_parquet(processed_dir / "train_transaction.parquet")

    logger.info(f"Base shape: {df.shape}")

    pipeline, _ = create_pipeline(cfg)
    X = pipeline.fit_transform(df)
    y = df[["isFraud"]]

    X_path = features_dir / "X_train.feather"
    y_path = features_dir / "y_train.feather"
    X.reset_index(drop=True).to_feather(X_path)
    y.reset_index(drop=True).to_feather(y_path)

    logger.info(f"X: {X.shape}, y: {y.shape}")
    logger.info(f"Saved to {features_dir}")
