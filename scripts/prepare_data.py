from pathlib import Path

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
import pandas as pd
import pandera.pandas as pa
from omegaconf import OmegaConf
from pandera.pandas import Column, DataFrameSchema

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config() -> DictConfig | ListConfig:
    cfg = OmegaConf.load("configs/data.yaml")
    assert cfg.get("raw_dir"), "raw_dir missing in config"
    assert cfg.get("target"), "target missing in config"
    return cfg


def build_transaction_schema(cfg) -> DataFrameSchema:
    return DataFrameSchema(
        {
            cfg.join_key: Column(int, nullable=False),
            cfg.target: Column(int, pa.Check.isin(list(cfg.validation.target_values))),
            "TransactionAmt": Column(float, pa.Check.greater_than(0)),
        },
        strict=False,
    )


def convert_to_parquet(cfg) -> None:
    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"{raw_dir} is empty or does not exist. "
            "Run 'python scripts/download_data.py' first."
        )

    files = OmegaConf.to_container(cfg.files)

    for name, filename in files.items():
        src = raw_dir / filename
        dest = processed_dir / f"{name}.parquet"

        if dest.exists():
            logger.info(f"Skipped {name} (already exists)")
            continue

        logger.info(f"Converting {name}...")
        df = pd.read_csv(src)

        if name == "train_transaction":
            logger.info("  Validating schema...")
            schema = build_transaction_schema(cfg)
            schema.validate(df)
            logger.info("  Schema OK")

        df.to_parquet(dest, index=False, engine="pyarrow")

        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info(f"  {len(df):,} rows, {len(df.columns)} cols → {size_mb:.1f} MB")

    logger.info("Done.")


def main() -> None:
    cfg = load_config()
    convert_to_parquet(cfg)


if __name__ == "__main__":
    main()
