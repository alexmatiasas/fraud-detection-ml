# TODO: refactor with sklearn classes

import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pandera.pandas import Column, DataFrameSchema

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DTYPE_MAP: dict[str, str] = {
    "isFraud": "int8",
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    "ProductCD": "category",
    "dist1": "float32",
    "dist2": "float32",
    "card1": "int16",
    "card2": "float32",
    "card3": "float32",
    "card4": "category",
    "card5": "float32",
    "card6": "category",
    "addr1": "float32",
    "addr2": "float32",
    "P_emaildomain": "category",
    "R_emaildomain": "category",
    "C1": "float32",
    "C2": "float32",
    "C4": "float32",
    "C5": "float32",
    "C6": "float32",
    "C7": "float32",
    "C8": "float32",
    "C9": "float32",
    "C10": "float32",
    "C11": "float32",
    "C12": "float32",
    "C13": "float32",
    "C14": "float32",
    "D1": "float32",
    "D2": "float32",
    "D3": "float32",
    "D4": "float32",
    "D5": "float32",
    "D10": "float32",
    "D11": "float32",
    "D15": "float32",
    "M4": "category",
    "DeviceType": "category",
    "DeviceInfo": "category",
    "id_01": "float32",
    "id_02": "float32",
    "id_03": "float32",
    "id_04": "float32",
    "id_05": "float32",
    "id_06": "float32",
    "id_07": "float32",
    "id_08": "float32",
    "id_09": "float32",
    "id_10": "float32",
    "id_11": "float32",
    "id_12": "category",
    "id_13": "float32",
    "id_14": "float32",
    "id_15": "category",
    "id_16": "category",
    "id_17": "float32",
    "id_18": "float32",
    "id_19": "float32",
    "id_20": "float32",
    "id_21": "float32",
    "id_22": "float32",
    "id_23": "category",
    "id_24": "float32",
    "id_25": "float32",
    "id_26": "float32",
    "id_27": "category",
    "id_28": "category",
    "id_29": "category",
    "id_30": "category",
    "id_31": "category",
    "id_32": "float32",
    "id_33": "category",
    "id_34": "category",
    "id_35": "category",
    "id_36": "category",
    "id_37": "category",
    "id_38": "category",
}


def load_config() -> DictConfig | ListConfig:
    """Loads configuration {keys: values} from de configs/data.yaml
    configuration file

    - raw_dir: is the directory where the data is storaged from the
    source
    [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset
    - target: is the target feature (IsFraud) in the dataset

    See configs/data.yml in the root dir of the project for more

    Returns:
        DictConfig | ListConfig: The configuration is loaded as a dictionary
    """
    cfg = OmegaConf.load("configs/data.yaml")
    assert cfg.get("raw_dir"), "raw_dir missing in config"
    assert cfg.get("target"), "target missing in config"
    return cfg


def build_dtype_map(csv_path: Path) -> dict[str, str]:
    dtype_map = DTYPE_MAP.copy()
    cols = pd.read_csv(csv_path, nrows=0).columns
    v_cols = [c for c in cols if c.startswith("V")]
    for v in v_cols:
        dtype_map.setdefault(v, "float32")
    return dtype_map


def build_transaction_schema(cfg: DictConfig) -> DataFrameSchema:
    return DataFrameSchema(
        {
            cfg.join_key: Column(int, nullable=False),
            cfg.target: Column(int, pa.Check.isin(list(cfg.validation.target_values))),
            "TransactionAmt": Column(float, pa.Check.greater_than(0)),
        },
        strict=False,
    )


def build_identity_schema(cfg: DictConfig) -> DataFrameSchema:
    return DataFrameSchema(
        {
            cfg.join_key: Column(int, nullable=False, unique=True),
        },
        strict=False,
    )


def _downcast(df: pd.DataFrame, dtype_map: dict[str, str]) -> None:
    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "category":
                df[col] = df[col].astype("category")
            elif dtype in ("int8", "int16", "int32"):
                if not df[col].isna().any():
                    df[col] = df[col].astype(dtype)
            elif dtype == "float32":
                df[col] = df[col].astype("float32")
        except (ValueError, TypeError):
            logger.warning(f"  Could not convert {col} to {dtype}")


def convert_to_parquet(cfg: DictConfig) -> None:
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
        elif name == "train_identity":
            logger.info("  Validating identity schema...")
            schema = build_identity_schema(cfg)
            schema.validate(df)
            logger.info("  Schema OK")

        logger.info("  Downcasting...")
        dtype_map = build_dtype_map(src)
        _downcast(df, dtype_map)

        df.to_parquet(dest, index=False, engine="pyarrow")

        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info(f"  {len(df):,} rows, {len(df.columns)} cols → {size_mb:.1f} MB")

    logger.info("Done.")


def main() -> None:
    cfg = load_config()
    convert_to_parquet(cfg)


if __name__ == "__main__":
    main()
