from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from omegaconf import OmegaConf
from pandera.pandas import Column, DataFrameSchema


def load_config():
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

    files = OmegaConf.to_container(cfg.files)

    for name, filename in files.items():
        src = raw_dir / filename
        dest = processed_dir / f"{name}.parquet"

        if dest.exists():
            print(f"Skipped {name} (already exists)")
            continue

        print(f"Converting {name}...")
        df = pd.read_csv(src)

        if name == "train_transaction":
            print("  Validating schema...")
            schema = build_transaction_schema(cfg)
            schema.validate(df)
            print("  Schema OK")

        df.to_parquet(dest, index=False, engine="pyarrow")

        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  {len(df):,} rows, {len(df.columns)} cols → {size_mb:.1f} MB")

    print("Done.")


def main() -> None:
    cfg = load_config()
    convert_to_parquet(cfg)


if __name__ == "__main__":
    main()
