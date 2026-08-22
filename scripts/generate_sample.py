"""Generate a small stratified sample of the validation fold for API demos.

The Kaggle test set has no ``isFraud`` labels, so the demo sample is drawn
from the temporal validation fold of the training data — the same fold the
evaluation report (``models/report.json``) was computed on. The sample keeps
the original fraud rate (~3.4%) so fraud cases are visible in the UI.

Output: ``data/samples/val_sample.parquet`` — full merged raw schema
(434 columns: transaction + identity) plus ``isFraud``.

Usage:
    uv run scripts/generate_sample.py --size 5000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fdml.features.factory import load_data, load_data_config, merge_tables
from fdml.models.config import load_train_config
from fdml.models.split import TemporalSplitter

DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_OUTPUT = "data/samples/val_sample.parquet"


def build_validation_set(train_cfg: object, data_cfg: object) -> pd.DataFrame:
    """Recreate the temporal validation fold exactly as the evaluator does."""
    train_df, identity_df = load_data(data_cfg)
    df = merge_tables(train_df, identity_df)

    splitter = TemporalSplitter(
        time_col=train_cfg.split.time_col,
        test_size=train_cfg.split.test_size,
        embargo_seconds=train_cfg.split.embargo_seconds,
    )
    _, val_idx = next(splitter.split(df, df["isFraud"]))
    return df.iloc[val_idx].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Sample preserving the original fraud rate.

    Samples ``n_fraud`` rows from the positive class (proportional to the
    dataset fraud rate) and the remainder from the negative class, then
    shuffles so fraud rows are not clustered at the top.
    """
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    n_fraud = round(size * len(fraud) / len(df))
    n_fraud = max(1, min(n_fraud, len(fraud)))
    n_legit = max(0, size - n_fraud)

    sampled = pd.concat(
        [
            fraud.sample(n=n_fraud, random_state=seed),
            legit.sample(n=n_legit, random_state=seed),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a stratified demo sample from the validation fold"
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    train_cfg = load_train_config()
    data_cfg = load_data_config()

    val = build_validation_set(train_cfg, data_cfg)
    sample = stratified_sample(val, args.size, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)

    print(
        f"Sample: {len(sample):,} rows | {sample.shape[1]} cols | "
        f"fraud rate {sample['isFraud'].mean():.4f} -> {out}"
    )


if __name__ == "__main__":
    main()
