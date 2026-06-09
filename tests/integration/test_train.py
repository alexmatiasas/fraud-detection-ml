from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.factory import create_pipeline, load_config as load_features_config
from src.models.categoricals import encode_categoricals
from src.models.config import load_train_config
from src.models.evaluate.metrics import compute_metrics
from src.models.train.model_builder import model_builder_registry
from src.models.train.runner import _get_splitter


@pytest.fixture()
def small_data() -> pd.DataFrame:
    processed_dir = Path(load_train_config().data.processed_dir)
    train = pd.read_parquet(
        processed_dir / "train_transaction.parquet",
        columns=[
            "TransactionID",
            "isFraud",
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card4",
            "card6",
            "C1",
            "C2",
            "D1",
            "D2",
            "V1",
        ],
    )
    identity_path = processed_dir / "train_identity.parquet"
    if identity_path.exists():
        identity = pd.read_parquet(identity_path)
        return train.merge(identity, on="TransactionID", how="left")
    return train


class TestTrainIntegration:
    def test_end_to_end_lightgbm(self, small_data: pd.DataFrame):
        df = small_data.head(5000)
        cfg = load_train_config()
        splitter = _get_splitter(cfg.split)
        train_idx, val_idx = next(splitter.split(df, df["isFraud"]))

        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline, _ = create_pipeline(load_features_config())
        X_train_fe = pipeline.fit_transform(X_train, y_train)
        X_val_fe = pipeline.transform(X_val)
        X_train_fe, X_val_fe = encode_categoricals(X_train_fe, X_val_fe)

        model = model_builder_registry.build(
            cfg.model.name, cfg.model.params.model_dump()
        )
        model.fit(X_train_fe, y_train)

        y_proba = model.predict_proba(X_val_fe)[:, 1]
        metrics = compute_metrics(y_val.values, y_proba)

        assert metrics["roc_auc"] > 0.5
        assert metrics["average_precision"] > 0.03

    @pytest.mark.parametrize("model_name", ["xgboost", "random_forest"])
    def test_end_to_end_other_models(self, small_data: pd.DataFrame, model_name: str):
        df = small_data.head(5000)
        cfg = load_train_config(cli_args=[f"model.name={model_name}"])
        splitter = _get_splitter(cfg.split)
        train_idx, val_idx = next(splitter.split(df, df["isFraud"]))

        X = df.drop(columns=["isFraud"])
        y = df["isFraud"]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline, _ = create_pipeline(load_features_config())
        X_train_fe = pipeline.fit_transform(X_train, y_train)
        X_val_fe = pipeline.transform(X_val)
        X_train_fe, X_val_fe = encode_categoricals(X_train_fe, X_val_fe)

        model = model_builder_registry.build(
            cfg.model.name, cfg.model.params.model_dump()
        )
        model.fit(X_train_fe, y_train)

        y_proba = model.predict_proba(X_val_fe)[:, 1]
        metrics = compute_metrics(y_val.values, y_proba)

        assert metrics["roc_auc"] > 0.5
        assert metrics["average_precision"] > 0.03
