"""Unit tests for the ModelLoader and its resolution logic."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fdml.api.internal.loader import (
    ModelLoadError,
    ModelLoader,
    to_model_input,
)

DROP_FOR_PREDICT = ("TransactionID", "isFraud")


class FakePipeline:
    """Minimal stand-in for the sklearn Pipeline (features + model)."""

    def __init__(self) -> None:
        self.named_steps = {"features": object(), "model": object()}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.full((n, 2), 0.35)


def make_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "TransactionID": 7,
                "TransactionDT": 18_403_224,
                "isFraud": 0,
                "TransactionAmt": 125.5,
                "ProductCD": "W",
                "card1": 13_000,
                "P_emaildomain": "gmail.com",
            }
        ]
    )


def test_to_model_input_drops_identifier_and_target() -> None:
    row = make_row()
    out = to_model_input(row)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, row.shape[1] - len(DROP_FOR_PREDICT))
    assert "TransactionID" not in out.columns
    assert "isFraud" not in out.columns
    assert out.loc[0, "TransactionAmt"] == 125.5


def test_to_model_input_preserves_numeric_dtypes() -> None:
    row = make_row().astype({"TransactionDT": "int32", "TransactionAmt": "float32"})
    out = to_model_input(row)
    assert out["TransactionDT"].dtype != object
    assert out["TransactionAmt"].dtype != object


def test_predict_proba_uses_cached_pipeline() -> None:
    loader = ModelLoader()
    loader._pipeline = FakePipeline()
    proba = loader.predict_proba(pd.DataFrame({"x": [1.0]}))
    assert proba.shape == (1,)
    assert proba[0] == pytest.approx(0.35)


def test_predict_proba_raises_when_not_loaded() -> None:
    loader = ModelLoader()
    with pytest.raises(ModelLoadError, match="not loaded"):
        loader.predict_proba(pd.DataFrame({"x": [1.0]}))


def test_lookup_returns_none_outside_sample() -> None:
    loader = ModelLoader()
    loader._sample = pd.DataFrame({"TransactionID": [1, 2], "x": [0.1, 0.2]})
    row_1 = loader.lookup(1)
    row_2 = loader.lookup(2)
    assert row_1 is not None
    assert row_2 is not None
    assert row_1.iloc[0]["x"] == pytest.approx(0.1)
    assert row_2.iloc[0]["x"] == pytest.approx(0.2)
    assert loader.lookup(999) is None


def test_lookup_preserves_frame_shape() -> None:
    loader = ModelLoader()
    loader._sample = pd.DataFrame(
        {"TransactionID": [1], "x": [0.1], "TransactionDT": [18403224]}
    )
    match = loader.lookup(1)
    assert isinstance(match, pd.DataFrame)
    assert match.shape == (1, 3)
    assert match["TransactionDT"].dtype != object


def test_lookup_none_when_sample_missing() -> None:
    loader = ModelLoader()
    assert loader.sample is None
    assert loader.lookup(1) is None


def test_report_loaded_from_json(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"model_name": "lightgbm", "best_threshold": 0.8}')
    pipeline_path = tmp_path / "pipeline.joblib"
    joblib.dump(FakePipeline(), pipeline_path)

    loader = ModelLoader(
        local_pipeline=str(pipeline_path), report_path=str(report_path)
    )
    loader.load(source="local")
    assert loader.source == "local"
    assert loader.report["model_name"] == "lightgbm"
    assert loader.report["best_threshold"] == 0.8


def test_load_from_local_missing_pipeline_raises(tmp_path: Path) -> None:
    loader = ModelLoader(
        local_pipeline=str(tmp_path / "nope.joblib"),
        report_path=str(tmp_path / "report.json"),
    )
    with pytest.raises(ModelLoadError, match="No model available"):
        loader.load(source="local")


def test_load_is_idempotent_without_force() -> None:
    loader = ModelLoader()
    loader._pipeline = FakePipeline()
    first = loader.loaded_at
    loader.load(source="local")
    assert loader.loaded_at is first


def test_load_sample_sets_dataframe(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.parquet"
    pd.DataFrame({"TransactionID": [1], "isFraud": [0]}).to_parquet(sample_path)
    loader = ModelLoader(sample_path=str(sample_path))
    loader.load_sample()
    assert loader.sample is not None
    assert len(loader.sample) == 1


def test_load_sample_missing_is_tolerated(tmp_path: Path) -> None:
    loader = ModelLoader(sample_path=str(tmp_path / "nope.parquet"))
    loader.load_sample()
    assert loader.sample is None


def test_model_version_marks_mlflow_source() -> None:
    loader = ModelLoader()
    loader._source = "mlflow"
    loader._version = "3"
    assert loader.model_version == "fraud-detection-lgbm:3"
    loader._source = "local"
    assert loader.model_version is None
