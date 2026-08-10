from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fdml.models.categoricals import encode_categoricals
from fdml.models.config import load_train_config
from fdml.models.evaluate.metrics import compute_metrics
from fdml.models.split import StratifiedSplitter, TemporalSplitter
from fdml.models.train.model_builder import model_builder_registry
from fdml.models.train.runner import _get_splitter


class TestGetSplitter:
    def test_temporal(self):
        cfg = load_train_config().split
        s = _get_splitter(cfg)
        assert isinstance(s, TemporalSplitter)
        assert s.time_col == "TransactionDT"

    def test_random(self):
        cfg = load_train_config(cli_args=["split.strategy=random"]).split
        s = _get_splitter(cfg)
        assert isinstance(s, StratifiedSplitter)

    def test_unknown_raises(self):
        class FakeCfg:
            strategy = "invalid"

        with pytest.raises(ValueError, match="Unknown split strategy"):
            _get_splitter(FakeCfg())


class TestBuildModel:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("lightgbm", "LGBMClassifier"),
            ("xgboost", "XGBClassifier"),
            ("random_forest", "RandomForestClassifier"),
        ],
    )
    def test_builds_known_models(self, name: str, expected: str):
        cfg = load_train_config(cli_args=[f"model.name={name}"]).model
        model = model_builder_registry.build(name, cfg.params.model_dump())
        assert model.__class__.__name__ == expected

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            model_builder_registry.build("unknown_model", {})

    @pytest.mark.parametrize(
        "name",
        ["lightgbm", "xgboost", "random_forest"],
    )
    def test_builder_honors_random_state(self, name: str):
        model = model_builder_registry.build(name, {"random_state": 7})
        assert model.random_state == 7

    def test_builder_default_random_state(self):
        model = model_builder_registry.build("lightgbm", {})
        assert model.random_state == 42


class TestEncodeCategoricals:
    @pytest.fixture()
    def df_with_cats(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train = pd.DataFrame(
            {
                "num": [1, 2, 3],
                "obj": pd.Series(["a", "b", "c"], dtype="object"),
                "cat": pd.Categorical(["x", "y", "z"]),
            }
        )
        val = pd.DataFrame(
            {
                "num": [4, 5],
                "obj": pd.Series(["b", "d"], dtype="object"),
                "cat": pd.Categorical(["y", "w"]),
            }
        )
        return train, val

    def test_encodes_object_and_category(self, df_with_cats):
        train, val = df_with_cats
        result_train, result_val = encode_categoricals(train, val)

        assert result_train["num"].dtype.name == "int64"
        assert result_train["obj"].dtype.name == "int32"
        assert result_train["cat"].dtype.name == "int32"

    def test_unknown_category_maps_to_neg_one(self, df_with_cats):
        train, val = df_with_cats
        _, result_val = encode_categoricals(train, val)

        assert result_val["obj"].iloc[1] == -1
        assert result_val["cat"].iloc[1] == -1

    def test_no_cat_cols_returns_unchanged(self):
        train = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        val = pd.DataFrame({"a": [5], "b": [6.0]})
        t, v = encode_categoricals(train, val)
        pd.testing.assert_frame_equal(t, train)
        pd.testing.assert_frame_equal(v, val)


class TestComputeMetrics:
    def test_returns_expected_keys(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        metrics = compute_metrics(y_true, y_proba)
        expected = {"roc_auc", "average_precision", "f1", "precision", "recall"}
        assert set(metrics.keys()) == expected

    def test_perfect_predictions(self):
        y_true = np.array([0, 1])
        y_proba = np.array([0.0, 1.0])
        metrics = compute_metrics(y_true, y_proba)
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0

    def test_worst_predictions(self):
        y_true = np.array([0, 1])
        y_proba = np.array([1.0, 0.0])
        metrics = compute_metrics(y_true, y_proba)
        assert metrics["roc_auc"] == 0.0
