from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.datasets import make_classification

import mlflow
from fdml.features.category_encoder import CategoryEncoder
from fdml.models.config import load_train_config
from fdml.models.evaluate.metrics import compute_metrics
from fdml.models.split import StratifiedSplitter, TemporalSplitter
from fdml.models.train.callbacks import IterationCallback
from fdml.models.train.model_builder import model_builder_registry
from fdml.models.train.runner import (
    _cap_train_fold,
    _fit_model,
    _get_splitter,
    _sample_eval_set,
)


class TestCapTrainFold:
    @pytest.fixture()
    def temporal_cfg(self):
        return load_train_config()

    @pytest.fixture()
    def random_cfg(self):
        return load_train_config(cli_args=["split.strategy=random"])

    @staticmethod
    def _data() -> tuple[pd.DataFrame, pd.Series]:
        X = pd.DataFrame(
            {
                "TransactionDT": [100, 50, 300, 200, 150],
                "x": [1, 2, 3, 4, 5],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])
        return X, y

    def test_temporal_keeps_earliest_rows_by_time(self, temporal_cfg):
        X, y = self._data()
        Xc, yc = _cap_train_fold(temporal_cfg, X, y, cap=2)
        assert Xc["TransactionDT"].tolist() == [50, 100]
        assert yc.tolist() == [1, 0]

    def test_random_keeps_head_rows(self, random_cfg):
        X, y = self._data()
        Xc, yc = _cap_train_fold(random_cfg, X, y, cap=3)
        assert Xc["TransactionDT"].tolist() == [100, 50, 300]
        assert yc.tolist() == [0, 1, 0]

    def test_cap_greater_than_size_keeps_all(self, temporal_cfg):
        X, y = self._data()
        Xc, yc = _cap_train_fold(temporal_cfg, X, y, cap=99)
        assert len(Xc) == 5
        assert sorted(Xc["TransactionDT"]) == sorted(X["TransactionDT"])
        assert len(yc) == 5


class TestGetSplitter:
    def test_temporal(self):
        cfg = load_train_config().split
        s = _get_splitter(cfg)
        assert isinstance(s, TemporalSplitter)
        assert s.time_col == "TransactionDT"
        assert s.embargo_seconds == cfg.embargo_seconds

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


class TestSampleEvalSet:
    @staticmethod
    def _val() -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
        y = pd.Series((rng.random(5000) < 0.035).astype(int))
        return X, y

    def test_subsample_respects_max_rows(self):
        X, y = self._val()
        X_es, y_es = _sample_eval_set(X, y, max_rows=2000)
        assert len(y_es) == 2000

    def test_stratified_keeps_class_ratio(self):
        X, y = self._val()
        X_es, y_es = _sample_eval_set(X, y, max_rows=2000)
        assert y.mean() > 0
        assert np.isclose(y_es.mean(), y.mean(), atol=0.01)

    def test_fixed_seed_deterministic(self):
        X, y = self._val()
        first = _sample_eval_set(X, y, max_rows=2000)
        second = _sample_eval_set(X, y, max_rows=2000)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_series_equal(first[1], second[1])

    def test_returns_full_when_smaller_or_zero(self):
        X, y = self._val()
        X_es, y_es = _sample_eval_set(X, y, max_rows=0)
        assert len(y_es) == 5000
        X_es, y_es = _sample_eval_set(X, y, max_rows=99999)
        assert len(y_es) == 5000


class TestFitModel:
    @staticmethod
    def _synth() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X, y = make_classification(
            n_samples=800, n_features=10, weights=[0.95, 0.05], random_state=0
        )
        Xv, yv = make_classification(
            n_samples=300, n_features=10, weights=[0.95, 0.05], random_state=1
        )
        return (
            pd.DataFrame(X),
            pd.Series(y),
            pd.DataFrame(Xv),
            pd.Series(yv),
        )

    def test_lgbm_early_stopping_sets_best_iteration(self):
        cfg = load_train_config(
            cli_args=["model.n_estimators=200", "early_stopping.rounds=20"]
        )
        X, y, Xv, yv = self._synth()
        model = model_builder_registry.build("lightgbm", cfg.model.params.model_dump())
        model = _fit_model(model, X, y, Xv, yv, cfg, callbacks=None)
        assert model.best_iteration_ > 0
        assert model.n_estimators_ < 200

    def test_lgbm_logs_iteration_metrics_to_mlflow(self, tmp_path):
        cfg = load_train_config(
            cli_args=["model.n_estimators=50", "early_stopping.rounds=10"]
        )
        X, y, Xv, yv = self._synth()
        model = model_builder_registry.build("lightgbm", cfg.model.params.model_dump())

        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("test-fit")
        client = MlflowClient(tracking_uri=uri)
        with mlflow.start_run():
            _fit_model(
                model,
                X,
                y,
                Xv,
                yv,
                cfg,
                callbacks=[IterationCallback(log_mlflow=True, log_console=False)],
            )
            hist = client.get_metric_history(mlflow.active_run().info.run_id, "val/auc")
        assert len(hist) > 0

    def test_disabled_early_stopping_fits_plain(self):
        cfg = load_train_config(cli_args=["early_stopping.enabled=false"])
        X, y, Xv, yv = self._synth()
        model = model_builder_registry.build("lightgbm", cfg.model.params.model_dump())
        model = _fit_model(model, X, y, Xv, yv, cfg, callbacks=None)
        assert model.n_estimators_ == cfg.model.params.n_estimators


class TestCategoryEncoder:
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
        encoder = CategoryEncoder().fit(train)
        result_train = encoder.transform(train)
        result_val = encoder.transform(val)

        assert result_train["num"].dtype.name == "int64"
        assert result_train["obj"].dtype.name == "int32"
        assert result_train["cat"].dtype.name == "int32"
        assert result_train["obj"].tolist() == [0, 1, 2]
        assert result_train["cat"].tolist() == [0, 1, 2]
        assert result_val["obj"].dtype.name == "int32"
        assert result_val["cat"].dtype.name == "int32"

    def test_unknown_category_maps_to_neg_one(self, df_with_cats):
        train, val = df_with_cats
        encoder = CategoryEncoder().fit(train)
        result_val = encoder.transform(val)

        assert result_val["obj"].iloc[1] == -1
        assert result_val["cat"].iloc[1] == -1

    def test_no_cat_cols_returns_unchanged(self):
        train = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        val = pd.DataFrame({"a": [5], "b": [6.0]})
        encoder = CategoryEncoder().fit(train)
        pd.testing.assert_frame_equal(encoder.transform(train), train)
        pd.testing.assert_frame_equal(encoder.transform(val), val)

    def test_deterministic_sorted_categories(self):
        train = pd.DataFrame({"obj": ["c", "a", "b", "a"]})
        first = CategoryEncoder().fit(train).transform(train)["obj"].tolist()
        second = CategoryEncoder().fit(train).transform(train)["obj"].tolist()
        assert first == second == [2, 0, 1, 0]


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
