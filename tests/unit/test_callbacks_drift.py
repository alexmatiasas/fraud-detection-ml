from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fdml.models.evaluate.drift import adversarial_validation
from fdml.models.train.callbacks import (
    IterationCallback,
    XGBoostIterationCallback,
    _higher_is_better,
    _metric_key,
    _normalize_dataset_name,
)


class TestAdversarialValidation:
    def _make_data(self, n=200):
        rng = np.random.default_rng(42)
        cols = [f"f{i}" for i in range(5)]
        X_train = pd.DataFrame(rng.standard_normal((n, 5)), columns=cols)
        X_val = pd.DataFrame(rng.standard_normal((n, 5)), columns=cols)
        return X_train, X_val

    def test_returns_auc_and_path(self, tmp_path):
        X_train, X_val = self._make_data()
        auc, path = adversarial_validation(X_train, X_val, output_dir=str(tmp_path))
        assert 0.0 <= auc <= 1.0
        assert path.exists()
        assert path.name == "adversarial_validation.png"

    def test_handles_categorical_columns(self, tmp_path):
        X_train, X_val = self._make_data()
        X_train["cat_col"] = pd.Categorical(["a", "b"] * 100)
        X_val["cat_col"] = pd.Categorical(["a", "c"] * 100)
        auc, path = adversarial_validation(X_train, X_val, output_dir=str(tmp_path))
        assert 0.0 <= auc <= 1.0
        assert path.exists()

    def test_handles_missing_values(self, tmp_path):
        X_train, X_val = self._make_data()
        X_train.iloc[0, 0] = np.nan
        X_val.iloc[0, 0] = np.nan
        auc, _ = adversarial_validation(X_train, X_val, output_dir=str(tmp_path))
        assert 0.0 <= auc <= 1.0


class TestCallbacksHelpers:
    def test_higher_is_better_known(self):
        assert _higher_is_better("auc") is True
        assert _higher_is_better("logloss") is False
        assert _higher_is_better("rmse") is False

    def test_higher_is_better_unknown_default(self):
        assert _higher_is_better("unknown_metric") is True
        assert _higher_is_better("unknown_metric", default=False) is False

    def test_metric_key_validation(self):
        assert _metric_key("validation", "auc") == "val/auc"
        assert _metric_key("valid", "auc") == "val/auc"
        assert _metric_key("val", "auc") == "val/auc"

    def test_metric_key_training(self):
        assert _metric_key("train", "auc") == "train/auc"
        assert _metric_key("training", "auc") == "train/auc"

    def test_metric_key_other(self):
        assert _metric_key("test", "auc") == "test_auc"

    def test_normalize_dataset_name(self):
        assert _normalize_dataset_name("validation_0") == "validation"
        assert _normalize_dataset_name("validation_1") == "validation"
        assert _normalize_dataset_name("train") == "train"

    def test_normalize_dataset_name_no_change(self):
        assert _normalize_dataset_name("validation") == "validation"


class TestIterationCallback:
    def test_calls_log_metric(self):
        cb = IterationCallback(log_mlflow=False, log_console=False)
        env = SimpleNamespace(
            iteration=1,
            evaluation_result_list=[("validation", "auc", 0.9, True)],
        )
        cb(env)

    def test_ignores_empty_env(self):
        cb = IterationCallback(log_mlflow=False)
        env = SimpleNamespace(iteration=1, evaluation_result_list=[])
        cb(env)

    def test_ignores_env_without_attr(self):
        cb = IterationCallback(log_mlflow=False)
        env = SimpleNamespace(iteration=1)
        cb(env)

    def test_console_logging(self, caplog):
        cb = IterationCallback(log_mlflow=False, log_console=True, console_every=1)
        env = SimpleNamespace(
            iteration=1,
            evaluation_result_list=[("validation", "auc", 0.9, True)],
        )
        import logging

        with caplog.at_level(logging.INFO, logger="fdml.models.train.callbacks"):
            cb(env)
        assert "auc=0.9000" in caplog.text

    def test_three_item_tuple_uses_higher_is_better(self):
        cb = IterationCallback(log_mlflow=False, log_console=False)
        env = SimpleNamespace(
            iteration=1,
            evaluation_result_list=[("validation", "logloss", 0.5)],
        )
        cb(env)


class TestXGBoostIterationCallback:
    def test_after_iteration_returns_false(self):
        cb = XGBoostIterationCallback(log_mlflow=False)
        evals_log = {"validation_0": {"auc": [0.9]}}
        result = cb.after_iteration(None, 0, evals_log)
        assert result is False

    def test_with_name_map(self):
        cb = XGBoostIterationCallback(
            log_mlflow=False, name_map={"validation_0": "validation"}
        )
        evals_log = {"validation_0": {"auc": [0.9]}}
        result = cb.after_iteration(None, 0, evals_log)
        assert result is False

    def test_empty_metrics(self):
        cb = XGBoostIterationCallback(log_mlflow=False)
        evals_log = {"validation_0": {}}
        result = cb.after_iteration(None, 0, evals_log)
        assert result is False

    def test_as_xgboost_creates_callback(self):
        lgb_cb = IterationCallback(log_mlflow=False)
        xgb_cb = lgb_cb.as_xgboost()
        assert isinstance(xgb_cb, XGBoostIterationCallback)

    def test_close_no_live(self):
        cb = IterationCallback(log_mlflow=False)
        cb.close()
