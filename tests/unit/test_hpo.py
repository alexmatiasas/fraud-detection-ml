from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import optuna
import pandas as pd
import pytest

from fdml.models.train.model_builder import model_builder_registry


class TestSearchSpace:
    @pytest.mark.parametrize(
        "name",
        ["lightgbm", "xgboost", "random_forest"],
    )
    def test_returns_dict(self, name: str):
        builder = model_builder_registry.get(name)
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 42
        trial.suggest_float.return_value = 0.5
        space = builder.get_search_space(trial)
        assert isinstance(space, dict)
        assert len(space) > 0

    @pytest.mark.parametrize(
        ("name", "expected_keys"),
        [
            (
                "lightgbm",
                [
                    "num_leaves",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "min_child_samples",
                    "reg_alpha",
                    "reg_lambda",
                ],
            ),
            (
                "xgboost",
                [
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                    "min_child_weight",
                    "reg_alpha",
                    "reg_lambda",
                ],
            ),
            (
                "random_forest",
                [
                    "n_estimators",
                    "max_depth",
                    "min_child_samples",
                    "subsample",
                ],
            ),
        ],
    )
    def test_lgbm_has_expected_keys(self, name: str, expected_keys: list[str]):
        builder = model_builder_registry.get(name)
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 42
        trial.suggest_float.return_value = 0.5
        space = builder.get_search_space(trial)
        assert list(space.keys()) == expected_keys

    def test_search_space_values_are_sampled_from_trial(self):
        builder = model_builder_registry.get("lightgbm")
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.side_effect = lambda name, low, high, **kw: (low + high) // 2
        trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
        space = builder.get_search_space(trial)
        assert space["num_leaves"] == 143
        assert space["learning_rate"] == pytest.approx(0.1505, rel=1e-3)


class TestObjective:
    def test_run_single_trial(self):
        builder = model_builder_registry.get("lightgbm")
        trial = MagicMock(spec=optuna.Trial)
        trial.suggest_int.return_value = 63
        trial.suggest_float.return_value = 0.1

        rng = np.random.default_rng(42)
        n = 500
        X = pd.DataFrame(
            {
                "feat1": rng.normal(size=n),
                "feat2": rng.normal(size=n),
                "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
            }
        )
        X = X.astype({"cat": "category"})
        y = pd.Series((rng.uniform(size=n) > 0.965).astype(int))

        from fdml.models.train.hpo import _objective

        score = _objective(
            trial,
            X,
            y,
            X,
            y,
            builder=builder,
            base_params={"scale_pos_weight": 27.6, "n_estimators": 10},
            es_rounds=5,
            es_metric="auc",
            seed=42,
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
