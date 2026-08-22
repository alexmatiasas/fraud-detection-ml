from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from fdml.models.evaluate.stability import (
    feature_importance_stability,
    learning_curves,
)


@pytest.fixture()
def small_dataset():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.95, 0.05]))
    return X, y


def _make_pipe():
    def model_fn():
        return LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=42, verbose=-1, n_jobs=1
        )

    pipe = Pipeline([("model", model_fn())])
    return pipe, model_fn


class TestLearningCurves:
    def test_creates_plot(self, small_dataset, tmp_path):
        X, y = small_dataset
        features = Pipeline([("features", "passthrough")])

        def model_fn():
            return LGBMClassifier(
                n_estimators=10, max_depth=3, random_state=42, verbose=-1, n_jobs=1
            )

        path = learning_curves(
            X,
            y,
            X,
            y,
            features,
            model_fn,
            train_sizes=[0.2, 0.5, 1.0],
            output_dir=str(tmp_path),
        )
        assert path.exists()
        assert path.name == "learning_curves.png"

    def test_default_sizes(self, small_dataset, tmp_path):
        X, y = small_dataset
        features = Pipeline([("features", "passthrough")])

        def model_fn():
            return LGBMClassifier(
                n_estimators=10, max_depth=3, random_state=42, verbose=-1, n_jobs=1
            )

        path = learning_curves(X, y, X, y, features, model_fn, output_dir=str(tmp_path))
        assert path.exists()


class TestFeatureImportanceStability:
    def test_creates_plot(self, small_dataset, tmp_path):
        X, y = small_dataset
        features = Pipeline([("passthrough", "passthrough")])
        model = LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=42, verbose=-1, n_jobs=1
        )
        pipe = Pipeline([("features", features), ("model", model)])
        pipe.fit(X, y)

        path = feature_importance_stability(
            model=model,
            feature_names=X.columns.tolist(),
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            pipeline=pipe,
            n_iterations=3,
            output_dir=str(tmp_path),
        )
        assert path.exists()
        assert path.name == "feature_importance_stability.png"

    def test_with_xgb(self, small_dataset, tmp_path):
        from xgboost import XGBClassifier

        X, y = small_dataset
        features = Pipeline([("passthrough", "passthrough")])
        model = XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42, verbosity=0
        )
        pipe = Pipeline([("features", features), ("model", model)])
        pipe.fit(X, y)

        path = feature_importance_stability(
            model=model,
            feature_names=X.columns.tolist(),
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            pipeline=pipe,
            n_iterations=2,
            output_dir=str(tmp_path),
        )
        assert path.exists()
