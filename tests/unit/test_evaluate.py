from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fdml.models.evaluate import (
    bootstrap_ci,
    compute_metrics,
    per_segment_analysis,
    threshold_tuning,
)


@pytest.fixture()
def y_true() -> np.ndarray:
    return np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1] * 10)


@pytest.fixture()
def y_proba() -> np.ndarray:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 100)
    return np.clip(p, 0.01, 0.99)


class TestComputeMetrics:
    def test_returns_expected_keys(self, y_true, y_proba):
        metrics = compute_metrics(y_true, y_proba)
        assert set(metrics.keys()) == {
            "roc_auc",
            "average_precision",
            "f1",
            "precision",
            "recall",
        }

    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_metrics(y_true, y_proba)
        assert metrics["roc_auc"] == 1.0
        assert metrics["average_precision"] == 1.0

    def test_all_zeros(self):
        y_true = np.array([0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3])
        metrics = compute_metrics(y_true, y_proba)
        assert np.isnan(metrics["roc_auc"])

    def test_threshold_effect(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.4, 0.6, 0.4, 0.6])

        low = compute_metrics(y_true, y_proba, threshold=0.5)
        high = compute_metrics(y_true, y_proba, threshold=0.7)

        assert low["recall"] > high["recall"]


class TestBootstrapCI:
    def test_ci_for_average_precision(self, y_true, y_proba):
        lower, upper = bootstrap_ci(
            y_true, y_proba, metric="average_precision", n_iterations=50
        )
        assert 0.0 <= lower <= upper <= 1.0

    def test_ci_for_roc_auc(self, y_true, y_proba):
        lower, upper = bootstrap_ci(y_true, y_proba, metric="roc_auc", n_iterations=50)
        assert 0.0 <= lower <= upper <= 1.0

    def test_ci_includes_point_estimate(self, y_true, y_proba):
        from sklearn.metrics import average_precision_score

        point = average_precision_score(y_true, y_proba)
        lower, upper = bootstrap_ci(
            y_true, y_proba, metric="average_precision", n_iterations=100
        )
        assert lower <= point <= upper

    def test_fewer_iterations(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        lower, upper = bootstrap_ci(y_true, y_proba, n_iterations=10)
        assert 0.0 <= lower <= upper <= 1.0


class TestThresholdTuning:
    def test_best_threshold_in_range(self, y_true, y_proba):
        best_thr, best_f1, curve = threshold_tuning(y_true, y_proba, n_thresholds=20)
        assert 0.01 <= best_thr <= 0.99
        assert 0.0 <= best_f1 <= 1.0

    def test_curve_length(self, y_true, y_proba):
        _, _, curve = threshold_tuning(y_true, y_proba, n_thresholds=50)
        assert len(curve) == 50

    def test_curve_keys(self, y_true, y_proba):
        _, _, curve = threshold_tuning(y_true, y_proba, n_thresholds=10)
        for entry in curve:
            assert set(entry.keys()) == {"threshold", "f1", "precision", "recall"}

    def test_f1_improves_over_default(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
        y_proba = np.array([0.05, 0.1, 0.15, 0.6, 0.7, 0.8, 0.2, 0.25, 0.3, 0.9])
        best_thr, best_f1, _ = threshold_tuning(y_true, y_proba, n_thresholds=50)

        default_f1 = 0.0
        from sklearn.metrics import f1_score

        for thr in [0.3, 0.5, 0.7]:
            yp = (y_proba >= thr).astype(int)
            default_f1 = max(default_f1, f1_score(y_true, yp))

        assert best_f1 >= default_f1


class TestPerSegmentAnalysis:
    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(42)
        n = 200
        y_true = (rng.uniform(0, 1, n) > 0.95).astype(int)
        y_proba = y_true + rng.uniform(-0.1, 0.1, n)
        y_proba = np.clip(y_proba, 0.0, 1.0)
        segments = pd.DataFrame(
            {
                "ProductCD": np.random.choice(["W", "H", "C"], n),
                "card4": np.random.choice(["visa", "mastercard", "amex"], n),
            }
        )
        return y_true, y_proba, segments

    def test_returns_dataframe(self, data):
        y_true, y_proba, segments = data
        result = per_segment_analysis(y_true, y_proba, segments)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, data):
        y_true, y_proba, segments = data
        result = per_segment_analysis(y_true, y_proba, segments)
        expected = {
            "segment_col",
            "segment_value",
            "count",
            "fraud_rate",
            "roc_auc",
            "average_precision",
        }
        assert expected.issubset(set(result.columns))

    def test_filters_small_segments(self, data):
        y_true, y_proba, segments = data
        result = per_segment_analysis(y_true, y_proba, segments)
        assert (result["count"] >= 50).all()

    def test_non_empty_with_enough_data(self, data):
        y_true, y_proba, segments = data
        result = per_segment_analysis(y_true, y_proba, segments)
        assert len(result) > 0
