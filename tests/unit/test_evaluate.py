from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fdml.models.evaluate import (
    bootstrap_ci,
    brier_score,
    compute_metrics,
    expected_cost,
    f_beta_score,
    per_segment_analysis,
    recall_at_top_k,
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


class TestBrierScore:
    def test_perfect_prediction_is_zero(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_proba) == pytest.approx(0.0)

    def test_always_fraud_rate_baseline(self):
        y_true = np.array([0, 0, 0, 1])
        p = np.full(4, 0.25)
        assert brier_score(y_true, p) == pytest.approx(0.1875)

    def test_worse_than_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        good = brier_score(y_true, np.array([0.1, 0.2, 0.8, 0.9]))
        bad = brier_score(y_true, np.array([0.9, 0.8, 0.2, 0.1]))
        assert bad > good


class TestFBeta:
    def test_beta_one_equals_f1(self):
        from sklearn.metrics import f1_score

        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.4, 0.6, 0.4, 0.6])
        expected = f1_score(y_true, (y_proba >= 0.5).astype(int))
        assert f_beta_score(y_true, y_proba, beta=1.0) == pytest.approx(expected)

    def test_higher_beta_rewards_recall(self):
        y_true = np.array([0, 1, 1, 1, 1])
        y_proba = np.array([0.9, 0.51, 0.51, 0.51, 0.51])
        f1 = f_beta_score(y_true, y_proba, beta=1.0)
        f2 = f_beta_score(y_true, y_proba, beta=2.0)
        assert f2 > f1

    def test_in_range(self, y_true, y_proba):
        assert 0.0 <= f_beta_score(y_true, y_proba, beta=2.0) <= 1.0


class TestExpectedCost:
    def test_returns_best_threshold_cost_and_curve(self, y_true, y_proba):
        best_thr, best_cost, curve = expected_cost(y_true, y_proba, n_thresholds=20)
        assert 0.01 <= best_thr <= 0.99
        assert best_cost >= 0.0
        assert len(curve) == 20
        for entry in curve:
            assert set(entry.keys()) == {
                "threshold",
                "expected_cost",
                "precision",
                "recall",
            }

    def test_best_cost_is_minimal(self, y_true, y_proba):
        best_thr, best_cost, curve = expected_cost(y_true, y_proba, n_thresholds=100)
        assert best_cost <= min(c["expected_cost"] for c in curve)

    def test_expensive_false_negatives_lower_threshold(self, y_true, y_proba):
        thr_fp, _, _ = expected_cost(
            y_true, y_proba, fp_cost=100.0, fn_cost=1.0, n_thresholds=50
        )
        thr_fn, _, _ = expected_cost(
            y_true, y_proba, fp_cost=1.0, fn_cost=100.0, n_thresholds=50
        )
        assert thr_fn <= thr_fp

    def test_perfect_separation_zero_cost(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        _, best_cost, _ = expected_cost(y_true, y_proba, n_thresholds=50)
        assert best_cost == pytest.approx(0.0)


class TestRecallAtTopK:
    def test_perfect_ranking_captures_all_frauds(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        assert recall_at_top_k(y_true, y_proba, k_fraction=0.5) == 1.0

    def test_reversed_ranking_captures_nothing(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        assert recall_at_top_k(y_true, y_proba, k_fraction=0.5) == 0.0

    def test_top_quarter_captures_half_the_frauds(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.95, 0.7, 0.65])
        assert recall_at_top_k(y_true, y_proba, k_fraction=0.25) == 0.5

    def test_no_positives_is_zero(self):
        y_true = np.zeros(10, dtype=int)
        y_proba = np.ones(10)
        assert recall_at_top_k(y_true, y_proba, k_fraction=0.1) == 0.0


class TestEvaluationReport:
    def test_accepts_extended_fields(self):
        from fdml.models.evaluate.reporter import CostPoint, EvaluationReport

        report = EvaluationReport(
            model_name="LGBM",
            split_strategy="temporal",
            n_features=10,
            n_train=100,
            n_val=50,
            fraud_rate=0.04,
            roc_auc=0.9,
            average_precision=0.5,
            f1=0.4,
            precision=0.5,
            recall=0.3,
            best_threshold=0.1,
            best_f1=0.4,
            brier=0.1,
            f_beta=0.35,
            cost_best_threshold=0.08,
            expected_cost=0.03,
            recall_at_k={"0.0100": 0.6},
            cost_curve=[
                CostPoint(threshold=0.5, expected_cost=0.5, precision=0.4, recall=0.3)
            ],
        )
        dumped = report.model_dump(mode="json")
        assert dumped["brier"] == 0.1
        assert dumped["expected_cost"] == 0.03
        assert dumped["recall_at_k"] == {"0.0100": 0.6}
        assert dumped["cost_curve"][0]["expected_cost"] == 0.5


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
