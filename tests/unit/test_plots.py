from __future__ import annotations

import numpy as np
import pytest

from fdml.models.evaluate.plots import (
    CalibrationPlotter,
    ErrorAnalysisPlotter,
    LearningCurvePlotter,
    PRCurvePlotter,
    ROCCurvePlotter,
    calibration_curve_table,
    pr_curve_table,
    roc_curve_table,
)


@pytest.fixture()
def synthetic_labels():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=200, p=[0.965, 0.035])
    y_proba = rng.uniform(0, 1, size=200)
    return y_true, y_proba


class TestCurveTables:
    def test_roc_curve_table(self, synthetic_labels):
        y_true, y_proba = synthetic_labels
        table = roc_curve_table(y_true, y_proba)
        assert isinstance(table, list)
        assert len(table) > 0
        assert {"fpr", "tpr", "threshold"} <= set(table[0].keys())
        assert table[0]["fpr"] >= 0
        assert table[-1]["tpr"] <= 1.0

    def test_pr_curve_table(self, synthetic_labels):
        y_true, y_proba = synthetic_labels
        table = pr_curve_table(y_true, y_proba)
        assert isinstance(table, list)
        assert len(table) > 0
        assert {"precision", "recall"} <= set(table[0].keys())

    def test_calibration_curve_table(self, synthetic_labels):
        y_true, y_proba = synthetic_labels
        table = calibration_curve_table(y_true, y_proba, n_bins=5)
        assert isinstance(table, list)
        assert len(table) > 0
        assert {"prob_pred", "prob_true"} <= set(table[0].keys())


class TestPlotterClasses:
    def test_roc_plotter_creates_file(self, synthetic_labels, tmp_path):
        y_true, y_proba = synthetic_labels
        plotter = ROCCurvePlotter()
        out = plotter.plot(y_true, y_proba, output_dir=str(tmp_path))
        assert out.exists()
        assert out.stat().st_size > 0
        assert out.name == "roc_curve.webp"

    def test_pr_plotter_creates_file(self, synthetic_labels, tmp_path):
        y_true, y_proba = synthetic_labels
        plotter = PRCurvePlotter()
        out = plotter.plot(y_true, y_proba, output_dir=str(tmp_path))
        assert out.exists()
        assert out.name == "pr_curve.webp"

    def test_calibration_plotter_creates_file(self, synthetic_labels, tmp_path):
        y_true, y_proba = synthetic_labels
        plotter = CalibrationPlotter(n_bins=5)
        out = plotter.plot(y_true, y_proba, output_dir=str(tmp_path))
        assert out.exists()
        assert out.name == "calibration_curve.webp"

    def test_learning_curve_plotter_noop(self, synthetic_labels, tmp_path):
        y_true, y_proba = synthetic_labels
        plotter = LearningCurvePlotter()
        out = plotter.plot(y_true, y_proba, output_dir=str(tmp_path))
        assert out.exists()


class TestErrorAnalysisPlotter:
    def test_plots_error_distribution(self, synthetic_labels, tmp_path):
        y_true, y_proba = synthetic_labels
        import pandas as pd

        X_val = pd.DataFrame(
            {"TransactionAmt": np.random.uniform(10, 500, size=len(y_true))}
        )
        plotter = ErrorAnalysisPlotter(features_to_plot=["TransactionAmt"])
        paths = plotter.plot(y_true, y_proba, X_val, output_dir=str(tmp_path))
        assert len(paths) == 1
        assert paths[0].exists()
