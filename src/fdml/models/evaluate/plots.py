from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")


def _maybe_log_figure(fig: plt.Figure, artifact_path: str) -> None:
    if mlflow.active_run() is not None:
        try:
            mlflow.log_figure(fig, artifact_path)
        except Exception:
            pass


class Plotter(ABC):
    @abstractmethod
    def filename(self) -> str: ...

    @abstractmethod
    def _plot(self, ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None: ...

    def figure_size(self) -> tuple[int, int]:
        return (6, 5)

    def plot(
        self, y_true: np.ndarray, y_proba: np.ndarray, output_dir: str = "models/"
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figure_size())
        self._plot(ax, y_true, y_proba)
        out = Path(output_dir) / self.filename()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=100)
        _maybe_log_figure(fig, f"plots/{self.filename()}")
        plt.close(fig)
        return out


class ROCCurvePlotter(Plotter):
    def filename(self) -> str:
        return "roc_curve.png"

    def _plot(self, ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve")
        ax.legend()


class PRCurvePlotter(Plotter):
    def filename(self) -> str:
        return "pr_curve.png"

    def _plot(self, ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"AP={ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        ax.legend()


class CalibrationPlotter(Plotter):
    def __init__(self, n_bins: int = 10):
        self._n_bins = n_bins

    def filename(self) -> str:
        return "calibration_curve.png"

    def _plot(self, ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=self._n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_true, y_proba)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", label=f"Brier={brier:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve (reliability diagram)")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


class LearningCurvePlotter(Plotter):
    def filename(self) -> str:
        return "learning_curves.png"

    def figure_size(self) -> tuple[int, int]:
        return (6, 5)

    def _plot(self, ax: plt.Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        pass  # Learning curves handle their own plotting in stability module


class ErrorAnalysisPlotter:
    def __init__(self, features_to_plot: list[str] | None = None):
        self._features = features_to_plot or ["TransactionAmt"]

    def plot(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        X_val: pd.DataFrame,
        output_dir: str = "models/",
    ) -> list[Path]:
        paths: list[Path] = []
        y_pred = (y_proba >= 0.5).astype(int)
        errors = y_pred != y_true

        data = X_val.copy()
        if "TransactionDT" in data.columns:
            data["hour"] = (data["TransactionDT"] % 86400) / 3600
            if "hour" not in self._features:
                self._features = self._features + ["hour"]

        for col in self._features:
            if col not in data.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            correct_mask = ~errors
            error_mask = errors
            if correct_mask.sum() > 0:
                ax.hist(
                    data.loc[correct_mask.values, col],
                    bins=30,
                    alpha=0.5,
                    label="Correct",
                )
            if error_mask.sum() > 0:
                ax.hist(
                    data.loc[error_mask.values, col], bins=30, alpha=0.5, label="Error"
                )
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Error distribution by {col}")
            ax.legend()
            out = Path(output_dir) / f"error_analysis_{col}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight", dpi=100)
            _maybe_log_figure(fig, f"plots/error_analysis_{col}.png")
            plt.close(fig)
            paths.append(out)
        return paths
