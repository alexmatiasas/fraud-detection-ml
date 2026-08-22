from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")


class Plotter(ABC):
    @abstractmethod
    def filename(self) -> str:
        """Base filename without extension, e.g. ``roc_curve``."""

    @abstractmethod
    def _plot(self, ax: Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None: ...

    def figure_size(self) -> tuple[int, int]:
        return (6, 5)

    def plot(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_dir: str = "models/",
        fmt: str = "webp",
        dpi: int = 150,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figure_size())
        self._plot(ax, y_true, y_proba)
        out = Path(output_dir) / f"{self.filename()}.{fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return out


class ROCCurvePlotter(Plotter):
    def filename(self) -> str:
        return "roc_curve"

    def _plot(self, ax: Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
        ax.plot(fpr, tpr, lw=2, color="#1f77b4", label="ROC")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve — validation")
        ax.text(
            0.03,
            0.95,
            f"AUC = {auc:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#1f77b4", alpha=0.15),
        )
        ax.text(
            0.03,
            0.86,
            f"pos={n_pos:,}  neg={n_neg:,}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="0.35",
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


class PRCurvePlotter(Plotter):
    def filename(self) -> str:
        return "pr_curve"

    def _plot(self, ax: Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        baseline = float(np.mean(y_true))
        ax.axhline(baseline, color="k", linestyle="--", alpha=0.5, label="No-skill")
        ax.plot(recall, precision, lw=2, color="#ff7f0e", label="PR curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve — validation")
        ax.text(
            0.03,
            0.95,
            f"AP = {ap:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#ff7f0e", alpha=0.15),
        )
        ax.text(
            0.03,
            0.86,
            f"fraud rate = {baseline:.2%}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="0.35",
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)


class CalibrationPlotter(Plotter):
    def __init__(self, n_bins: int = 10):
        self._n_bins = n_bins

    def filename(self) -> str:
        return "calibration_curve"

    def _plot(self, ax: Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=self._n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_true, y_proba)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", lw=2, color="#2ca02c", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve — validation")
        ax.text(
            0.05,
            0.95,
            f"Brier = {brier:.4f}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#2ca02c", alpha=0.15),
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


class LearningCurvePlotter(Plotter):
    def filename(self) -> str:
        return "learning_curves"

    def figure_size(self) -> tuple[int, int]:
        return (6, 5)

    def _plot(self, ax: Axes, y_true: np.ndarray, y_proba: np.ndarray) -> None:
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
        fmt: str = "webp",
        dpi: int = 150,
    ) -> list[Path]:
        paths: list[Path] = []
        y_pred = (y_proba >= 0.5).astype(int)
        errors = y_pred != y_true

        data = X_val.copy()
        if "TransactionDT" in data.columns:
            data["hour"] = (data["TransactionDT"] % 86400) / 3600
            if "hour" not in self._features:
                self._features = self._features + ["hour"]

        n_err = int(errors.sum())
        n_ok = len(y_true) - n_err
        for col in self._features:
            if col not in data.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            correct_mask = ~errors
            error_mask = errors
            if correct_mask.sum() > 0:
                ax.hist(
                    data.loc[correct_mask, col],
                    bins=30,
                    alpha=0.5,
                    color="#1f77b4",
                    label="Correct",
                )
            if error_mask.sum() > 0:
                ax.hist(
                    data.loc[error_mask, col],
                    bins=30,
                    alpha=0.5,
                    color="#d62728",
                    label="Error",
                )
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Error distribution by {col}")
            ax.text(
                0.98,
                0.95,
                f"correct={n_ok:,}  errors={n_err:,}",
                transform=ax.transAxes,
                fontsize=9,
                ha="right",
                va="top",
                color="0.35",
            )
            ax.grid(alpha=0.3)
            ax.legend()
            out = Path(output_dir) / f"error_analysis_{col}.{fmt}"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            paths.append(out)
        return paths


def roc_curve_table(y_true: np.ndarray, y_proba: np.ndarray) -> list[dict[str, float]]:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    return [
        {"fpr": float(f), "tpr": float(t), "threshold": float(h)}
        for f, t, h in zip(fpr, tpr, thr)
    ]


def pr_curve_table(y_true: np.ndarray, y_proba: np.ndarray) -> list[dict[str, float]]:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return [
        {"precision": float(p), "recall": float(r)} for p, r in zip(precision, recall)
    ]


def calibration_curve_table(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    return [
        {"prob_pred": float(p), "prob_true": float(t)}
        for p, t in zip(prob_pred, prob_true)
    ]
