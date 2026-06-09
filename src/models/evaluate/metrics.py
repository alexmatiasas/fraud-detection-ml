from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "average_precision",
    n_iterations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores: list[float] = []

    metric_fn = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
    }[metric]

    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        p_boot = y_proba[idx]
        try:
            scores.append(float(metric_fn(y_boot, p_boot)))
        except ValueError:
            continue

    arr = np.array(scores)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def threshold_tuning(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 100,
) -> tuple[float, float, list[dict[str, float]]]:
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1 = 0.0
    best_thr = 0.5
    curve: list[dict[str, float]] = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        curve.append(
            {
                "threshold": float(thr),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
            }
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1, curve
