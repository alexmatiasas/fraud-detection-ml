from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore[reportArgumentType]
        ),
        "recall": float(recall_score(y_true, y_pred)),
    }


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and outcomes.

    Lower is better; 0.25 = always predicting the fraud rate.
    """
    return float(brier_score_loss(y_true, y_proba))


def f_beta_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
    threshold: float = 0.5,
) -> float:
    """F-beta at a fixed threshold (beta=2 weights recall 2x precision)."""
    y_pred = (y_proba >= threshold).astype(int)
    return float(
        fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0)  # type: ignore[reportArgumentType]
    )


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
    n_thresholds: int = 100,
) -> tuple[float, float, list[dict[str, float]]]:
    """Mean expected cost per transaction over a threshold grid.

    Cost = FP·cost + FN·cost. The business-optimal threshold minimizes it.
    Returns (best_threshold, best_cost, curve).
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thr, best_cost = thresholds[0], np.inf
    curve: list[dict[str, float]] = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = (fp * fp_cost + fn * fn_cost) / len(y_true)
        curve.append(
            {
                "threshold": float(thr),
                "expected_cost": float(cost),
                "precision": float(
                    precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore[reportArgumentType]
                ),
                "recall": float(recall_score(y_true, y_pred)),
            }
        )
        if cost < best_cost:
            best_thr, best_cost = float(thr), float(cost)

    return best_thr, float(best_cost), curve


def recall_at_top_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_fraction: float = 0.01,
) -> float:
    """Fraction of frauds captured in the top k% of highest scores.

    Mirrors the operational flow: score all transactions, manually review the
    top-k% by risk, measure how much fraud was caught.
    """
    n = len(y_true)
    k = max(1, int(round(n * k_fraction)))
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_proba, kind="stable")[:k]
    return float(y_true[order].sum() / n_pos)


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

    for _ in tqdm(range(n_iterations), desc="Bootstrapping", leave=False):
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
        prec = precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore[reportArgumentType]
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
