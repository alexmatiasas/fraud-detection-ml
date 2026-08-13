from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def learning_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pipeline: Pipeline,
    model_fn: Callable[[], Any],
    train_sizes: list[float] | None = None,
    output_dir: str = "models/",
) -> Path:
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    train_scores: list[float] = []
    val_scores: list[float] = []

    for size in train_sizes:
        n = max(2, int(len(X_train) * size))
        X_sub = X_train.iloc[:n]
        y_sub = y_train.iloc[:n]

        pipe = Pipeline(
            [
                ("features", pipeline.named_steps["features"]),
                ("model", model_fn()),
            ]
        )
        pipe.fit(X_sub, y_sub)
        p_val = pipe.predict_proba(X_val)[:, 1]
        val_scores.append(float(average_precision_score(y_val, p_val)))

        if size < 1.0:
            p_train = pipe.predict_proba(X_sub)[:, 1]
            try:
                train_scores.append(float(average_precision_score(y_sub, p_train)))
            except ValueError:
                train_scores.append(0.0)
        else:
            train_scores.append(val_scores[-1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([s * 100 for s in train_sizes], train_scores, "o-", label="Train (AP)")
    ax.plot([s * 100 for s in train_sizes], val_scores, "o-", label="Validation (AP)")
    ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Average Precision")
    ax.set_title("Learning curves")
    ax.legend()
    ax.set_ylim(0, 1)

    out = Path(output_dir) / "learning_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out


def feature_importance_stability(
    model: Any,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pipeline: Pipeline,
    n_iterations: int = 10,
    seed: int = 42,
    output_dir: str = "models/",
) -> Path:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000, n_iterations).tolist()

    importance_df = pd.DataFrame(index=range(n_iterations), columns=feature_names)

    for i, s in enumerate(seeds):
        cfg_name = model.__class__.__name__
        params = {
            k: v
            for k, v in model.get_params().items()
            if k not in ("random_state", "n_jobs", "verbose", "verbosity")
        }
        if "LGBMClassifier" in cfg_name:
            new_model = model.__class__(**params, random_state=s, n_jobs=-1, verbose=-1)
        elif "XGBClassifier" in cfg_name:
            new_model = model.__class__(
                **params, random_state=s, n_jobs=-1, verbosity=0
            )
        else:
            new_model = model.__class__(**params, random_state=s, n_jobs=-1, verbose=0)

        pipe = Pipeline(
            [("features", pipeline.named_steps["features"]), ("model", new_model)]
        )
        pipe.fit(X_train, y_train)

        imp = pipe.named_steps["model"].feature_importances_
        if imp is not None and len(imp) == len(feature_names):
            importance_df.iloc[i] = imp

    mean_imp = importance_df.mean(axis=0).sort_values(ascending=False)
    std_imp = importance_df.std(axis=0)[mean_imp.index]

    top_n = min(20, len(mean_imp))
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(top_n)
    ax.barh(list(y_pos), mean_imp.iloc[:top_n][::-1], xerr=std_imp.iloc[:top_n][::-1])
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(mean_imp.index[:top_n][::-1])
    ax.set_xlabel("Mean feature importance")
    ax.set_title(f"Feature importance stability ({n_iterations} fits)")

    rank_df = importance_df.rank(axis=1, ascending=False)
    rank_consistency = float(rank_df.std(axis=1).mean())
    logger.info(
        "  Feature importance stability: mean rank std=%.2f (lower = more stable)",
        rank_consistency,
    )

    out = Path(output_dir) / "feature_importance_stability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return out
