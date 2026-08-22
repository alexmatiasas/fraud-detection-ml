from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")


def adversarial_validation(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    output_dir: str = "models/",
) -> tuple[float, Path]:
    n_train = len(X_train)
    n_val = len(X_val)

    X_adv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * n_train + [1] * n_val)

    cat_cols = X_adv.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_adv[col] = X_adv[col].astype(str).astype("category").cat.codes

    X_adv = X_adv.fillna(-999)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=0
    )
    clf.fit(X_adv, y_adv)
    p_adv = np.asarray(clf.predict_proba(X_adv))[:, 1]
    auc_adv = roc_auc_score(y_adv, p_adv)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_adv[y_adv == 0], bins=30, alpha=0.5, label="Train")
    ax.hist(p_adv[y_adv == 1], bins=30, alpha=0.5, label="Validation")
    ax.set_xlabel("Predicted probability of being validation")
    ax.set_title(f"Adversarial validation AUC={auc_adv:.3f}")
    ax.legend()

    out = Path(output_dir) / "adversarial_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return float(auc_adv), out
