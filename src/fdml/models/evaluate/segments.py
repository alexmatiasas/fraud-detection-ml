from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def per_segment_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    segments: pd.DataFrame,
    min_samples: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in segments.columns:
        for val in segments[col].dropna().unique():
            mask = segments[col] == val
            if mask.sum() < min_samples:
                continue
            y_seg = y_true[mask]
            p_seg = y_proba[mask]
            rows.append(
                {
                    "segment_col": col,
                    "segment_value": val,
                    "count": int(mask.sum()),
                    "fraud_rate": float(y_seg.mean()),
                    "roc_auc": float(roc_auc_score(y_seg, p_seg)),
                    "average_precision": float(average_precision_score(y_seg, p_seg)),
                }
            )
    return pd.DataFrame(rows).sort_values("average_precision", ascending=True)
