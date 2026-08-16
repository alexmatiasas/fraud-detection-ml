"""Feature overrides and per-prediction SHAP explanations.

Overrides mutate a raw sample row (pre feature-engineering) so the caller can
ask "what if TransactionAmt were 500 instead of 68.5?". Values are cast to the
column dtype where possible; unknown columns or uncastable values raise
:class:`OverrideError`.

Explanations use ``shap.TreeExplainer`` on the fitted tree model inside the
pipeline, computed on the feature-engineered space. ``shap`` is imported
lazily so the API still boots when it is not installed (explanation falls back
to ``None``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Engineered feature space is 341 wide; no point in explaining more.
DEFAULT_TOP_K = 20
MAX_TOP_K = 341


class OverrideError(ValueError):
    """Raised when an override references an unknown column or bad value."""


def apply_overrides(
    row: pd.DataFrame, overrides: dict[str, Any] | None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply feature overrides to a raw single-row DataFrame.

    Args:
        row: Single-row raw DataFrame (from the demo sample).
        overrides: Mapping of raw column name → new value.

    Returns:
        The modified row plus the normalized overrides actually applied.

    Raises:
        OverrideError: Unknown column, or value not castable to the column dtype.
    """
    if not overrides:
        return row, {}

    unknown = [key for key in overrides if key not in row.columns]
    if unknown:
        raise OverrideError(f"Unknown feature(s): {', '.join(sorted(unknown))}")

    result = row.copy()
    applied: dict[str, Any] = {}
    for column, value in overrides.items():
        result.loc[:, column] = _coerce(value, row[column].dtype, column)
        applied[column] = _jsonable(result.iloc[0][column])
    return result, applied


def _coerce(value: Any, dtype: Any, column: str) -> Any:
    """Cast an override value to a column's dtype, raising OverrideError."""
    kind = getattr(dtype, "kind", "O")
    if kind in "iuf":  # signed / unsigned integer, float
        if isinstance(value, bool):
            return int(value)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise OverrideError(
                f"Feature '{column}' expects a number, got {value!r}"
            ) from exc
    if kind == "b":
        return bool(value)
    if kind == "M":  # datetime
        try:
            return pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise OverrideError(
                f"Feature '{column}' expects a datetime, got {value!r}"
            ) from exc
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def compute_explanation(
    pipeline: Any, X_raw: pd.DataFrame, top_k: int = DEFAULT_TOP_K
) -> dict[str, Any] | None:
    """Tree SHAP explanation for the first row of ``X_raw``.

    Returns a serializable dict (``base_value``, ``n_features``,
    ``top_features``) or ``None`` when shap is unavailable or the model is not
    explainable.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    try:
        import shap  # noqa: PLC0415 - optional serving dependency
    except ImportError:
        return None

    try:
        model = pipeline.named_steps["model"]
        features_step = pipeline.named_steps["features"]
        X_fe = features_step.transform(X_raw)

        explainer: Any = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_fe, check_additivity=False)
        shap_1 = _positive_class_values(raw)
        base: Any = explainer.expected_value  # float or (neg_class, pos_class)
        if isinstance(base, Sequence):
            base = base[1] if len(base) > 1 else base[0]

        names = _feature_names(X_fe, model, len(shap_1))
        values = (
            X_fe.iloc[0].tolist()
            if isinstance(X_fe, pd.DataFrame)
            else [None] * len(names)
        )
        ranked = sorted(
            zip(names, values, shap_1), key=lambda item: abs(item[2]), reverse=True
        )[:top_k]

        return {
            "base_value": float(base),
            "n_features": len(names),
            "top_features": [
                {"feature": name, "value": _jsonable(val), "shap": float(shap)}
                for name, val, shap in ranked
            ],
        }
    except Exception as exc:  # noqa: BLE001 - explanation is best-effort
        logger.warning("  SHAP explanation failed: %s", exc)
        return None


def _positive_class_values(raw: Any) -> np.ndarray:
    """Extract the positive-class SHAP row for a binary classifier."""
    values = raw
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    return arr[0] if arr.ndim == 2 else arr


def _feature_names(X_fe: Any, model: Any, n: int) -> list[str]:
    if isinstance(X_fe, pd.DataFrame):
        return [str(column) for column in X_fe.columns]
    for attr in ("feature_names_in_", "feature_name_", "feature_names_"):
        names = getattr(model, attr, None)
        if names is not None and len(names) == n:
            return [str(name) for name in names]
    return [f"feature_{i}" for i in range(n)]
