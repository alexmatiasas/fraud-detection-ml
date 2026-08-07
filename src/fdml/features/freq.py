from typing import Optional

import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class FrequencyEncoder(BaseFeatureTransformer):
    """Replace high-cardinality columns with their frequency in the training set.

    Learns the frequency map from ``fit()`` and applies it in ``transform()``.
    Categories unseen during ``fit()`` receive a frequency of 1 (treated as
    rare values).

    Args:
        enabled: When False the transform returns X unchanged.
        columns: List of column names to encode.  Non-existent columns are
            silently skipped.
    """

    def __init__(self, enabled: bool = True, columns: Optional[list[str]] = None):
        self.enabled = enabled
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FrequencyEncoder":
        if not self.enabled:
            self._fitted_ = True
            return self
        self._freq_maps: dict[str, dict] = {}
        for col in self.columns:
            if col not in X.columns:
                continue
            self._freq_maps[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()
        for col, mapping in self._freq_maps.items():
            if col not in X.columns:
                continue
            name = f"{col}_freq"
            X[name] = X[col].map(mapping).fillna(1).astype("int32")
        return X
