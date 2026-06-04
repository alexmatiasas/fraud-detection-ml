from typing import Optional

import numpy as np
import pandas as pd

from src.features.base import BaseFeatureTransformer


class VFeatureFilter(BaseFeatureTransformer):
    """Filter V features by variance then correlation.

    Learns which columns to keep from ``fit()`` (on the training set) and
    applies the same mask in ``transform()``.

    Args:
        enabled: When False the transform returns X unchanged.
        variance_threshold: Minimum variance to retain a feature.
        correlation_threshold: Maximum absolute Pearson correlation to keep
            both features of a pair.  When two features are correlated above
            this threshold the one with lower variance is dropped.
    """

    def __init__(
        self,
        enabled: bool = True,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
    ):
        self.enabled = enabled
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VFeatureFilter":
        if not self.enabled:
            return self
        v_cols = [c for c in X.columns if c.startswith("V")]
        if not v_cols:
            self._keep_cols: list[str] = []
            return self

        keep = list(v_cols)
        v_data = X[v_cols]

        var = v_data.var()
        keep = [c for c in keep if var.get(c, 0) > self.variance_threshold]

        if len(keep) > 1:
            corr = v_data[keep].corr()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            to_drop = [
                c
                for c in upper.columns
                if any(abs(upper[c]) > self.correlation_threshold)
            ]
            keep = [c for c in keep if c not in to_drop]

        self._keep_cols = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        if not self._keep_cols:
            return X
        v_cols = [c for c in X.columns if c.startswith("V")]
        drop_cols = [c for c in v_cols if c not in self._keep_cols]
        return X.drop(columns=drop_cols, errors="ignore")
