from typing import Optional

import numpy as np
import pandas as pd

from fdml.features.base import BaseFeatureTransformer


def _pairwise_corr(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Pairwise Pearson correlation using only rows where both columns are
    present (the semantics of ``pandas.DataFrame.corr``) computed with masked
    matrix multiplications.

    For each pair (i, j) the statistics below are accumulated over rows where
    both ``x_i`` and ``x_j`` are non-NaN.  With ``A = where(~nan, X, 0)`` and
    ``M = ~nan`` the needed sums come from four GEMMs::

        cnt = M.T @ M          #  n        both present
        sxy = A.T @ A          #  sum x_i x_j
        sx  = A.T @ M          #  sum x_i  (rows where x_j present)
        sxx = (A*A).T @ M      #  sum x_i^2 (rows where x_j present)

    This is ``O(k^2 n)`` in a single BLAS pass instead of pandas' per-pair
    loop, which on ~340 columns x 470k rows is ~70s slower.
    """

    X = np.asarray(X, dtype=np.float64)
    mask = ~np.isnan(X)
    A = np.where(mask, X, 0.0)

    cnt = mask.astype(np.float64).T @ mask.astype(np.float64)
    sxy = A.T @ A
    sx = A.T @ mask.astype(np.float64)
    sxx = (A * A).T @ mask.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sx / cnt
        var = sxx / cnt - mean**2
        cov = sxy / cnt - mean * mean.T
        denom = np.sqrt(var * var.T)
        corr = np.where(denom > 0, cov / denom, np.nan)
    np.fill_diagonal(corr, 1.0)
    return corr


class VFeatureFilter(BaseFeatureTransformer):
    """Filter prefix-matched columns by variance then correlation.

    Learns which columns to keep from ``fit()`` (on the training set) and
    applies the same mask in ``transform()``.

    Args:
        enabled: When False the transform returns X unchanged.
        variance_threshold: Minimum variance to retain a feature.
        correlation_threshold: Maximum absolute Pearson correlation to keep
            both features of a pair.  When two features are correlated above
            this threshold the one with lower variance is dropped.
        prefixes: Column prefixes to filter (e.g. ``("V",)`` or ``("C",)``).
    """

    def __init__(
        self,
        enabled: bool = True,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        prefixes: tuple[str, ...] = ("V",),
    ):
        self.enabled = enabled
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.prefixes = list(prefixes)

    def _matching(self, X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if c.startswith(tuple(self.prefixes))]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "VFeatureFilter":
        if not self.enabled:
            self._fitted_ = True
            return self
        v_cols = self._matching(X)
        if not v_cols:
            self._keep_cols: list[str] = []
            return self

        keep = list(v_cols)
        v_data = X[v_cols]

        variances = {c: float(v_data[c].var()) for c in keep}
        keep = [c for c in keep if variances[c] > self.variance_threshold]

        if len(keep) > 1:
            corr = _pairwise_corr(v_data[keep])
            upper = np.abs(corr)
            upper[np.tril_indices_from(upper)] = np.nan
            to_drop = [
                c
                for c, m in zip(
                    keep, np.any(upper > self.correlation_threshold, axis=0)
                )
                if m
            ]
            keep = [c for c in keep if c not in to_drop]

        self._keep_cols = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        if not self._keep_cols:
            return X
        v_cols = self._matching(X)
        drop_cols = [c for c in v_cols if c not in self._keep_cols]
        return X.drop(columns=drop_cols, errors="ignore")
