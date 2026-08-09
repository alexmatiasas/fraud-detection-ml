from typing import Optional

import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class IdCodeEncoder(BaseFeatureTransformer):
    """Re-cast integer ``id_#`` code columns to ``category`` dtype.

    Most numeric ``id_#`` identity columns are integer codes (browser
    version, screen index, match counters) rather than continuous
    measurements.  LightGBM treats plain integers as ordered numerics and
    splits on them natively, so the default pipeline leaves them as-is; this
    transformer exists to empirically test whether treating them as unordered
    categories changes performance (hypothesis: ~0).

    ``_encode_categoricals`` label-encodes ``category`` columns into ordinal
    codes, which is the downstream encoding path used by the rest of the
    pipeline.

    Args:
        enabled: When False the transform returns X unchanged.
        columns: ``id_#`` columns to recast to ``category`` dtype.
    """

    def __init__(self, enabled: bool = True, columns: Optional[list[str]] = None):
        self.enabled = enabled
        self.columns = columns or []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()
        for col in self.columns:
            if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        return X
