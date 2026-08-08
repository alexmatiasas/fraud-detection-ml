import numpy as np
import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class DtypeOptimizer(BaseFeatureTransformer):
    """Downcast numeric columns to the smallest safe dtype.

    ``float64`` → ``float32`` and ``int64`` → ``int32`` (only when the values
    fit), shrinking memory before training.  Categorical and object columns
    are left untouched.

    Args:
        enabled: When False the transform returns X unchanged.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()

        for col in X.columns:
            if X[col].dtype == np.dtype("float64"):
                X[col] = X[col].astype("float32")
            elif X[col].dtype == np.dtype("int64"):
                lo = int(X[col].min())
                hi = int(X[col].max())
                if np.iinfo("int32").min < lo and hi < np.iinfo("int32").max:
                    X[col] = X[col].astype("int32")

        return X
