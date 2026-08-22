from typing import Optional

import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class CategoryEncoder(BaseFeatureTransformer):
    """Ordinal-encode object/category columns into ``int32`` codes.

    Categories are learned from the training set (sorted, so the mapping is
    deterministic).  Values seen at transform time that were not in training
    map to ``-1``, mirroring the LightGBM "missing" convention.

    This step lives *inside* the feature pipeline (last, after the selector)
    so that the pipeline is self-contained: ``pipeline.transform`` on raw rows
    already returns the exact frame the model was trained on.  Without it,
    serving a raw request through ``Pipeline([features, model])`` would feed
    ``object``/``category`` columns to a model fitted on ordinal codes.

    Args:
        enabled: When False the transform returns X unchanged.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.categories_: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoryEncoder":
        self._fitted_ = True
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        self.categories_ = {
            col: sorted(X[col].astype(str).unique()) for col in cat_cols
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()
        for col, categories in self.categories_.items():
            if col not in X.columns:
                continue
            mapping = {cat: i for i, cat in enumerate(categories)}
            X[col] = X.loc[:, col].astype(str).map(mapping).fillna(-1).astype("int32")
        return X
