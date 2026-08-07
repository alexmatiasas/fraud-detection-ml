from typing import Optional

import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class FeatureSelector(BaseFeatureTransformer):
    """Keep only the columns expected by the model.

    Behaves as an allow-list: everything not in ``feature_columns`` (and not
    a V feature when ``vesta_include`` is *True*) is dropped.  The output
    column order follows ``feature_columns`` with V columns appended.

    Args:
        enabled: When False the transform returns X unchanged.
        feature_columns: Ordered list of columns to retain.  Columns not
            present in the DataFrame are silently skipped.
        target: Name of the target column to exclude from the output.
        vesta_include: When *True* all columns starting with ``V`` that
            exist in the DataFrame are included automatically (appended
            after the static feature list).
    """

    def __init__(
        self,
        enabled: bool = True,
        feature_columns: Optional[list[str]] = None,
        target: Optional[str] = None,
        vesta_include: bool = True,
    ):
        self.enabled = enabled
        self.feature_columns = feature_columns or []
        self.target = target
        self.vesta_include = vesta_include

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X

        cols = [c for c in self.feature_columns if c in X.columns]
        if self.vesta_include:
            v_cols = sorted([c for c in X.columns if c.startswith("V")])
            cols.extend(c for c in v_cols if c not in cols)
        if self.target and self.target in cols:
            cols.remove(self.target)

        return X[cols]
