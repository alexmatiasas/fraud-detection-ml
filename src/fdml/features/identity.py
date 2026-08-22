import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class IdentityFlagExtractor(BaseFeatureTransformer):
    """Flag rows that have an identity-table record.

    After the left-merge of the identity table, rows without a match have
    NaN in every identity column.  ``has_identity`` is 1 when the record
    exists (``DeviceType`` or ``DeviceInfo`` present), 0 otherwise.  Must
    run before any step that drops the raw identity columns.

    Args:
        enabled: When False the transform returns X unchanged.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        markers = [c for c in ("DeviceType", "DeviceInfo") if c in X.columns]
        if not markers:
            return X
        X = X.copy()
        present = pd.Series(False, index=X.index, dtype=bool)
        for col in markers:
            present = present | X[col].notna()
        X["has_identity"] = present.astype("int8")
        return X
