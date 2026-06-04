import pandas as pd

from src.features.base import BaseFeatureTransformer

# M4 is categorical: M0/M1/M2/NaN → "missing"
_M4_CATEGORIES = frozenset({"M0", "M1", "M2"})


class MFlagEncoder(BaseFeatureTransformer):
    """Encode M flag columns according to their type.

    * M1–M3, M5–M9 — binary match flags: T → 1, F → 0, NaN → -1.
    * M4 — categorical (``M0``/``M1``/``M2``), NaN → ``"missing"``.

    Columns are auto-detected by the ``M`` prefix; non-existent columns are
    silently skipped.

    Args:
        enabled: When False the transform returns X unchanged.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()

        m_cols = [c for c in X.columns if c.startswith("M")]

        for col in m_cols:
            if col == "M4":
                X[col] = self._encode_m4(X[col])
            else:
                X[col] = X[col].map({"T": 1, "F": 0}).fillna(-1).astype("int8")

        return X

    @staticmethod
    def _encode_m4(series: pd.Series) -> pd.Series:
        if isinstance(series.dtype, pd.CategoricalDtype):
            missing = set(series.cat.categories) - _M4_CATEGORIES
            if missing:
                series = series.cat.remove_categories(missing)
            return series.cat.add_categories(["missing"]).fillna("missing")
        return series.fillna("missing")
