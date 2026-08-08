import numpy as np
import pandas as pd

from fdml.features.base import BaseFeatureTransformer

# Microsoft-owned email domains — folded into a single "microsoft" entity
# (EDA 5.4: outlook/hotmail/live/msn behave as one provider).
_MICROSOFT_DOMAINS = frozenset({"outlook.com", "hotmail.com", "live.com", "msn.com"})
_EMAIL_COLUMNS = ("P_emaildomain", "R_emaildomain")


def _normalize_domain(value: object) -> object:
    if value is None:
        return value
    domain = str(value).strip().lower()
    if domain in ("", "nan"):
        return value
    if domain in _MICROSOFT_DOMAINS:
        return "microsoft"
    return domain


class EmailFeatureExtractor(BaseFeatureTransformer):
    """Normalize email domains and flag payer/recipient domain matches.

    * Applies a consistent mapping to ``P_emaildomain`` and
      ``R_emaildomain`` (lowercase, Microsoft domains folded into
      ``microsoft``) so downstream frequency encoding treats e.g.
      ``outlook.com`` and ``hotmail.com`` as one entity.
    * Adds ``p_r_domain_match`` — 1 when payer and recipient domains match,
      0 otherwise, NaN when either side is missing.

    Args:
        enabled: When False the transform returns X unchanged.
        use_grouping: Normalize the domain columns in place.
        use_match: Create ``p_r_domain_match``.
    """

    def __init__(
        self,
        enabled: bool = True,
        use_grouping: bool = True,
        use_match: bool = True,
    ):
        self.enabled = enabled
        self.use_grouping = use_grouping
        self.use_match = use_match

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()

        if self.use_grouping:
            for col in _EMAIL_COLUMNS:
                if col in X.columns:
                    X[col] = X[col].map(_normalize_domain)

        if self.use_match and all(c in X.columns for c in _EMAIL_COLUMNS):
            p = X["P_emaildomain"]
            r = X["R_emaildomain"]
            match = p == r
            X["p_r_domain_match"] = np.where(match, 1, 0).astype("float32")
            X.loc[p.isna() | r.isna(), "p_r_domain_match"] = np.nan

        return X
