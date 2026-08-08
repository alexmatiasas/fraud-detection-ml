import numpy as np
import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class AmountFeatureExtractor(BaseFeatureTransformer):
    """Add log- and roundness features for ``TransactionAmt``.

    * ``TransactionAmt_log`` — ``log1p`` of the raw amount (EDA 5.1).
    * ``is_round_amount`` — 1 when the amount is a whole number, 0 otherwise.

    Missing amounts stay NaN so ``MissingImputer`` can apply the ``-999``
    sentinel later.

    Args:
        enabled: When False the transform returns X unchanged.
        use_log: Create ``TransactionAmt_log``.
        use_round: Create ``is_round_amount``.
    """

    def __init__(
        self,
        enabled: bool = True,
        use_log: bool = True,
        use_round: bool = True,
    ):
        self.enabled = enabled
        self.use_log = use_log
        self.use_round = use_round

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        if "TransactionAmt" not in X.columns:
            return X

        X = X.copy()
        amt = X["TransactionAmt"]

        if self.use_log:
            X["TransactionAmt_log"] = np.log1p(amt).astype("float32")

        if self.use_round:
            is_round = amt == amt.round(0)
            X["is_round_amount"] = np.where(is_round, 1, 0).astype("float32")
            X.loc[amt.isna(), "is_round_amount"] = np.nan

        return X
