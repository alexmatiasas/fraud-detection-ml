import pandas as pd

from fdml.features.base import BaseFeatureTransformer

_SECONDS_IN_HOUR = 3600
_SECONDS_IN_DAY = 86400


class TimeFeatureExtractor(BaseFeatureTransformer):
    """Extract temporal features from ``TransactionDT`` (delta seconds).

    The absolute origin of ``TransactionDT`` is arbitrary but the relative
    patterns (hour-of-day, day-of-week, day-of-month) are valid regardless.

    Args:
        enabled: When False the transform returns X unchanged.
        use_hour: Create ``hour_of_day`` column.
        use_dow: Create ``day_of_week`` column.
        use_dom: Create ``day_of_month`` column.
    """

    def __init__(
        self,
        enabled: bool = True,
        use_hour: bool = True,
        use_dow: bool = True,
        use_dom: bool = True,
    ):
        self.enabled = enabled
        self.use_hour = use_hour
        self.use_dow = use_dow
        self.use_dom = use_dom

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        if "TransactionDT" not in X.columns:
            return X

        X = X.copy()
        dt = X["TransactionDT"]

        if self.use_hour:
            X["hour_of_day"] = (dt // _SECONDS_IN_HOUR) % 24
        if self.use_dow:
            X["day_of_week"] = (dt // _SECONDS_IN_DAY) % 7
        if self.use_dom:
            X["day_of_month"] = (dt // _SECONDS_IN_DAY) % 30

        return X
