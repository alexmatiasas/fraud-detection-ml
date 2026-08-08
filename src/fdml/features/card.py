from collections.abc import Mapping, Sequence
from typing import Optional

import pandas as pd

from fdml.features.base import BaseFeatureTransformer


class CardAggregator(BaseFeatureTransformer):
    """Aggregate ``TransactionAmt`` statistics by card groups.

    Learns the groupby aggregations from ``fit()`` and merges them back in
    ``transform()``.  Groups unseen during ``fit()`` receive NaN which is
    later handled by ``MissingImputer``.

    Args:
        enabled: When False the transform returns X unchanged.
        group_by: List of columns to group by.
        aggregations: Mapping of column name to list of aggregation
            functions, e.g. ``{"TransactionAmt": ["mean", "std", "max", "count"]}``.
    """

    def __init__(
        self,
        enabled: bool = True,
        group_by: Optional[list[str]] = None,
        aggregations: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        self.enabled = enabled
        self.group_by = group_by or ["card1", "card2", "card3", "card5"]
        self.aggregations = aggregations or {
            "TransactionAmt": ["mean", "std", "max", "count"]
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CardAggregator":
        if not self.enabled:
            self._fitted_ = True
            return self

        valid_groups = [c for c in self.group_by if c in X.columns]
        if len(valid_groups) < 1:
            self._agg_data: dict[str, pd.DataFrame] = {}
            return self

        self._agg_data = {}
        groups = X.groupby(valid_groups, observed=False)
        for col, stats in self.aggregations.items():
            if col not in X.columns:
                continue
            self._agg_data[col] = groups[col].agg(list(stats)).reset_index()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()

        valid_groups = [c for c in self.group_by if c in X.columns]
        if len(valid_groups) < 1:
            return X

        for col, agg_df in self._agg_data.items():
            if col not in X.columns:
                continue
            for stat in self.aggregations.get(col, []):
                name = f"card_{stat}_{col.lower()}"
                if stat not in agg_df.columns:
                    continue
                X = X.merge(
                    agg_df.rename(columns={stat: name})[valid_groups + [name]],
                    on=valid_groups,
                    how="left",
                )
        return X
