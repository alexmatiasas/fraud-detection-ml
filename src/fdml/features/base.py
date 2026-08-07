from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base for all feature transformers.

    Every transformer exposes an ``enabled`` flag that lets Optuna toggle
    whole feature groups on/off via ``set_params(step__enabled=False)``.

    Stateless transforms override only :meth:`transform`; stateful ones
    also override :meth:`fit` (e.g. to learn a frequency map from the
    training set).

    Args:
        BaseEstimator: Base class for all estimators in scikit-learn.
        TransformerMixin: Mixin class for all transformers
        in scikit-learn.
        ABC: Helper class that provides a standard way to
        create an ABC using inheritance.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureTransformer":
        self._fitted_ = True
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to *X*.

        Args:
            X: Input DataFrame.

        Returns:
            Transformed DataFrame (may add, modify or drop columns).
        """
