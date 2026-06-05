import pandas as pd

from src.features.base import BaseFeatureTransformer

_NUMERIC_DTYPES = (
    "float64",
    "float32",
    "int64",
    "int32",
    "int16",
    "int8",
    "Int64",
    "Int32",
    "Int16",
    "Int8",
)
_CATEGORY_DTYPES = ("object", "category")


class MissingImputer(BaseFeatureTransformer):
    """Impute missing values with LightGBM-friendly sentinels.

    * Numeric columns → ``numerical_value`` (default ``-999``).
    * Object / category columns → ``categorical_value`` (default ``"missing"``).

    Args:
        enabled: When False the transform returns X unchanged.
        numerical_value: Sentinel for missing numeric values.
        categorical_value: Sentinel for missing categorical values.
    """

    def __init__(
        self,
        enabled: bool = True,
        numerical_value: int = -999,
        categorical_value: str = "missing",
    ):
        self.enabled = enabled
        self.numerical_value = numerical_value
        self.categorical_value = categorical_value

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()

        num_cols = X.select_dtypes(include=_NUMERIC_DTYPES).columns
        cat_cols = X.select_dtypes(include=_CATEGORY_DTYPES).columns

        for col in num_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(self.numerical_value)

        for col in cat_cols:
            if not X[col].isna().any():
                continue
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = (
                    X[col]
                    .cat.add_categories([self.categorical_value])
                    .fillna(self.categorical_value)
                )
            else:
                X[col] = X[col].fillna(self.categorical_value)

        return X
