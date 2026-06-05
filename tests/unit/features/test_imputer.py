import pandas as pd
from hypothesis import given, assume
from hypothesis.extra.pandas import data_frames, column, range_indexes
from hypothesis.strategies import floats, text, none, one_of

from src.features.imputer import MissingImputer


class TestNumericImputation:
    def test_float_missing_filled_with_neg999(self):
        df = pd.DataFrame({"amt": [100.0, None, 200.0]})
        result = MissingImputer().transform(df)
        assert list(result["amt"]) == [100.0, -999, 200.0]

    def test_no_missing_numeric_unchanged(self):
        df = pd.DataFrame({"amt": [100.0, 200.0]})
        result = MissingImputer().transform(df)
        assert list(result["amt"]) == [100.0, 200.0]


class TestCategoryImputation:
    def test_object_column_filled(self):
        df = pd.DataFrame({"cat": ["a", None, "c"]})
        result = MissingImputer().transform(df)
        assert list(result["cat"]) == ["a", "missing", "c"]

    def test_category_dtype_filled(self):
        df = pd.DataFrame({"cat": pd.Categorical(["a", None, "c"])})
        result = MissingImputer().transform(df)
        assert list(result["cat"]) == ["a", "missing", "c"]
        assert isinstance(result["cat"].dtype, pd.CategoricalDtype)
        assert "missing" in result["cat"].cat.categories


class TestEdgeCases:
    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame({"amt": [100.0, None]})
        result = MissingImputer(enabled=False).transform(df)
        assert result["amt"].isna().iloc[1]

    def test_custom_sentinel_values(self):
        df = pd.DataFrame(
            {
                "num": pd.array([pd.NA], dtype="Int64"),
                "cat": pd.array([pd.NA], dtype="object"),
            }
        )
        imp = MissingImputer(numerical_value=-1, categorical_value="NA")
        result = imp.transform(df)
        assert result["num"].iloc[0] == -1
        assert result["cat"].iloc[0] == "NA"

    def test_no_missing_values_elsewhere(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = MissingImputer().transform(df)
        assert list(result["a"]) == [1, 2]
        assert list(result["b"]) == ["x", "y"]


class TestHypothesisContracts:
    @given(
        data_frames(
            [
                column(
                    "num",
                    elements=one_of(
                        floats(allow_nan=False, allow_infinity=False), none()
                    ),
                    dtype=float,
                ),
                column("cat", elements=one_of(text(max_size=5), none()), dtype=object),
            ],
            index=range_indexes(min_size=0, max_size=10),
        )
    )
    def test_no_missing_after_impute(self, df):
        assume("num" in df.columns and "cat" in df.columns)
        result = MissingImputer().transform(df)
        assert result.isna().sum().sum() == 0
