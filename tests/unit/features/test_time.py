import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import data_frames, column, range_indexes
from hypothesis.strategies import integers

from src.features.time import TimeFeatureExtractor


class TestTimeFeatures:
    def test_hour_of_day(self, sample_df: pd.DataFrame):
        result = TimeFeatureExtractor().transform(sample_df)
        assert list(result["hour_of_day"]) == [0, 0, 0, 1]

    def test_day_of_week(self, sample_df: pd.DataFrame):
        result = TimeFeatureExtractor().transform(sample_df)
        assert list(result["day_of_week"]) == [0, 1, 2, 0]

    def test_day_of_month(self, sample_df: pd.DataFrame):
        result = TimeFeatureExtractor().transform(sample_df)
        assert list(result["day_of_month"]) == [0, 1, 2, 0]


class TestPartialUsage:
    def test_use_hour_only(self, sample_df: pd.DataFrame):
        result = TimeFeatureExtractor(
            use_hour=True, use_dow=False, use_dom=False
        ).transform(sample_df)
        assert "hour_of_day" in result.columns
        assert "day_of_week" not in result.columns
        assert "day_of_month" not in result.columns


class TestEdgeCases:
    def test_no_transactiondt_returns_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = TimeFeatureExtractor().transform(df)
        assert list(result.columns) == ["a"]

    def test_disabled_returns_unchanged(self, sample_df: pd.DataFrame):
        result = TimeFeatureExtractor(enabled=False).transform(sample_df)
        assert "hour_of_day" not in result.columns


class TestHypothesisContracts:
    @given(
        data_frames(
            [
                column(
                    "TransactionDT", elements=integers(min_value=0, max_value=20000000)
                )
            ],
            index=range_indexes(min_size=0, max_size=10),
        )
    )
    def test_never_crashes(self, df):
        if "TransactionDT" not in df.columns:
            return
        result = TimeFeatureExtractor().transform(df)
        assert isinstance(result, pd.DataFrame)
