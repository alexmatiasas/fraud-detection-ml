import pandas as pd
from hypothesis import assume, given
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import none, one_of, sampled_from

from fdml.features.flags import MFlagEncoder


class TestBinaryMFlags:
    def test_m1_encoded_correctly(self, sample_df: pd.DataFrame):
        result = MFlagEncoder().transform(sample_df)
        assert list(result["M1"]) == [1, 0, -1, 1]

    def test_m1_dtype_is_int8(self, sample_df: pd.DataFrame):
        result = MFlagEncoder().transform(sample_df)
        assert result["M1"].dtype.name == "int8"

    def test_non_m_columns_untouched(self, sample_df: pd.DataFrame):
        result = MFlagEncoder().transform(sample_df)
        assert list(result["ProductCD"]) == ["W", "C", "W", "H"]

    def test_no_m_columns_returns_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = MFlagEncoder().transform(df)
        assert list(result.columns) == ["a"]


class TestM4:
    def test_m4_values_preserved(self, sample_df: pd.DataFrame):
        result = MFlagEncoder().transform(sample_df)
        assert list(result["M4"]) == ["M0", "M1", "missing", "M2"]

    def test_m4_stays_category(self, sample_df: pd.DataFrame):
        result = MFlagEncoder().transform(sample_df)
        assert isinstance(result["M4"].dtype, pd.CategoricalDtype)


class TestEdgeCases:
    def test_disabled_returns_unchanged(self, sample_df: pd.DataFrame):
        result = MFlagEncoder(enabled=False).transform(sample_df)
        assert list(result["M1"]) == ["T", "F", None, "T"]

    def test_m_cols_missing_skips_gracefully(self):
        df = pd.DataFrame({"not_m": [1]})
        result = MFlagEncoder().transform(df)
        assert list(result.columns) == ["not_m"]


class TestHypothesisContracts:
    @given(
        data_frames(
            [
                column("M1", elements=one_of(sampled_from(["T", "F"]), none())),
                column("M2", elements=one_of(sampled_from(["T", "F"]), none())),
            ],
            index=range_indexes(min_size=0, max_size=10),
        )
    )
    def test_never_crashes(self, df):
        assume("M1" in df.columns)
        result = MFlagEncoder().transform(df)
        assert isinstance(result, pd.DataFrame)
