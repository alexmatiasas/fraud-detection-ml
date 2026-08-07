import pandas as pd

from fdml.features.vfilter import VFeatureFilter


class TestVFeatureFilter:
    def test_zero_variance_v_dropped(self, v_df: pd.DataFrame):
        vf = VFeatureFilter(variance_threshold=0.01)
        vf.fit(v_df)
        result = vf.transform(v_df)
        assert "V1" not in result.columns

    def test_nan_variance_v_dropped(self, v_df: pd.DataFrame):
        vf = VFeatureFilter(variance_threshold=0.01)
        vf.fit(v_df)
        result = vf.transform(v_df)
        assert "V5" not in result.columns

    def test_non_v_columns_preserved(self, v_df: pd.DataFrame):
        vf = VFeatureFilter(variance_threshold=0.01)
        vf.fit(v_df)
        result = vf.transform(v_df)
        assert "not_v" in result.columns
        assert "id" in result.columns

    def test_high_variance_v_kept(self, v_df: pd.DataFrame):
        vf = VFeatureFilter(variance_threshold=0.01)
        vf.fit(v_df)
        result = vf.transform(v_df)
        kept = [c for c in result.columns if c.startswith("V")]
        assert len(kept) > 0


class TestEdgeCases:
    def test_no_v_columns_returns_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        vf = VFeatureFilter()
        vf.fit(df)
        result = vf.transform(df)
        assert list(result.columns) == ["a"]

    def test_disabled_returns_unchanged(self, v_df: pd.DataFrame):
        vf = VFeatureFilter(enabled=False)
        vf.fit(v_df)
        result = vf.transform(v_df)
        assert "V1" in result.columns
