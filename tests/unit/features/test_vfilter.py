import numpy as np
import pandas as pd

from fdml.features.vfilter import VFeatureFilter, _pairwise_corr


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


class TestPairwiseCorr:
    def test_matches_pandas_pairwise_corr(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 8)).astype(np.float64)
        X[:50, 1] = np.nan
        X[:80, 4] = np.nan
        X[:, 5] = np.nan  # fully missing column
        X[rng.random((200, 8)) < 0.1] = np.nan
        df = pd.DataFrame(X)

        expected = df.corr().to_numpy()
        got = _pairwise_corr(df.to_numpy())

        np.testing.assert_allclose(
            got[np.isfinite(expected)], expected[np.isfinite(expected)], atol=1e-12
        )

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(50, 4))
        got = _pairwise_corr(X)
        np.testing.assert_array_equal(np.diag(got), np.ones(4))

    def test_correlated_pair_above_threshold(self, v_df: pd.DataFrame):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 2))
        df = pd.DataFrame(
            {
                "id": range(200),
                "not_v": 1.0,
                "V1": X[:, 0],
                "V2": X[:, 0] + rng.normal(0, 0.01, 200),
            }
        )
        vf = VFeatureFilter()
        vf.fit(df)
        result = vf.transform(df)
        assert "V1" in result.columns
        assert "V2" not in result.columns
