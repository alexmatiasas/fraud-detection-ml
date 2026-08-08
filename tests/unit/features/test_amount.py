import numpy as np
import pandas as pd

from fdml.features.amount import AmountFeatureExtractor


class TestAmountFeatureExtractor:
    def test_log_created(self, sample_df: pd.DataFrame):
        result = AmountFeatureExtractor().transform(sample_df)
        expected = np.log1p(sample_df["TransactionAmt"])
        np.testing.assert_allclose(result["TransactionAmt_log"], expected)
        assert result["TransactionAmt_log"].dtype == np.dtype("float32")

    def test_is_round_amount(self, sample_df: pd.DataFrame):
        result = AmountFeatureExtractor().transform(sample_df)
        assert (result["is_round_amount"] == 1).all()

    def test_non_round_detected(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, 99.99, 0.5]})
        result = AmountFeatureExtractor().transform(df)
        assert list(result["is_round_amount"]) == [1.0, 0.0, 0.0]

    def test_missing_amount_stays_nan(self):
        df = pd.DataFrame({"TransactionAmt": [100.0, np.nan]})
        result = AmountFeatureExtractor().transform(df)
        assert result["TransactionAmt_log"].isna().iloc[1]
        assert np.isnan(result["is_round_amount"].iloc[1])

    def test_use_log_only(self):
        df = pd.DataFrame({"TransactionAmt": [10.0]})
        result = AmountFeatureExtractor(use_round=False).transform(df)
        assert "TransactionAmt_log" in result.columns
        assert "is_round_amount" not in result.columns

    def test_no_column_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = AmountFeatureExtractor().transform(df)
        assert list(result.columns) == ["a"]

    def test_disabled_unchanged(self, sample_df: pd.DataFrame):
        result = AmountFeatureExtractor(enabled=False).transform(sample_df)
        assert "TransactionAmt_log" not in result.columns
        assert "is_round_amount" not in result.columns
