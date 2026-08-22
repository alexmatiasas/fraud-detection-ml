import pandas as pd

from fdml.features.card import CardAggregator


class TestCardAggregations:
    def test_fit_learns_groupby_stats(self):
        train = pd.DataFrame(
            {"card1": [100, 100, 200], "TransactionAmt": [10.0, 20.0, 30.0]}
        )
        agg = CardAggregator(
            group_by=["card1"], aggregations={"TransactionAmt": ["mean", "count"]}
        )
        agg.fit(train)
        assert "TransactionAmt" in agg._agg_data
        assert len(agg._agg_data["TransactionAmt"]) == 2

    def test_transform_merges_stats(self):
        train = pd.DataFrame(
            {"card1": [100, 100, 200], "TransactionAmt": [10.0, 20.0, 30.0]}
        )
        test = pd.DataFrame(
            {"card1": [100, 200, 100], "TransactionAmt": [5.0, 10.0, 15.0]}
        )
        agg = CardAggregator(
            group_by=["card1"], aggregations={"TransactionAmt": ["mean"]}
        )
        agg.fit(train)
        result = agg.transform(test)
        assert list(result["card_mean_transactionamt"]) == [15.0, 30.0, 15.0]

    def test_unseen_group_gets_nan(self):
        train = pd.DataFrame({"card1": [100], "TransactionAmt": [10.0]})
        test = pd.DataFrame({"card1": [999], "TransactionAmt": [50.0]})
        agg = CardAggregator(
            group_by=["card1"], aggregations={"TransactionAmt": ["mean"]}
        )
        agg.fit(train)
        result = agg.transform(test)
        assert result["card_mean_transactionamt"].isna().iloc[0]


class TestEdgeCases:
    def test_insufficient_group_columns_skips(self):
        df = pd.DataFrame({"a": [1], "TransactionAmt": [10.0]})
        agg = CardAggregator(group_by=["card1", "card2", "card3", "card5"])
        agg.fit(df)
        assert agg._agg_data == {}

    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame({"card1": [100], "TransactionAmt": [10.0]})
        agg = CardAggregator(enabled=False)
        agg.fit(df)
        result = agg.transform(df)
        assert "card_mean_transactionamt" not in result.columns
