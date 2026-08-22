import pandas as pd

from fdml.features.freq import FrequencyEncoder


class TestFrequencyEncoding:
    def test_fit_learns_freq_map(self):
        train = pd.DataFrame({"card1": [100, 200, 100, 300]})
        enc = FrequencyEncoder(columns=["card1"])
        enc.fit(train)
        assert enc._freq_maps["card1"] == {100: 2, 200: 1, 300: 1}

    def test_transform_applies_freq(self):
        train = pd.DataFrame({"card1": [100, 200, 100]})
        test = pd.DataFrame({"card1": [100, 200]})
        enc = FrequencyEncoder(columns=["card1"])
        enc.fit(train)
        result = enc.transform(test)
        assert list(result["card1_freq"]) == [2, 1]

    def test_freq_dtype_is_int32(self):
        train = pd.DataFrame({"card1": [100, 200]})
        enc = FrequencyEncoder(columns=["card1"])
        enc.fit(train)
        result = enc.transform(train)
        assert result["card1_freq"].dtype.name == "int32"


class TestEdgeCases:
    def test_unseen_category_gets_freq_1(self):
        train = pd.DataFrame({"card1": [100, 200]})
        test = pd.DataFrame({"card1": [100, 999]})
        enc = FrequencyEncoder(columns=["card1"])
        enc.fit(train)
        result = enc.transform(test)
        assert result["card1_freq"].iloc[1] == 1

    def test_missing_column_skips(self):
        train = pd.DataFrame({"a": [1]})
        enc = FrequencyEncoder(columns=["missing_col"])
        enc.fit(train)
        assert enc._freq_maps == {}

    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame({"card1": [100]})
        enc = FrequencyEncoder(enabled=False, columns=["card1"])
        enc.fit(df)
        result = enc.transform(df)
        assert "card1_freq" not in result.columns

    def test_non_encoded_columns_preserved(self):
        train = pd.DataFrame({"card1": [100], "other": ["x"]})
        enc = FrequencyEncoder(columns=["card1"])
        enc.fit(train)
        result = enc.transform(train)
        assert "other" in result.columns
        assert "card1_freq" in result.columns
