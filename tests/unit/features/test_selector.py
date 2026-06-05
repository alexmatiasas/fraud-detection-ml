import pandas as pd

from src.features.selector import FeatureSelector


class TestFeatureSelector:
    def test_keeps_only_specified_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        sel = FeatureSelector(feature_columns=["a", "c"])
        result = sel.transform(df)
        assert list(result.columns) == ["a", "c"]

    def test_vesta_columns_included_automatically(self):
        df = pd.DataFrame({"a": [1], "V1": [0.5], "V2": [0.3]})
        sel = FeatureSelector(feature_columns=["a"], vesta_include=True)
        result = sel.transform(df)
        assert "V1" in result.columns
        assert "V2" in result.columns

    def test_vesta_columns_not_included_when_flag_false(self):
        df = pd.DataFrame({"a": [1], "V1": [0.5]})
        sel = FeatureSelector(feature_columns=["a"], vesta_include=False)
        result = sel.transform(df)
        assert "V1" not in result.columns

    def test_target_column_excluded(self):
        df = pd.DataFrame({"a": [1], "isFraud": [0]})
        sel = FeatureSelector(feature_columns=["a", "isFraud"], target="isFraud")
        result = sel.transform(df)
        assert "isFraud" not in result.columns
        assert "a" in result.columns

    def test_missing_feature_column_skipped(self):
        df = pd.DataFrame({"a": [1]})
        sel = FeatureSelector(feature_columns=["a", "nonexistent"])
        result = sel.transform(df)
        assert list(result.columns) == ["a"]


class TestEdgeCases:
    def test_disabled_returns_unchanged(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        sel = FeatureSelector(enabled=False, feature_columns=["a"])
        result = sel.transform(df)
        assert "b" in result.columns

    def test_vesta_columns_sorted(self):
        df = pd.DataFrame({"V2": [0.3], "V1": [0.5], "a": [1]})
        sel = FeatureSelector(feature_columns=["a"], vesta_include=True)
        result = sel.transform(df)
        v_cols = [c for c in result.columns if c.startswith("V")]
        assert v_cols == sorted(v_cols)
