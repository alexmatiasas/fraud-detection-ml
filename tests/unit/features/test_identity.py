import pandas as pd

from fdml.features.identity import IdentityFlagExtractor


class TestIdentityFlagExtractor:
    def test_has_identity_flagged(self):
        df = pd.DataFrame(
            {
                "DeviceType": ["mobile", None, "desktop", None],
                "DeviceInfo": ["iPhone", None, None, "Android"],
            }
        )
        result = IdentityFlagExtractor().transform(df)
        assert list(result["has_identity"]) == [1, 0, 1, 1]
        assert result["has_identity"].dtype.name == "int8"

    def test_no_markers_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = IdentityFlagExtractor().transform(df)
        assert list(result.columns) == ["a"]

    def test_disabled_unchanged(self):
        df = pd.DataFrame({"DeviceType": ["mobile", None]})
        result = IdentityFlagExtractor(enabled=False).transform(df)
        assert "has_identity" not in result.columns
