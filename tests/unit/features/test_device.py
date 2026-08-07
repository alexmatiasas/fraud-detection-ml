import pandas as pd
from hypothesis import assume, given
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import none, one_of, text

from fdml.features.device import DeviceFeatureExtractor


class TestOsDetection:
    def test_iphone_detected_as_ios(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor().transform(sample_df)
        assert list(result["device_os"]) == ["ios", "missing", "android", "missing"]

    def test_windows_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["Windows 10"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_os"].iloc[0] == "windows"

    def test_macos_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["Mac OS X"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_os"].iloc[0] == "macos"

    def test_linux_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["Linux"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_os"].iloc[0] == "linux"

    def test_unknown_os(self):
        df = pd.DataFrame({"DeviceInfo": ["Commodore64"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_os"].iloc[0] == "unknown"


class TestBrandDetection:
    def test_samsung_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["SM-G950U Build/NRD90M"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_brand"].iloc[0] == "samsung"

    def test_apple_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["iPhone"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_brand"].iloc[0] == "apple"

    def test_huawei_detected(self):
        df = pd.DataFrame({"DeviceInfo": ["HUAWEI P30"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_brand"].iloc[0] == "huawei"

    def test_other_brand(self):
        df = pd.DataFrame({"DeviceInfo": ["SomeUnknownBrand"]})
        result = DeviceFeatureExtractor().transform(df)
        assert result["device_brand"].iloc[0] == "other"


class TestEdgeCases:
    def test_missing_deviceinfo_gets_missing(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor().transform(sample_df)
        assert result["device_os"].iloc[1] == "missing"
        assert result["device_brand"].iloc[1] == "missing"

    def test_deviceinfo_column_is_dropped(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor().transform(sample_df)
        assert "DeviceInfo" not in result.columns

    def test_no_deviceinfo_column_returns_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = DeviceFeatureExtractor().transform(df)
        assert list(result.columns) == ["a"]

    def test_disabled_returns_unchanged(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor(enabled=False).transform(sample_df)
        assert "DeviceInfo" in result.columns
        assert "device_os" not in result.columns

    def test_other_columns_preserved(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor().transform(sample_df)
        for c in ["TransactionID", "isFraud", "TransactionAmt"]:
            assert c in result.columns

    def test_use_os_false_skips_os(self, sample_df: pd.DataFrame):
        result = DeviceFeatureExtractor(use_os=False).transform(sample_df)
        assert "device_os" not in result.columns
        assert "device_brand" in result.columns


class TestHypothesisContracts:
    @given(
        data_frames(
            [
                column("DeviceInfo", elements=one_of(text(max_size=50), none())),
            ],
            index=range_indexes(min_size=0, max_size=10),
        )
    )
    def test_never_crashes(self, df):
        assume("DeviceInfo" in df.columns)
        assume(df["DeviceInfo"].dtype == "object" or df["DeviceInfo"].dtype == "O")
        result = DeviceFeatureExtractor().transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_get_params_exposes_switches(self):
        ext = DeviceFeatureExtractor(use_os=True, use_brand=False)
        params = ext.get_params()
        assert params["use_os"] is True
        assert params["use_brand"] is False
