import pandas as pd

from src.features.base import BaseFeatureTransformer


class DeviceFeatureExtractor(BaseFeatureTransformer):
    """Extract structured device info from the free-text ``DeviceInfo`` column.

    Replaces the old binary ``has_identity`` flag with interpretable
    categorical features: ``device_os`` and ``device_brand``.  Rows without
    an identity record get ``"missing"`` for both — LightGBM handles this
    natively.

    The raw ``DeviceInfo`` string is dropped after extraction.

    Args:
        enabled: When False the transform returns X unchanged.
        use_os: If True, create the ``device_os`` column.
        use_brand: If True, create the ``device_brand`` column.
    """

    _OS_PATTERNS: list[tuple[str, list[str]]] = [
        ("android", ["ANDROID"]),
        ("android", ["BUILD/", "LRX", "MMB", "NRD", "KOT", "LMY", "JZO", "KTU"]),
        ("ios", ["IOS", "IPHONE", "IPAD"]),
        ("windows", ["WINDOWS", "WOW64"]),
        ("macos", ["MACOS", "MAC OS"]),
        ("linux", ["LINUX"]),
    ]

    _BRAND_PATTERNS: list[tuple[str, list[str]]] = [
        ("samsung", ["SAMSUNG"]),
        ("samsung", ["SM-"]),
        ("lg", ["LG-"]),
        ("lg", ["LG"]),
        ("huawei", ["HUAWEI", "HONOR"]),
        ("apple", ["APPLE", "IPHONE", "IPAD"]),
        ("xiaomi", ["XIAOMI", "MI "]),
        ("motorola", ["MOTO"]),
        ("google", ["GOOGLE", "PIXEL", "NEXUS"]),
        ("htc", ["HTC"]),
        ("lenovo", ["LENOVO"]),
        ("sony", ["SONY", "XPERIA"]),
        ("nokia", ["NOKIA"]),
        ("oneplus", ["ONEPLUS"]),
    ]

    def __init__(
        self,
        enabled: bool = True,
        use_os: bool = True,
        use_brand: bool = True,
    ):
        self.enabled = enabled
        self.use_os = use_os
        self.use_brand = use_brand

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()
        if "DeviceInfo" not in X.columns:
            return X

        has_info = X["DeviceInfo"].notna()

        if self.use_os:
            os_series = pd.Series("missing", index=X.index, dtype="object")
            os_series[has_info] = X.loc[has_info, "DeviceInfo"].apply(self._detect_os)
            X["device_os"] = os_series

        if self.use_brand:
            brand_series = pd.Series("missing", index=X.index, dtype="object")
            brand_series[has_info] = X.loc[has_info, "DeviceInfo"].apply(
                self._detect_brand
            )
            X["device_brand"] = brand_series

        X = X.drop(columns=["DeviceInfo"])
        return X

    @staticmethod
    def _detect_os(value: str) -> str:
        value_upper = value.upper()
        for os_name, patterns in DeviceFeatureExtractor._OS_PATTERNS:
            if any(p in value_upper for p in patterns):
                return os_name
        return "unknown"

    @staticmethod
    def _detect_brand(value: str) -> str:
        value_upper = value.upper()
        for brand, patterns in DeviceFeatureExtractor._BRAND_PATTERNS:
            if any(p in value_upper for p in patterns):
                return brand
        return "other"
