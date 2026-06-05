import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 0, 1, 0],
            "TransactionDT": [0, 86400, 172800, 3600],
            "TransactionAmt": [100.0, 50.0, 200.0, 75.0],
            "ProductCD": ["W", "C", "W", "H"],
            "DeviceType": ["mobile", None, "desktop", None],
            "DeviceInfo": ["iPhone", None, "Android", None],
            "M1": ["T", "F", None, "T"],
            "M4": pd.Categorical(["M0", "M1", None, "M2"]),
            "card1": [100, 200, 100, 300],
            "card2": [1, 2, 1, 3],
            "card3": [10, 20, 10, 30],
            "card5": [100, 200, 100, 300],
            "P_emaildomain": ["gmail.com", "yahoo.com", "gmail.com", "outlook.com"],
            "R_emaildomain": ["yahoo.com", None, "gmail.com", None],
        }
    )


@pytest.fixture()
def v_df() -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    n = 20
    cols = {"id": range(n), "not_v": 1.0}
    for i in range(1, 7):
        cols[f"V{i}"] = rng.normal(0, 1 + i * 0.1, n)
    cols["V1"] = 0.0  # zero variance
    cols["V5"] = float("nan")  # NaN variance
    return pd.DataFrame(cols)
