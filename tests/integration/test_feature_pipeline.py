import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fdml.features.factory import create_pipeline, load_fe_config

_NROWS = 200
_RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def cfg():
    return load_fe_config()


@pytest.fixture(scope="module")
def pipeline(cfg) -> Pipeline:
    p, _ = create_pipeline(cfg)
    return p


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    n = _NROWS

    df = pd.DataFrame(
        {
            # ── core ────────────────────────────────────────
            "TransactionID": range(n),
            "isFraud": _RNG.choice([0, 1], n, p=[0.965, 0.035]),
            "TransactionDT": _RNG.integers(86400, 15811131, n),
            "TransactionAmt": _RNG.uniform(1, 1000, n).round(2),
            # ── categorical ─────────────────────────────────
            "ProductCD": _RNG.choice(["W", "C", "H", "R", "S"], n),
            "card4": _RNG.choice(["visa", "mastercard", "amex"], n),
            "card6": _RNG.choice(["credit", "debit"], n),
            "P_emaildomain": _RNG.choice(
                ["gmail.com", "yahoo.com", "outlook.com", None],
                n,
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "R_emaildomain": _RNG.choice(
                ["gmail.com", "yahoo.com", None],
                n,
                p=[0.4, 0.3, 0.3],
            ),
            # ── high cardinality ────────────────────────────
            "card1": _RNG.integers(1000, 15000, n),
            "card2": _RNG.integers(100, 600, n),
            "card3": _RNG.integers(50, 250, n),
            "card5": _RNG.integers(1, 200, n),
            "addr1": _RNG.integers(1, 1000, n),
            "addr2": _RNG.integers(1, 100, n),
            # ── identity ────────────────────────────────────
            "DeviceType": _RNG.choice(
                ["mobile", "desktop", None], n, p=[0.39, 0.59, 0.02]
            ),
            "DeviceInfo": _RNG.choice(
                ["iPhone", "SM-G950U Build/NRD90M", "Windows", "Mac OS X", None],
                n,
                p=[0.2, 0.2, 0.3, 0.1, 0.2],
            ),
            "id_01": _RNG.uniform(-5, 5, n).round(2),
            "id_02": _RNG.choice([True, False, None], n, p=[0.3, 0.3, 0.4]).astype(
                "object"
            ),
            "id_03": _RNG.uniform(0, 1, n).round(4),
            # ── M flags ─────────────────────────────────────
            "M1": _RNG.choice(["T", "F", None], n),
            "M2": _RNG.choice(["T", "F", None], n),
            "M4": pd.Categorical(_RNG.choice(["M0", "M1", "M2", None], n)),
            "M5": _RNG.choice(["T", "F", None], n),
            "M6": _RNG.choice(["T", "F", None], n),
            # ── count / delta ───────────────────────────────
            "C1": _RNG.uniform(0, 10, n),
            "C2": _RNG.uniform(0, 5, n),
            "D1": _RNG.uniform(0, 100, n),
            "D2": _RNG.uniform(0, 50, n),
            # ── distance ────────────────────────────────────
            "dist1": _RNG.uniform(0, 100, n),
            "dist2": _RNG.uniform(0, 50, n),
        }
    )

    for i in range(1, 21):
        df[f"V{i}"] = _RNG.normal(0, 1 + i * 0.1, n)
    df["V1"] = 0.0  # zero variance — should be dropped

    return df


class TestPipelineFixtures:
    """Test the pipeline itself with semi-realistic data."""

    def test_pipeline_fit_transform_succeeds(
        self, pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        X = pipeline.fit_transform(raw_df)
        assert isinstance(X, pd.DataFrame)

    def test_output_has_reasonable_shape(
        self, pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        X = pipeline.fit_transform(raw_df)
        assert X.shape[0] == _NROWS
        assert X.shape[1] > 20

    def test_no_missing_values_after_pipeline(
        self, pipeline: Pipeline, raw_df: pd.DataFrame
    ):
        X = pipeline.fit_transform(raw_df)
        assert X.isna().sum().sum() == 0

    def test_y_still_accessible(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        pipeline.fit_transform(raw_df)
        assert "isFraud" in raw_df.columns
        y = raw_df[["isFraud"]]
        assert y.shape == (_NROWS, 1)


class TestEngineeredFeatures:
    """Verify that each transformer creates its expected columns."""

    def test_device_os_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "device_os" in X.columns

    def test_device_brand_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "device_brand" in X.columns

    def test_device_info_is_dropped(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "DeviceInfo" not in X.columns

    def test_m1_encoded_as_int(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert X["M1"].dtype.name in ("int8", "int16", "int32", "int64")

    def test_m4_preserved_as_category(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert isinstance(X["M4"].dtype, pd.CategoricalDtype)

    def test_hour_of_day_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "hour_of_day" in X.columns
        assert X["hour_of_day"].between(0, 23).all()

    def test_freq_columns_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        freq_cols = [c for c in X.columns if c.endswith("_freq")]
        assert len(freq_cols) >= 3

    def test_has_identity_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "has_identity" in X.columns
        assert set(X["has_identity"].unique()) <= {0, 1}

    def test_amount_features_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "TransactionAmt_log" in X.columns
        assert "is_round_amount" in X.columns

    def test_cyclic_encoding_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "hour_of_day_sin" in X.columns
        assert "hour_of_day_cos" in X.columns

    def test_email_match_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        assert "p_r_domain_match" in X.columns

    def test_card_aggregation_created(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        card_cols = [c for c in X.columns if c.startswith("card_")]
        assert len(card_cols) >= 3

    def test_v_features_filtered(self, pipeline: Pipeline, raw_df: pd.DataFrame):
        X = pipeline.fit_transform(raw_df)
        v_cols = [c for c in X.columns if c.startswith("V")]
        assert "V1" not in v_cols  # zero variance
        assert len(v_cols) < 20  # some were dropped


class TestFeatureToggle:
    """Toggling steps off should remove their features."""

    def test_device_disabled_removes_os_and_brand(self, cfg, raw_df: pd.DataFrame):
        p, _ = create_pipeline(cfg)
        p.set_params(device__enabled=False)
        p.fit(raw_df)
        X = p.transform(raw_df)
        assert "device_os" not in X.columns
        assert "device_brand" not in X.columns
        assert "DeviceInfo" in X.columns  # not dropped

    def test_time_disabled_removes_hour(self, cfg, raw_df: pd.DataFrame):
        p, _ = create_pipeline(cfg)
        p.set_params(time__enabled=False)
        X = p.fit_transform(raw_df)
        assert "hour_of_day" not in X.columns

    def test_mflags_disabled_leaves_m_untouched(self, cfg, raw_df: pd.DataFrame):
        p, _ = create_pipeline(cfg)
        p.set_params(mflags__enabled=False)
        X = p.fit_transform(raw_df)
        assert X["M1"].dtype.name == "object"  # still T/F/None


class TestTrainTestConsistency:
    """The pipeline must behave consistently between fit and transform."""

    def test_frequency_encoding_applies_to_test(self, cfg):
        train = pd.DataFrame({"card1": [100, 200, 100, 300]})
        test = pd.DataFrame({"card1": [100, 999]})

        p, _ = create_pipeline(cfg)
        p.set_params(
            device__enabled=False,
            mflags__enabled=False,
            time__enabled=False,
            card__enabled=False,
            vfilter__enabled=False,
        )
        p.fit(train)
        result = p.transform(test)
        assert result["card1_freq"].iloc[0] == 2  # 100 appears twice in train
        assert result["card1_freq"].iloc[1] == 1  # 999 unseen

    def test_v_filter_uses_train_variance(self, cfg):
        train = pd.DataFrame({"V1": [0.0] * 10, "V2": _RNG.normal(0, 2, 10)})
        test = pd.DataFrame({"V1": [0.0] * 5, "V2": _RNG.normal(0, 2, 5)})

        p, _ = create_pipeline(cfg)
        p.set_params(
            device__enabled=False,
            mflags__enabled=False,
            time__enabled=False,
            freq__enabled=False,
            card__enabled=False,
            imputer__enabled=False,
        )
        p.fit(train)
        result = p.transform(test)
        assert "V1" not in result.columns  # zero variance in train → dropped
        assert "V2" in result.columns
