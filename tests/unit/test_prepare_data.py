import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.prepare_data import (
    DTYPE_MAP,
    _downcast,
    build_dtype_map,
    build_transaction_schema,
    build_identity_schema,
)


class TestDTYPE_MAP:
    def test_has_known_column(self):
        assert DTYPE_MAP["isFraud"] == "int8"
        assert DTYPE_MAP["TransactionAmt"] == "float32"
        assert DTYPE_MAP["TransactionID"] == "int32"

    def test_category_columns_included(self):
        cat_cols = {k for k, v in DTYPE_MAP.items() if v == "category"}
        assert "ProductCD" in cat_cols
        assert "card4" in cat_cols
        assert "M4" in cat_cols
        assert "DeviceType" in cat_cols

    def test_m1_not_in_map(self):
        assert "M1" not in DTYPE_MAP

    def test_m4_present(self):
        assert DTYPE_MAP["M4"] == "category"


class TestBuildDtypeMap:
    def test_adds_v_features(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("TransactionID,V1,V2,V5\n1,0.5,1.0,2.0\n")
        result = build_dtype_map(csv)
        assert result["TransactionID"] == "int32"
        assert result["V1"] == "float32"
        assert result["V2"] == "float32"
        assert result["V5"] == "float32"

    def test_preserves_existing_keys(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("isFraud,TransactionID\n0,1\n")
        result = build_dtype_map(csv)
        assert result["isFraud"] == "int8"

    def test_handles_empty_v_cols(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("TransactionID,isFraud\n1,0\n")
        result = build_dtype_map(csv)
        v_keys = [k for k in result if k.startswith("V")]
        assert len(v_keys) == 0


class TestBuildTransactionSchema:
    def test_validates_ok(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
                "target": "isFraud",
                "validation": {"target_values": [0, 1]},
            }
        )
        schema = build_transaction_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "isFraud": [0, 1, 0],
                "TransactionAmt": [100.0, 50.0, 200.0],
                "ExtraCol": ["a", "b", "c"],
            }
        )
        result = schema.validate(df)
        assert result is not None

    def test_rejects_bad_is_fraud(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
                "target": "isFraud",
                "validation": {"target_values": [0, 1]},
            }
        )
        schema = build_transaction_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "isFraud": [0, 5],
                "TransactionAmt": [100.0, 50.0],
            }
        )
        with pytest.raises(Exception):
            schema.validate(df)

    def test_rejects_negative_amount(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
                "target": "isFraud",
                "validation": {"target_values": [0, 1]},
            }
        )
        schema = build_transaction_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "isFraud": [0, 1],
                "TransactionAmt": [100.0, -50.0],
            }
        )
        with pytest.raises(Exception):
            schema.validate(df)


class TestBuildIdentitySchema:
    def test_validates_ok(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
            }
        )
        schema = build_identity_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "DeviceType": ["mobile", "desktop", None],
            }
        )
        result = schema.validate(df)
        assert result is not None

    def test_rejects_duplicate_transaction_id(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
            }
        )
        schema = build_identity_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, 1, 3],
            }
        )
        with pytest.raises(Exception):
            schema.validate(df)

    def test_rejects_null_transaction_id(self):
        cfg = OmegaConf.create(
            {
                "join_key": "TransactionID",
            }
        )
        schema = build_identity_schema(cfg)
        df = pd.DataFrame(
            {
                "TransactionID": [1, None, 3],
            }
        )
        with pytest.raises(Exception):
            schema.validate(df)


class TestDowncast:
    def test_float64_to_float32(self):
        df = pd.DataFrame({"Amt": [100.5, 200.7]})
        assert df["Amt"].dtype == "float64"
        _downcast(df, {"Amt": "float32"})
        assert df["Amt"].dtype == "float32"

    def test_object_to_category(self):
        df = pd.DataFrame({"Col": ["a", "b", "c"]})
        assert df["Col"].dtype == "object"
        _downcast(df, {"Col": "category"})
        assert df["Col"].dtype == "category"

    def test_int64_to_int16(self):
        df = pd.DataFrame({"Val": [100, 200, 300]})
        _downcast(df, {"Val": "int16"})
        assert df["Val"].dtype == "int16"

    def test_int64_skipped_with_nan(self):
        df = pd.DataFrame({"Val": [100, None, 300]})
        _downcast(df, {"Val": "int16"})
        assert df["Val"].dtype == "float64"

    def test_unknown_col_skipped(self):
        df = pd.DataFrame({"A": [1.0, 2.0]})
        _downcast(df, {"NonExistent": "float32", "A": "float32"})
        assert df["A"].dtype == "float32"

    def test_v_features_cast_to_float32(self):
        df = pd.DataFrame({"V1": [0.5, 1.0], "V2": [2.0, 3.0]})
        _downcast(df, {"V1": "float32", "V2": "float32"})
        assert df["V1"].dtype == "float32"
        assert df["V2"].dtype == "float32"
