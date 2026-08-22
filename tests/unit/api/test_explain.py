"""Unit tests for raw-feature overrides and SHAP explanation helpers."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fdml.api.internal.explain import OverrideError, apply_overrides
import fdml.api.internal.explain as explain_mod


def make_row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1],
            "isFraud": [0],
            "TransactionAmt": [99.0],
            "ProductCD": ["W"],
            "card1": [13000],
        }
    ).astype({"card1": "int32", "TransactionID": "int64", "isFraud": "int8"})


def test_apply_overrides_none_is_noop() -> None:
    row = make_row()
    out, applied = apply_overrides(row, None)
    assert applied == {}
    assert out.equals(row)


def test_apply_overrides_empty_is_noop() -> None:
    row = make_row()
    out, applied = apply_overrides(row, {})
    assert applied == {}
    assert out.equals(row)


def test_apply_overrides_updates_numeric_and_category() -> None:
    row = make_row()
    out, applied = apply_overrides(row, {"TransactionAmt": 500.0, "card1": 14000})
    assert out["TransactionAmt"].iloc[0] == pytest.approx(500.0)
    assert out["card1"].iloc[0] == 14000
    assert out["card1"].dtype == np.dtype("int32")  # int stays int
    assert applied == {"TransactionAmt": 500.0, "card1": 14000}


def test_apply_overrides_coerces_strings_to_category() -> None:
    out, applied = apply_overrides(make_row(), {"ProductCD": "C"})
    assert out["ProductCD"].iloc[0] == "C"
    assert applied == {"ProductCD": "C"}


def test_apply_overrides_unknown_feature_raises() -> None:
    with pytest.raises(OverrideError, match="Unknown feature"):
        apply_overrides(make_row(), {"NotAColumn": 1})


def test_apply_overrides_bad_numeric_value_raises() -> None:
    with pytest.raises(OverrideError, match="expects a number"):
        apply_overrides(make_row(), {"TransactionAmt": "abc"})


def test_apply_overrides_does_not_mutate_input() -> None:
    row = make_row()
    apply_overrides(row, {"TransactionAmt": 1.0})
    assert row["TransactionAmt"].iloc[0] == pytest.approx(99.0)


def test_compute_explanation_shap_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "shap":
            raise ImportError("no shap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = explain_mod.compute_explanation(object(), make_row())
    assert result is None


class _FakeFeatures:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[["TransactionAmt", "ProductCD"]].copy()


class _FakeModel:
    def __init__(self) -> None:
        self.feature_names_in_ = np.array(["TransactionAmt", "ProductCD"])


class _FakeExplain:
    def __init__(self, model: Any) -> None:
        self.expected_value = -1.5

    def shap_values(self, X: pd.DataFrame, check_additivity: bool = True) -> Any:
        return [np.array([[0.5, -0.3]])]


def test_compute_explanation_with_fake_shap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    import fdml.api.internal.explain as explain_mod

    class _FakeModule:
        TreeExplainer = _FakeExplain

    monkeypatch.setitem(sys.modules, "shap", _FakeModule)

    class _FakePipeline:
        named_steps = {"features": _FakeFeatures(), "model": _FakeModel()}

    result = explain_mod.compute_explanation(_FakePipeline(), make_row(), top_k=1)
    assert result is not None
    assert result["base_value"] == pytest.approx(-1.5)
    assert result["n_features"] == 2
    assert len(result["top_features"]) == 1
    top = result["top_features"][0]
    assert top["feature"] == "TransactionAmt"
    assert top["shap"] == pytest.approx(0.5)
    assert top["value"] == pytest.approx(99.0)


class TestCoerce:
    def test_bool_to_int(self):
        result = explain_mod._coerce(True, np.dtype("int64"), "col")
        assert result == 1
        assert isinstance(result, int)

    def test_float_value(self):
        result = explain_mod._coerce(3.14, np.dtype("float64"), "col")
        assert result == pytest.approx(3.14)

    def test_invalid_number_raises(self):
        with pytest.raises(OverrideError):
            explain_mod._coerce("not_a_number", np.dtype("float64"), "col")

    def test_bool_dtype(self):
        result = explain_mod._coerce(1, np.dtype("bool"), "col")
        assert result is True

    def test_datetime_value(self):
        result = explain_mod._coerce("2024-01-01", np.dtype("datetime64[ns]"), "col")
        assert isinstance(result, pd.Timestamp)

    def test_invalid_datetime_raises(self):
        with pytest.raises(OverrideError):
            explain_mod._coerce("not-a-date", np.dtype("datetime64[ns]"), "col")

    def test_string_fallback(self):
        result = explain_mod._coerce(42, np.dtype("O"), "col")
        assert result == "42"


class TestJsonable:
    def test_numpy_integer(self):
        assert explain_mod._jsonable(np.int64(5)) == 5

    def test_numpy_float(self):
        assert explain_mod._jsonable(np.float64(3.14)) == pytest.approx(3.14)

    def test_numpy_bool(self):
        assert explain_mod._jsonable(np.bool_(True)) is True

    def test_timestamp(self):
        ts = pd.Timestamp("2024-01-01")
        assert explain_mod._jsonable(ts) == "2024-01-01T00:00:00"

    def test_nan_returns_none(self):
        assert explain_mod._jsonable(np.nan) is None

    def test_regular_value(self):
        assert explain_mod._jsonable("hello") == "hello"


class TestFeatureNames:
    def test_dataframe_columns(self):
        X = pd.DataFrame({"a": [1], "b": [2]})
        names = explain_mod._feature_names(X, None, 2)
        assert names == ["a", "b"]

    def test_model_with_names(self):
        model = MagicMock()
        model.feature_names_in_ = ["x", "y"]
        names = explain_mod._feature_names(np.array([[1, 2]]), model, 2)
        assert names == ["x", "y"]

    def test_fallback(self):
        names = explain_mod._feature_names(np.array([[1, 2, 3]]), object(), 3)
        assert names == ["feature_0", "feature_1", "feature_2"]


class TestPositiveClassValues:
    def test_list_of_two(self):
        raw = [np.array([1, 2]), np.array([3, 4])]
        result = explain_mod._positive_class_values(raw)
        np.testing.assert_array_equal(result, [3, 4])

    def test_single_element_list(self):
        raw = [np.array([1, 2])]
        result = explain_mod._positive_class_values(raw)
        np.testing.assert_array_equal(result, [1, 2])

    def test_plain_array(self):
        raw = np.array([1, 2, 3])
        result = explain_mod._positive_class_values(raw)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_2d_array_first_row(self):
        raw = np.array([[1, 2], [3, 4]])
        result = explain_mod._positive_class_values(raw)
        np.testing.assert_array_equal(result, [1, 2])
