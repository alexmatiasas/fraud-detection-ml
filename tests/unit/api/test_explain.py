"""Unit tests for raw-feature overrides and SHAP explanation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fdml.api.internal.explain import OverrideError, apply_overrides
import fdml.api.internal.explain as explain_mod
import sys


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
