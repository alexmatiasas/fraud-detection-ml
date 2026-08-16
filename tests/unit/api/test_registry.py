"""Unit tests for MultiModelLoader discovery, fallback, and lazy loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fdml.api.internal.registry as reg_mod
from fdml.api.internal.registry import ModelRegistryError, MultiModelLoader


class FakeModelVersion:
    def __init__(
        self,
        version: int,
        run_id: str,
        status: str = "READY",
        aliases: list[str] | None = None,
        name: str = "m",
    ) -> None:
        self.version = str(version)
        self.run_id = run_id
        self.status = status
        self.aliases = aliases or []
        self.name = name


class FakeRegistered:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeClient:
    def __init__(
        self,
        versions: dict[str, list[FakeModelVersion]],
        aliases: dict[str, int] | None = None,
        reports: dict[str, dict] | None = None,
    ) -> None:
        self._versions = versions
        self._aliases = aliases or {}
        self._reports = reports or {}

    def search_registered_models(self, max_results: int = 100) -> list[FakeRegistered]:
        return [FakeRegistered(name) for name in self._versions]

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeModelVersion:
        from mlflow.exceptions import MlflowException

        alias_version = self._aliases.get(name)
        if alias_version is None:
            raise MlflowException(f"no alias '{alias}' for {name}")
        return next(v for v in self._versions[name] if v.version == str(alias_version))

    def search_model_versions(self, query: str) -> list[FakeModelVersion]:
        name = query.split("'")[1]
        return self._versions.get(name, [])

    def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str) -> str:
        report = self._reports.get(run_id)
        if report is None:
            raise FileNotFoundError(f"no report.json for {run_id}")
        path = Path(dst_path) / artifact_path
        path.write_text(json.dumps(report), encoding="utf-8")
        return str(path)


class FakePipeline:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), 0.35)


@pytest.fixture
def install_client(monkeypatch: pytest.MonkeyPatch):
    def _install(
        versions: dict[str, list[FakeModelVersion]],
        aliases: dict[str, int] | None = None,
        reports: dict[str, dict] | None = None,
    ) -> FakeClient:
        client = FakeClient(versions, aliases, reports)
        monkeypatch.setattr("mlflow.MlflowClient", lambda *a, **k: client)
        return client

    return _install


@pytest.fixture
def no_dagshub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reg_mod, "resolve_mlflow_tracking", lambda cfg: cfg)
    monkeypatch.setattr(reg_mod, "load_mlflow_config", lambda *a, **k: object())


@pytest.fixture
def fake_load_model(monkeypatch: pytest.MonkeyPatch):
    def _install(fn=None):
        fn = fn or (lambda model_uri: FakePipeline())
        import mlflow.sklearn  # materialize the lazy submodule before patching

        monkeypatch.setattr(mlflow.sklearn, "load_model", fn)
        return fn

    return _install


def test_load_discovers_registered_models(no_dagshub, install_client) -> None:
    install_client(
        versions={
            "fraud-detection-lgbm": [
                FakeModelVersion(49, "run-lgbm", aliases=["champion"])
            ],
            "fraud-detection-xgboost": [FakeModelVersion(7, "run-xgb")],
        },
        reports={
            "run-lgbm": {"model_name": "lightgbm", "best_threshold": 0.8},
            "run-xgb": {"model_name": "xgboost", "n_features": 300},
        },
    )
    loader = MultiModelLoader()
    loader.load()
    assert loader.is_loaded
    assert loader.model_names() == ["fraud-detection-lgbm", "fraud-detection-xgboost"]
    lgbm = loader.get("fraud-detection-lgbm")
    assert lgbm.version == 49
    assert lgbm.best_threshold == pytest.approx(0.8)
    assert lgbm.metric_summary()["best_threshold"] == pytest.approx(0.8)
    xgb = loader.get("fraud-detection-xgboost")
    assert xgb.version == 7
    assert xgb.n_features == 300


def test_champion_alias_preferred_over_newest(no_dagshub, install_client) -> None:
    install_client(
        versions={
            "fraud-detection-lgbm": [
                FakeModelVersion(40, "run-old", aliases=["champion"]),
                FakeModelVersion(41, "run-new"),
            ]
        },
        aliases={"fraud-detection-lgbm": 40},
        reports={"run-old": {}, "run-new": {}},
    )
    loader = MultiModelLoader()
    loader.load()
    assert loader.get("fraud-detection-lgbm").version == 40


def test_newest_ready_fallback_skips_archived(no_dagshub, install_client) -> None:
    install_client(
        versions={
            "fraud-detection-lgbm": [
                FakeModelVersion(1, "run-1", status="ARCHIVED"),
                FakeModelVersion(2, "run-2", status="READY"),
                FakeModelVersion(3, "run-3", status="READY"),
            ]
        },
        reports={"run-2": {}, "run-3": {}},
    )
    loader = MultiModelLoader()
    loader.load()
    assert loader.get("fraud-detection-lgbm").version == 3


def test_report_missing_falls_back_to_defaults(no_dagshub, install_client) -> None:
    install_client(
        versions={"fraud-detection-lgbm": [FakeModelVersion(5, "run-no-report")]},
        reports={},
    )
    loader = MultiModelLoader()
    loader.load()
    info = loader.get("fraud-detection-lgbm")
    assert info.report == {}
    assert info.best_threshold == pytest.approx(0.5)
    assert info.n_features is None
    assert info.metric_summary() == {}


def test_model_without_usable_version_is_skipped(no_dagshub, install_client) -> None:
    install_client(
        versions={
            "fraud-detection-lgbm": [FakeModelVersion(2, "run-2")],
            "fraud-detection-broken": [FakeModelVersion(1, "run-b", status="ARCHIVED")],
        },
        reports={"run-2": {}},
    )
    loader = MultiModelLoader()
    loader.load()
    assert loader.model_names() == ["fraud-detection-lgbm"]


def test_load_raises_when_registry_empty(no_dagshub, install_client) -> None:
    install_client(versions={})
    loader = MultiModelLoader()
    with pytest.raises(ModelRegistryError, match="No registered models"):
        loader.load()


def test_predict_proba_loads_lazily_and_caches(no_dagshub, fake_load_model) -> None:
    loaded: list[str] = []

    def counting_load(model_uri: str):
        loaded.append(model_uri)
        return FakePipeline()

    fake_load_model(counting_load)
    loader = MultiModelLoader()
    loader._models = {
        "fraud-detection-lgbm": reg_mod.RegisteredModel(
            name="fraud-detection-lgbm", version=49, run_id="run-1"
        )
    }
    X = pd.DataFrame({"x": [1.0]})
    assert not loader.is_loaded_model("fraud-detection-lgbm")

    proba = loader.predict_proba("fraud-detection-lgbm", X)
    assert proba[0] == pytest.approx(0.35)
    assert loaded == ["models:/fraud-detection-lgbm/49"]
    assert loader.is_loaded_model("fraud-detection-lgbm")

    loader.predict_proba("fraud-detection-lgbm", X)
    assert len(loaded) == 1  # cached, no second load


def test_predict_proba_unknown_model_raises(no_dagshub) -> None:
    loader = MultiModelLoader()
    with pytest.raises(KeyError, match="not registered"):
        loader.predict_proba("nope", pd.DataFrame({"x": [1.0]}))


def test_load_failure_records_error(no_dagshub, fake_load_model) -> None:
    def failing_load(model_uri: str):
        raise RuntimeError("download failed")

    fake_load_model(failing_load)
    loader = MultiModelLoader()
    loader._models = {
        "fraud-detection-lgbm": reg_mod.RegisteredModel(
            name="fraud-detection-lgbm", version=49, run_id="run-1"
        )
    }
    X = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ModelRegistryError, match="Could not load"):
        loader.predict_proba("fraud-detection-lgbm", X)
    info = loader.get("fraud-detection-lgbm")
    assert info.error is not None

    with pytest.raises(ModelRegistryError, match="Could not load"):
        loader.predict_proba("fraud-detection-lgbm", X)


def test_best_threshold_falls_back_on_bad_value() -> None:
    info = reg_mod.RegisteredModel(name="m", report={"best_threshold": "n/a"})
    assert info.best_threshold == pytest.approx(0.5)
