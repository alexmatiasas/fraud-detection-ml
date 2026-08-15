"""API integration tests using a fake pre-loaded loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fdml.api.dependencies as deps


class FakePipeline:
    def __init__(self, proba: float = 0.35) -> None:
        self.named_steps = {"features": object(), "model": object()}
        self._proba = proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), self._proba)


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch):
    loader = deps.get_model_loader()
    loader._pipeline = FakePipeline(proba=0.35)
    loader._source = "test"
    loader._run_id = "abc123"
    loader._version = 1
    loader._report = {
        "model_name": "lightgbm",
        "n_features": 341,
        "roc_auc": 0.9124,
        "best_threshold": 0.8,
    }
    loader._loaded_at = None
    loader._sample = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [18_403_224] * 3,
            "TransactionAmt": [99.0, 5000.0, 12.5],
            "ProductCD": ["W", "C", "W"],
            "card1": [13000, 13000, 14000],
        }
    )
    # Prevent the app lifespan from touching the network / disk.
    monkeypatch.setattr(loader, "load", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_sample", lambda *a, **k: None)
    yield loader
    monkeypatch.setattr(loader, "_sample", None)
    monkeypatch.setattr(loader, "_pipeline", None)


def make_client(loader) -> TestClient:
    from fdml.api.main import app

    return TestClient(app)


def test_health(loader) -> None:
    resp = make_client(loader).get("/v1/health/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_when_model_loaded(loader) -> None:
    resp = make_client(loader).get("/v1/ready/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "model_loaded": True}


def test_predict_by_transaction_id(loader) -> None:
    client = make_client(loader)
    resp = client.post("/v1/predict/", json={"transaction_id": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == 2
    assert body["probability"] == pytest.approx(0.35)
    assert body["is_fraud"] is False  # 0.35 < threshold 0.8
    assert body["threshold"] == pytest.approx(0.8)
    assert body["raw"]["TransactionAmt"] == pytest.approx(5000.0)
    assert body["raw"]["isFraud"] == 1  # ground-truth label kept for comparison


def test_predict_threshold_applied(loader) -> None:
    loader._pipeline = FakePipeline(proba=0.95)
    client = make_client(loader)
    body = client.post("/v1/predict/", json={"transaction_id": 1}).json()
    assert body["is_fraud"] is True  # 0.95 >= 0.8


def test_predict_unknown_transaction_404(loader) -> None:
    resp = make_client(loader).post("/v1/predict/", json={"transaction_id": 999})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_predict_batch(loader) -> None:
    client = make_client(loader)
    resp = client.post(
        "/v1/predict/batch", json=[{"transaction_id": 1}, {"transaction_id": 3}]
    )
    assert resp.status_code == 200
    assert [r["transaction_id"] for r in resp.json()] == [1, 3]


def test_batch_rejects_too_many_transactions(loader) -> None:
    items = [{"transaction_id": 1}] * 51  # BATCH_MAX = 50
    resp = make_client(loader).post("/v1/predict/batch", json=items)
    assert resp.status_code == 422


def test_model_info(loader) -> None:
    resp = make_client(loader).get("/v1/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "lightgbm"
    assert body["run_id"] == "abc123"
    assert body["n_features"] == 341
    assert body["metrics"]["roc_auc"] == pytest.approx(0.9124)


def test_metrics_endpoint(loader) -> None:
    resp = make_client(loader).get("/v1/metrics/")
    assert resp.status_code == 200
    assert resp.json()["roc_auc"] == pytest.approx(0.9124)
    assert resp.json()["best_threshold"] == pytest.approx(0.8)


def test_list_models(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loader,
        "list_versions",
        lambda: [
            {
                "version": 49,
                "status": "READY",
                "run_id": "abc123",
                "aliases": ["champion"],
                "active": True,
            },
            {
                "version": 51,
                "status": "ARCHIVED",
                "run_id": "def456",
                "aliases": [],
                "active": False,
            },
        ],
    )
    resp = make_client(loader).get("/v1/models/")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["version"] == 49
    assert body[0]["aliases"] == ["champion"]
    assert body[0]["active"] is True
    assert body[1]["active"] is False


def test_list_models_degrades_to_empty(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "list_versions", lambda: [])
    resp = make_client(loader).get("/v1/models/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_reload_requires_api_key(loader) -> None:
    resp = make_client(loader).put("/v1/model/reload")
    assert resp.status_code in (401, 403, 422)


def test_reload_with_valid_key(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDML_API_KEY", "secret")
    client = make_client(loader)
    resp = client.put("/v1/model/reload", headers={"X-API-Key": "secret"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"


def test_reload_with_invalid_key(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDML_API_KEY", "secret")
    resp = make_client(loader).put("/v1/model/reload", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_switch_requires_api_key(loader) -> None:
    resp = make_client(loader).put("/v1/model/switch", json={"version": 51})
    assert resp.status_code in (401, 403, 422)


def test_switch_with_valid_key(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDML_API_KEY", "secret")

    def fake_switch(version: int) -> None:
        loader._version = version
        loader._source = "mlflow"

    monkeypatch.setattr(loader, "load_version", fake_switch)
    client = make_client(loader)
    resp = client.put(
        "/v1/model/switch",
        json={"version": 51},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "switched"
    assert body["version"] == 51
    assert body["source"] == "mlflow"
    assert loader.version == 51


def test_switch_with_invalid_key(loader, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDML_API_KEY", "secret")
    resp = make_client(loader).put(
        "/v1/model/switch", json={"version": 51}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401
