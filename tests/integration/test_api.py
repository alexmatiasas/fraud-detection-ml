"""API integration tests using a fake pre-loaded loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fdml.api.dependencies as deps
from fdml.api.internal.registry import ModelRegistryError, RegisteredModel


class FakePipeline:
    def __init__(self, proba: float = 0.35) -> None:
        self.named_steps = {"features": object(), "model": object()}
        self._proba = proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 2), self._proba)


class FakeMultiLoader:
    """Minimal stand-in for MultiModelLoader."""

    def __init__(
        self,
        models: dict[str, RegisteredModel] | None = None,
        proba: float = 0.35,
    ) -> None:
        self.default_model = "fraud-detection-lgbm"
        self._models = models or {}
        self._proba = proba
        self._loaded: set[str] = set()

    def load(self, force: bool = False) -> None:
        return None

    def iter_models(self) -> list[RegisteredModel]:
        return list(self._models.values())

    def model_names(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> RegisteredModel | None:
        return self._models.get(name)

    def is_default(self, name: str) -> bool:
        return name == self.default_model

    def is_loaded_model(self, name: str) -> bool:
        return name in self._loaded

    def predict_proba(self, name: str, X: pd.DataFrame) -> np.ndarray:
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        model = self._models[name]
        if model.error is not None:
            raise ModelRegistryError(model.error)
        self._loaded.add(name)
        return np.full(len(X), self._proba)

    def explain(self, name: str, X: pd.DataFrame, top_k: int = 20) -> dict | None:
        return None


def make_registered_model(
    name: str,
    version: int,
    run_id: str,
    threshold: float = 0.8,
    roc_auc: float = 0.9124,
    error: str | None = None,
) -> RegisteredModel:
    return RegisteredModel(
        name=name,
        version=version,
        status="READY",
        run_id=run_id,
        aliases=["champion"] if name == "fraud-detection-lgbm" else [],
        report={
            "model_name": name.replace("fraud-detection-", ""),
            "n_features": 341,
            "roc_auc": roc_auc,
            "best_threshold": threshold,
        },
        error=error,
    )


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch):
    loader = deps.get_model_loader()
    loader._pipeline = FakePipeline(proba=0.35)
    loader._source = "test"
    loader._run_id = "abc123"
    loader._version = "1"
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


@pytest.fixture
def multi_loader(monkeypatch: pytest.MonkeyPatch):
    models = {
        "fraud-detection-lgbm": make_registered_model(
            "fraud-detection-lgbm", 49, "abc123", threshold=0.8
        ),
        "fraud-detection-xgboost": make_registered_model(
            "fraud-detection-xgboost", 7, "def456", threshold=0.65, roc_auc=0.9010
        ),
    }
    fake = FakeMultiLoader(models=models, proba=0.35)
    monkeypatch.setattr(deps, "_multi_loader", fake)
    yield fake
    monkeypatch.setattr(deps, "_multi_loader", None)


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


def test_ready_503_when_model_not_loaded(loader) -> None:
    loader._pipeline = None
    resp = make_client(loader).get("/v1/ready/")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready", "model_loaded": False}


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


def test_predict_with_overrides(loader) -> None:
    client = make_client(loader)
    resp = client.post(
        "/v1/predict/",
        json={
            "transaction_id": 2,
            "overrides": {"TransactionAmt": 999.0, "card1": 15000},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overrides_applied"] == {"TransactionAmt": 999.0, "card1": 15000}
    assert body["raw"]["TransactionAmt"] == pytest.approx(999.0)
    assert body["raw"]["card1"] == 15000
    assert body["probability"] == pytest.approx(0.35)


def test_predict_with_unknown_override_422(loader) -> None:
    resp = make_client(loader).post(
        "/v1/predict/", json={"transaction_id": 1, "overrides": {"NotAColumn": 1}}
    )
    assert resp.status_code == 422
    assert "Unknown feature" in resp.json()["detail"]


def test_predict_with_bad_override_value_422(loader) -> None:
    resp = make_client(loader).post(
        "/v1/predict/",
        json={"transaction_id": 1, "overrides": {"TransactionAmt": "abc"}},
    )
    assert resp.status_code == 422
    assert "expects a number" in resp.json()["detail"]


def test_predict_with_shap(loader, monkeypatch) -> None:
    def fake_explain(X, top_k=20):  # noqa: ANN001, ANN202
        return {
            "base_value": -1.5,
            "n_features": 2,
            "top_features": [{"feature": "TransactionAmt", "value": 99.0, "shap": 0.5}],
        }

    monkeypatch.setattr(loader, "explain", fake_explain)
    resp = make_client(loader).post("/v1/predict/", json={"transaction_id": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["explanation"]["base_value"] == pytest.approx(-1.5)
    assert body["explanation"]["top_features"][0]["feature"] == "TransactionAmt"
    assert body["explanation"]["top_features"][0]["shap"] == pytest.approx(0.5)


def test_predict_shap_absent_when_unavailable(loader) -> None:
    resp = make_client(loader).post("/v1/predict/", json={"transaction_id": 1})
    assert resp.status_code == 200
    assert resp.json()["explanation"] is None  # fake pipeline has no tree model


def test_list_transactions(loader) -> None:
    resp = make_client(loader).get("/v1/transactions/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["transactions"]) == 3
    first = body["transactions"][0]
    assert first["transaction_id"] == 1
    assert first["amount"] == pytest.approx(99.0)
    assert first["product_cd"] == "W"
    assert first["is_fraud"] is False


def test_list_transactions_paginated(loader) -> None:
    resp = make_client(loader).get("/v1/transactions/?limit=2&offset=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert [t["transaction_id"] for t in body["transactions"]] == [2, 3]


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


def test_list_models(loader, multi_loader) -> None:
    resp = make_client(loader).get("/v1/models/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] == "fraud-detection-lgbm"
    assert len(body["models"]) == 2
    lgbm, xgb = body["models"]
    assert lgbm["name"] == "fraud-detection-lgbm"
    assert lgbm["version"] == 49
    assert lgbm["aliases"] == ["champion"]
    assert lgbm["active"] is True
    assert lgbm["metrics"]["roc_auc"] == pytest.approx(0.9124)
    assert xgb["name"] == "fraud-detection-xgboost"
    assert xgb["active"] is False


def test_list_models_degrades_to_empty(loader, multi_loader, monkeypatch) -> None:
    monkeypatch.setattr(multi_loader, "_models", {})
    resp = make_client(loader).get("/v1/models/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == []


def test_model_detail(loader, multi_loader) -> None:
    resp = make_client(loader).get("/v1/models/fraud-detection-xgboost")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "fraud-detection-xgboost"
    assert body["version"] == 7
    assert body["best_threshold"] == pytest.approx(0.65)
    assert body["metrics"]["roc_auc"] == pytest.approx(0.9010)
    assert body["active"] is False


def test_model_detail_top_features(loader, multi_loader) -> None:
    multi_loader._models["fraud-detection-lgbm"].report["top_features"] = [
        {"feature": "TransactionAmt", "importance": 0.0674},
        {"feature": "TransactionDT", "importance": 0.0644},
    ]
    resp = make_client(loader).get("/v1/models/fraud-detection-lgbm")
    assert resp.status_code == 200
    top = resp.json()["top_features"]
    assert top is not None
    assert len(top) == 2
    assert top[0] == {"feature": "TransactionAmt", "importance": pytest.approx(0.0674)}


def test_model_detail_top_features_none_when_absent(loader, multi_loader) -> None:
    resp = make_client(loader).get("/v1/models/fraud-detection-lgbm")
    assert resp.status_code == 200
    assert resp.json()["top_features"] is None


def test_model_detail_unknown_404(loader, multi_loader) -> None:
    resp = make_client(loader).get("/v1/models/nope")
    assert resp.status_code == 404
    assert "not registered" in resp.json()["detail"]


def test_predict_with_named_model(loader, multi_loader) -> None:
    client = make_client(loader)
    resp = client.post(
        "/v1/predict/fraud-detection-xgboost", json={"transaction_id": 2}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == 2
    assert body["probability"] == pytest.approx(0.35)
    assert body["is_fraud"] is False  # 0.35 < threshold 0.65
    assert body["threshold"] == pytest.approx(0.65)
    assert body["model_version"] == "fraud-detection-xgboost:7"


def test_predict_with_named_model_and_overrides(loader, multi_loader) -> None:
    resp = make_client(loader).post(
        "/v1/predict/fraud-detection-xgboost",
        json={"transaction_id": 1, "overrides": {"TransactionAmt": 1.0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overrides_applied"] == {"TransactionAmt": 1.0}
    assert body["raw"]["TransactionAmt"] == pytest.approx(1.0)


def test_predict_with_unknown_model_404(loader, multi_loader) -> None:
    resp = make_client(loader).post("/v1/predict/nope", json={"transaction_id": 1})
    assert resp.status_code == 404
    assert "not registered" in resp.json()["detail"]


def test_compare_all_models(loader, multi_loader) -> None:
    resp = make_client(loader).post("/v1/predict/compare", json={"transaction_id": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == 2
    assert body["raw"]["TransactionAmt"] == pytest.approx(5000.0)
    names = [p["model"] for p in body["predictions"]]
    assert names == ["fraud-detection-lgbm", "fraud-detection-xgboost"]
    lgbm = body["predictions"][0]
    assert lgbm["probability"] == pytest.approx(0.35)
    assert lgbm["is_fraud"] is False  # 0.35 < threshold 0.8
    assert lgbm["threshold"] == pytest.approx(0.8)
    assert lgbm["version"] == 49
    xgb = body["predictions"][1]
    assert xgb["threshold"] == pytest.approx(0.65)
    assert xgb["is_fraud"] is False


def test_compare_selected_models(loader, multi_loader) -> None:
    resp = make_client(loader).post(
        "/v1/predict/compare",
        json={"transaction_id": 1, "models": ["fraud-detection-xgboost"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [p["model"] for p in body["predictions"]] == ["fraud-detection-xgboost"]


def test_compare_unknown_transaction_404(loader, multi_loader) -> None:
    resp = make_client(loader).post("/v1/predict/compare", json={"transaction_id": 999})
    assert resp.status_code == 404


def test_compare_reports_model_error(loader, multi_loader) -> None:
    multi_loader._models["fraud-detection-xgboost"].error = "boom"
    resp = make_client(loader).post("/v1/predict/compare", json={"transaction_id": 1})
    assert resp.status_code == 200
    items = {p["model"]: p for p in resp.json()["predictions"]}
    assert items["fraud-detection-xgboost"]["error"] == "boom"
    assert items["fraud-detection-xgboost"]["probability"] is None


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


# ── rate limiting ─────────────────────────────────────────────────────────
def _reset_limiter() -> None:
    from fdml.api.limiter import limiter

    limiter.reset()


def test_predict_rate_limit_429_after_budget(loader) -> None:
    """Exhaust the 60/min predict budget and expect a 429 with Retry-After."""
    _reset_limiter()
    client = make_client(loader)
    try:
        first = client.post("/v1/predict/", json={"transaction_id": 1})
        assert first.status_code == 200

        statuses = [
            client.post("/v1/predict/", json={"transaction_id": 1}).status_code
            for _ in range(60)
        ]
        assert statuses[-1] == 429
        assert statuses.count(429) >= 1  # only the tail hits the wall

        last = client.post("/v1/predict/", json={"transaction_id": 1})
        assert last.status_code == 429
        assert last.json()["detail"] == "Rate limit exceeded. Try again later."
        assert last.headers["Retry-After"] == "60"
    finally:
        _reset_limiter()


def test_batch_rate_limit_429_after_budget(loader) -> None:
    """The 20/min batch budget also 429s past the limit."""
    _reset_limiter()
    client = make_client(loader)
    try:
        for _ in range(20):
            resp = client.post("/v1/predict/batch", json=[{"transaction_id": 1}])
            assert resp.status_code == 200

        resp = client.post("/v1/predict/batch", json=[{"transaction_id": 1}])
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "60"
    finally:
        _reset_limiter()


def test_rate_limits_reset_between_tests(loader) -> None:
    """After a budget-exhausting test, reset() leaves a fresh window."""
    _reset_limiter()
    resp = make_client(loader).post("/v1/predict/", json={"transaction_id": 1})
    assert resp.status_code == 200
