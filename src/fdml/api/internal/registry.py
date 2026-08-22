"""Multi-model discovery and loading from the MLflow registry.

Discovers every registered model, resolves the champion alias (falling back to
the newest READY version), and pulls the evaluation report (``report.json``)
from each model's run so the API can expose a leaderboard and per-model
predictions. Pipelines are loaded lazily on first use and cached in memory.

When a registered model is missing artifacts (e.g. no ``report.json`` in its
run, or a brand-new registration), the loader falls back to default values so
the API keeps answering — the frontend surfaces the gaps and the model can be
re-uploaded later.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fdml.api.internal.explain import compute_explanation
from fdml.api.internal.metrics import DEFAULT_THRESHOLD, METRIC_KEYS
from fdml.config import load_mlflow_config, resolve_mlflow_tracking

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fraud-detection-lgbm"
CHAMPION_ALIAS = "champion"


class ModelRegistryError(RuntimeError):
    """Raised when the registry is unreachable or a model cannot be loaded."""


def download_report(client: Any, run_id: str | None) -> dict[str, Any]:
    """Fetch ``report.json`` from the run, returning ``{}`` when unavailable.

    The heavy ``curves`` key is dropped — the API only needs the metrics.
    """
    if not run_id:
        return {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_artifacts(run_id, "report.json", dst_path=tmp)
            with Path(path).open(encoding="utf-8") as f:
                report = json.load(f)
    except Exception as exc:  # noqa: BLE001 - artifact may be missing
        logger.warning("  report.json unavailable for run %s: %s", run_id, exc)
        return {}
    report.pop("curves", None)
    return report


@dataclass
class RegisteredModel:
    """Metadata for one registered model, plus the report pulled from its run."""

    name: str
    version: int | None = None
    status: str = "READY"
    run_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime | None = None
    error: str | None = None

    @property
    def best_threshold(self) -> float:
        value = self.report.get("best_threshold", DEFAULT_THRESHOLD)
        try:
            return float(value)
        except (TypeError, ValueError):
            return DEFAULT_THRESHOLD

    @property
    def n_features(self) -> int | None:
        value = self.report.get("n_features")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def metric_summary(self) -> dict[str, Any]:
        return {key: self.report[key] for key in METRIC_KEYS if key in self.report}


class MultiModelLoader:
    """Discovers and serves every model registered in the MLflow registry."""

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        champion_alias: str = CHAMPION_ALIAS,
    ) -> None:
        self._default_model = default_model
        self._champion_alias = champion_alias
        self._models: dict[str, RegisteredModel] = {}
        self._pipelines: dict[str, Any] = {}

    # ── state ────────────────────────────────────────────────────────────
    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)

    def iter_models(self) -> list[RegisteredModel]:
        return list(self._models.values())

    def model_names(self) -> list[str]:
        return list(self._models)

    def get(self, name: str) -> RegisteredModel | None:
        return self._models.get(name)

    def is_default(self, name: str) -> bool:
        return name == self._default_model

    def is_loaded_model(self, name: str) -> bool:
        return name in self._pipelines

    # ── discovery ────────────────────────────────────────────────────────
    def load(self, force: bool = False) -> None:
        """Discover registered models and pull each one's evaluation report."""
        if self.is_loaded and not force:
            return
        from mlflow import MlflowClient

        resolve_mlflow_tracking(load_mlflow_config())
        client = MlflowClient()

        found: dict[str, RegisteredModel] = {}
        for registered in client.search_registered_models(max_results=100):
            info = self._resolve(client, registered)
            if info is not None:
                found[info.name] = info

        self._models = found
        self._pipelines.clear()
        if not found:
            raise ModelRegistryError("No registered models found in MLflow registry")
        logger.info("  Registered models: %s", ", ".join(found))

    def _resolve(self, client: Any, registered: Any) -> RegisteredModel | None:
        name = registered.name
        version = self._champion(client, name)
        if version is None:
            version = self._newest_ready(client, name)
        if version is None:
            logger.warning("  Model '%s' has no usable version in the registry", name)
            return None
        report = download_report(client, version.run_id)
        return RegisteredModel(
            name=name,
            version=int(version.version),
            status=version.status,
            run_id=version.run_id,
            aliases=getattr(version, "aliases", []) or [],
            report=report,
        )

    def _champion(self, client: Any, name: str) -> Any | None:
        from mlflow.exceptions import MlflowException

        try:
            return client.get_model_version_by_alias(name, self._champion_alias)
        except MlflowException:
            return None

    def _newest_ready(self, client: Any, name: str) -> Any | None:
        versions = client.search_model_versions(f"name='{name}'")
        ready = [v for v in versions if v.status == "READY"]
        if not ready:
            return None
        return max(ready, key=lambda v: int(v.version))

    # ── prediction ───────────────────────────────────────────────────────
    def predict_proba(self, name: str, X_raw: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities for ``name``, loading on demand."""
        pipeline = self._get_pipeline(name)
        return pipeline.predict_proba(X_raw)[:, 1]

    def explain(
        self, name: str, X_raw: pd.DataFrame, top_k: int = 20
    ) -> dict[str, Any] | None:
        """Tree SHAP explanation for ``name``, or ``None`` when unavailable."""
        try:
            pipeline = self._get_pipeline(name)
        except (ModelRegistryError, KeyError):
            return None
        return compute_explanation(pipeline, X_raw, top_k=top_k)

    def _get_pipeline(self, name: str) -> Any:
        if name in self._pipelines:
            return self._pipelines[name]
        info = self._models.get(name)
        if info is None:
            raise KeyError(f"Model '{name}' is not registered")
        if info.version is None:
            raise ModelRegistryError(f"Model '{name}' has no version to load")
        if info.error is not None:
            raise ModelRegistryError(info.error)
        from mlflow.sklearn import load_model

        try:
            model_uri = f"models:/{name}/{info.version}"
            pipeline = load_model(model_uri)
        except Exception as exc:  # noqa: BLE001 - surface every load failure
            info.error = f"Could not load {name} v{info.version} from MLflow: {exc}"
            raise ModelRegistryError(info.error) from exc
        info.loaded_at = datetime.now(timezone.utc)
        self._pipelines[name] = pipeline
        return pipeline
