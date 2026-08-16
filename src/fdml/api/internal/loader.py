"""Model loading and prediction for the fraud detection service.

Resolution order (first success wins):

1. **MLflow registry** — ``models:/<name>/<alias>`` (e.g. ``champion``),
   falling back to the newest non-deleted version.
2. **Local joblib** — ``models/pipeline.joblib`` saved by the training stage.

The DagsHub tracking URI is resolved from ``.env`` via
``fdml.models.config.resolve_mlflow_tracking``. If no model is reachable the
loader stays unloaded and the API answers ``503`` on readiness/predict until
``reload()`` succeeds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fdml.api.internal.explain import compute_explanation
from fdml.api.internal.registry import download_report
from fdml.config import load_mlflow_config, resolve_mlflow_tracking

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = "data/samples/val_sample.parquet"
DROP_FOR_PREDICT = ("TransactionID", "isFraud")


class ModelLoadError(RuntimeError):
    """Raised when no model can be loaded from any configured source."""


def _read_report(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


class ModelLoader:
    """Caches the fitted pipeline, evaluation report, and demo sample."""

    def __init__(
        self,
        mlflow_model: str = "fraud-detection-lgbm",
        mlflow_alias: str = "champion",
        local_pipeline: str = "models/pipeline.joblib",
        report_path: str = "models/report.json",
        sample_path: str = DEFAULT_SAMPLE,
    ) -> None:
        self._pipeline: Any = None
        self._report: dict[str, Any] = {}
        self._sample: pd.DataFrame | None = None
        self._loaded_at: datetime | None = None
        self._source: str | None = None
        self._run_id: str | None = None
        self._version: str | None = None

        self._mlflow_model = mlflow_model
        self._mlflow_alias = mlflow_alias
        self._local_pipeline = local_pipeline
        self._report_path = report_path
        self._sample_path = sample_path

    # ── state ────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def report(self) -> dict[str, Any]:
        return self._report

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def model_version(self) -> str | None:
        if self._source == "mlflow":
            return f"{self._mlflow_model}:{self._version}"
        return None

    @property
    def sample(self) -> pd.DataFrame | None:
        return self._sample

    # ── loading ──────────────────────────────────────────────────────────
    def load(self, source: str | None = None, force: bool = False) -> None:
        """Load the pipeline from ``source`` (``"mlflow"``, ``"local"``, or ``None`` = auto).

        Args:
            source: Restrict resolution to one source.
            force: Reload even if a model is already cached.
        """
        if self.is_loaded and not force:
            return

        if source is None:
            sources: tuple[str, ...] = ("mlflow", "local")
        else:
            sources = (source,)

        errors: list[str] = []
        for candidate in sources:
            try:
                if candidate == "mlflow":
                    ok = self._load_from_mlflow()
                elif candidate == "local":
                    ok = self._load_from_local()
                else:
                    raise ModelLoadError(f"Unknown source: {candidate}")
            except Exception as exc:  # noqa: BLE001 - surface every source's failure
                errors.append(f"{candidate}: {exc}")
                logger.warning("  Model source '%s' failed: %s", candidate, exc)
                continue
            if ok:
                self._report = self._load_report()
                self._loaded_at = datetime.now(timezone.utc)
                logger.info(
                    "  Model loaded from %s (%d features)",
                    self._source,
                    len(self._report.get("top_features", [])),
                )
                return

        raise ModelLoadError("No model available. Tried: " + "; ".join(errors))

    def load_version(self, version: int) -> None:
        """Load a specific version of the configured model from the MLflow registry.

        Used by ``PUT /v1/model/switch`` to hot-swap the served model without a
        restart. Raises ``ModelLoadError`` if MLflow is unreachable or the
        version does not exist.
        """
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException
        from mlflow.sklearn import load_model

        resolve_mlflow_tracking(load_mlflow_config())
        client = MlflowClient()

        try:
            mv = client.get_model_version(self._mlflow_model, str(version))
        except MlflowException as exc:
            raise ModelLoadError(
                f"Version {version} of {self._mlflow_model} not found: {exc}"
            ) from exc

        try:
            model_uri = f"models:/{self._mlflow_model}/{version}"
            self._pipeline = load_model(model_uri)
        except MlflowException as exc:
            raise ModelLoadError(
                f"Could not load {self._mlflow_model} v{version} from MLflow: {exc}"
            ) from exc
        self._source = "mlflow"
        self._version = mv.version
        self._run_id = mv.run_id
        self._report = self._load_report()
        self._loaded_at = datetime.now(timezone.utc)
        logger.info("  Model switched to %s version %d", self._mlflow_model, version)

    def list_versions(self) -> list[dict[str, Any]]:
        """All versions of the configured model in the MLflow registry.

        Returns an empty list when the registry is unreachable (e.g. local dev
        with a joblib pipeline) so the endpoint degrades gracefully.
        """
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        try:
            resolve_mlflow_tracking(load_mlflow_config())
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{self._mlflow_model}'")
        except MlflowException as exc:
            logger.warning("  Could not list registry versions: %s", exc)
            return []

        rows: list[dict[str, Any]] = []
        for v in sorted(versions, key=lambda item: int(item.version), reverse=True):
            rows.append(
                {
                    "version": int(v.version),
                    "status": v.status,
                    "run_id": v.run_id,
                    "aliases": getattr(v, "aliases", []) or [],
                    "active": self._source == "mlflow" and v.version == self._version,
                }
            )
        return rows

    def _load_from_mlflow(self) -> bool:
        from mlflow import MlflowClient
        from mlflow.sklearn import load_model

        resolve_mlflow_tracking(load_mlflow_config())
        client = MlflowClient()

        version = self._latest_registry_version(client)
        if version is None:
            return False

        self._version = version.version
        self._run_id = version.run_id
        self._source = "mlflow"
        model_uri = f"models:/{self._mlflow_model}/{version.version}"
        self._pipeline = load_model(model_uri)
        return True

    def _latest_registry_version(self, client: Any) -> Any | None:
        from mlflow.exceptions import MlflowException

        try:
            return client.get_model_version_by_alias(
                self._mlflow_model, self._mlflow_alias
            )
        except MlflowException:
            pass

        versions = client.search_model_versions(f"name='{self._mlflow_model}'")
        active = [v for v in versions if v.status == "READY"]
        if not active:
            return None
        return max(active, key=lambda v: v.version)

    def _load_from_local(self) -> bool:
        path = Path(self._local_pipeline)
        if not path.exists():
            return False
        self._pipeline = joblib.load(path)
        self._source = "local"
        self._run_id = None
        self._version = None
        return True

    def _load_report(self) -> dict[str, Any]:
        """Evaluation report, from the local file or the MLflow run as fallback.

        The 11MB ``report.json`` is not shipped in the container image, so a
        fresh deploy has no local copy. When the model came from MLflow we
        pull the report straight from the run's artifacts; otherwise the
        report stays ``{}`` and the API falls back to default values.
        """
        report = _read_report(self._report_path)
        if report or self._source != "mlflow" or not self._run_id:
            return report
        try:
            from mlflow import MlflowClient

            resolve_mlflow_tracking(load_mlflow_config())
            return download_report(MlflowClient(), self._run_id)
        except Exception as exc:  # noqa: BLE001 - registry may be unreachable
            logger.warning("  Could not fetch report.json from MLflow: %s", exc)
            return {}

    # ── sample ───────────────────────────────────────────────────────────
    def load_sample(self, path: str | None = None) -> None:
        p = Path(path or self._sample_path)
        if not p.exists():
            logger.warning("  Demo sample not found: %s", p)
            self._sample = None
            return
        self._sample = pd.read_parquet(p)
        logger.info("  Demo sample loaded: %s rows", len(self._sample))

    def lookup(self, transaction_id: int) -> pd.DataFrame | None:
        """Return the raw row for a TransactionID, or ``None`` if absent.

        Returns a single-row DataFrame (not a Series) so the original parquet
        dtypes are preserved — extracting a Series coerces mixed columns to
        ``object``, which breaks numeric ufuncs downstream (e.g. ``np.sin``
        in the time features).
        """
        if self._sample is None:
            return None
        mask = self._sample["TransactionID"] == transaction_id
        match = self._sample.loc[mask]
        if match.empty:
            return None
        return match

    def list_transactions(
        self, limit: int | None = None, offset: int = 0
    ) -> tuple[int, list[dict[str, Any]]]:
        """Summaries of the demo-sample transactions for the selection UI.

        Returns ``(total, page)``; the page holds ``TransactionID``, amount,
        product code, and the ground-truth fraud label. Missing columns are
        skipped so the summary survives sample layout changes.
        """
        if self._sample is None:
            return 0, []
        cols = [
            c
            for c in ("TransactionID", "TransactionAmt", "ProductCD", "isFraud")
            if c in self._sample.columns
        ]
        if not cols:
            return len(self._sample), []
        df = self._sample[cols]
        total = len(df)
        page = df.iloc[offset : offset + limit] if limit else df.iloc[offset:]

        records: list[dict[str, Any]] = []
        for _, row in page.iterrows():
            record: dict[str, Any] = {}
            if "TransactionID" in cols:
                record["transaction_id"] = int(row["TransactionID"])
            if "TransactionAmt" in cols:
                record["amount"] = float(row["TransactionAmt"])
            if "ProductCD" in cols:
                record["product_cd"] = str(row["ProductCD"])
            if "isFraud" in cols:
                record["is_fraud"] = bool(int(row["isFraud"]))
            records.append(record)
        return total, records

    # ── prediction ───────────────────────────────────────────────────────
    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities for a raw (pre-FE) DataFrame."""
        if not self.is_loaded:
            raise ModelLoadError("Model not loaded")
        return self._pipeline.predict_proba(X_raw)[:, 1]

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted feature pipeline (for explainability later)."""
        if not self.is_loaded:
            raise ModelLoadError("Model not loaded")
        return self._pipeline.named_steps["features"].transform(X_raw)

    def explain(self, X_raw: pd.DataFrame, top_k: int = 20) -> dict[str, Any] | None:
        """Tree SHAP explanation, or ``None`` when unavailable."""
        if not self.is_loaded:
            return None
        return compute_explanation(self._pipeline, X_raw, top_k=top_k)


def to_model_input(row: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/target columns from a single-row DataFrame."""
    return row.drop(columns=list(DROP_FOR_PREDICT), errors="ignore")
