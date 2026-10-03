from __future__ import annotations

import logging

import mlflow

from fdml.schemas.mlflow import MlflowFullConfig

_REGISTRY_NAMES = {
    "lightgbm": "fraud-detection-lgbm",
    "xgboost": "fraud-detection-xgboost",
    "random_forest": "fraud-detection-rf",
}

logger = logging.getLogger(__name__)


def _passes_quality_gate(
    mlflow_cfg: MlflowFullConfig, auc: float, average_precision: float
) -> bool:
    """True when the run clears the registry's minimum-metric thresholds."""
    gate = mlflow_cfg.registry
    failures: list[str] = []
    if gate.min_auc > 0 and auc < gate.min_auc:
        failures.append(f"AUC {auc:.4f} < {gate.min_auc}")
    if gate.min_average_precision > 0 and average_precision < gate.min_average_precision:
        failures.append(f"AP {average_precision:.4f} < {gate.min_average_precision}")
    if failures:
        logger.warning("  Registry: quality gate REJECTED — %s", "; ".join(failures))
        if mlflow.active_run() is not None:
            mlflow.set_tag("validation_status", "rejected")
        return False
    return True


def _register_model(
    mlflow_cfg: MlflowFullConfig,
    auc: float,
    run_id: str,
    model_uri: str | None = None,
    average_precision: float = 0.0,
    cfg_model_name: str = "lightgbm",
) -> None:
    from mlflow import MlflowClient

    if not _passes_quality_gate(mlflow_cfg, auc, average_precision):
        return

    client = MlflowClient()
    model_name: str = _REGISTRY_NAMES.get(
        cfg_model_name, mlflow_cfg.registry.model_name or "fraud-detection-lgbm"
    )
    registry_tags = dict(mlflow_cfg.registry.tags)

    # MLflow 3 logs models as LoggedModels outside the run's artifacts; use
    # the models:/m-<id> URI returned by log_model(). The legacy
    # runs:/{run_id}/model pointer only works when the server resolves it to
    # that LoggedModel — unreliable on DagsHub (v50/v53 registered empty
    # schemas this way).
    if not model_uri:
        logger.warning(
            "  Registry: no logged-model URI (log_model disabled or failed) — skipping registration"
        )
        return

    try:
        client.create_registered_model(model_name, description=mlflow_cfg.registry.description)
        logger.info("  Registry: created model '%s'", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("  Registry: model '%s' exists, reusing (%s)", model_name, exc)

    try:
        _model_label = cfg_model_name.replace("_", " ").title()
        version_desc = f"{_model_label} baseline — AUC={auc:.4f}, AP={average_precision:.4f}"
        version_tags = {
            **registry_tags,
            "validation_status": "approved",
            "validation_auc": f"{auc:.4f}",
            "validation_ap": f"{average_precision:.4f}",
        }
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=version_desc,
            tags=version_tags,
        )
        version = mv.version
        mlflow.log_param("registered_model_version", version)
        mlflow.set_tag("registered_model_version", version)
        mlflow.set_tag("validation_status", "approved")
        logger.info("  Registry: created version %s (AUC=%.4f)", version, auc)

        try:
            champion_mv = client.get_model_version_by_alias(model_name, "champion")
            champion_run_id = champion_mv.run_id
            champion_auc = 0.0
            if champion_run_id is not None:
                champion_run = client.get_run(champion_run_id)
                champion_auc = champion_run.data.metrics.get("val/roc_auc", 0.0)

            if auc > champion_auc:
                client.set_registered_model_alias(model_name, "champion", version)
                logger.info(
                    "  Registry: alias 'champion' → v%s (AUC=%.4f > %.4f)",
                    version,
                    auc,
                    champion_auc,
                )
            else:
                client.set_registered_model_alias(model_name, "challenger", version)
                logger.info(
                    "  Registry: alias 'challenger' → v%s (AUC=%.4f ≤ champion %.4f)",
                    version,
                    auc,
                    champion_auc,
                )
        except Exception:  # noqa: BLE001
            client.set_registered_model_alias(model_name, "champion", version)
            logger.info("  Registry: alias 'champion' → version %s (first model)", version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Registry: failed to register model: %s", exc)
