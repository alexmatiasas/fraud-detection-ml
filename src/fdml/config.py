"""Shared configuration helpers used by both training and the serving API."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from fdml.schemas.mlflow import MlflowFullConfig

logger = logging.getLogger(__name__)


def resolve_mlflow_tracking(mlflow_cfg: MlflowFullConfig) -> MlflowFullConfig:
    """Auto-detect DagsHub and override the tracking URI if credentials exist.

    Resolution order:
    1. Environment variables already set (Docker ``--env-file``, Cloud Run,
       CI). This is the path that matters in the container — there is no
       ``.env`` file in the image.
    2. A ``.env`` file in the working directory (local development).
    """
    token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN")
    user = os.environ.get("DAGSHUB_USERNAME")
    repo = os.environ.get("DAGSHUB_REPO")

    if not (token and user and repo):
        env_path = Path(".env")
        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
                token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get(
                    "DAGSHUB_USER_TOKEN"
                )
                user = os.environ.get("DAGSHUB_USERNAME")
                repo = os.environ.get("DAGSHUB_REPO")
            except Exception as exc:
                logger.warning("  MLflow: could not load .env: %s", exc)

    if token and user and repo:
        # Token deliberately NOT embedded in the URI: MLflow echoes the
        # tracking URI in the run/experiment URLs it prints, so a token
        # in the URI would leak the credential into every log/CI output.
        # Auth comes from MLFLOW_TRACKING_USERNAME / _PASSWORD env vars,
        # which mlflow's HTTP store reads for basic auth.
        dagshub_uri = f"https://dagshub.com/{user}/{repo}.mlflow"
        mlflow_cfg.tracking.tracking_uri = dagshub_uri
        mlflow_cfg.tracking.backend = "dagshub"
        os.environ.setdefault("MLFLOW_TRACKING_URI", dagshub_uri)
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", user)
        os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", token)
        logger.info("  MLflow: DagsHub remote detected (%s/%s)", user, repo)
    return mlflow_cfg


def load_mlflow_config(
    config_path: str = "configs/mlflow.yaml",
    cli_args: list[str] | None = None,
) -> MlflowFullConfig:
    raw = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli(args_list=cli_args)
    merged = OmegaConf.merge(raw, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    return MlflowFullConfig.model_validate(data)
