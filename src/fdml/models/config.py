from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.mlflow import MlflowFullConfig
from fdml.schemas.train import TrainConfig

logger = logging.getLogger(__name__)


def resolve_mlflow_tracking(mlflow_cfg: MlflowFullConfig) -> MlflowFullConfig:
    """Auto-detect DagsHub from .env and override tracking URI if found."""
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
        except Exception as exc:
            logger.warning("  MLflow: could not load .env: %s", exc)
    return mlflow_cfg


def load_train_config(
    config_path: str = "configs/train.yaml",
    cli_args: list[str] | None = None,
) -> TrainConfig:
    raw = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli(args_list=cli_args)
    merged = OmegaConf.merge(raw, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    return TrainConfig.model_validate(data)


def load_evaluation_config(
    config_path: str = "configs/evaluate.yaml",
    cli_args: list[str] | None = None,
) -> EvaluateConfig:
    raw = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli(args_list=cli_args)
    merged = OmegaConf.merge(raw, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    return EvaluateConfig.model_validate(data)


def load_mlflow_config(
    config_path: str = "configs/mlflow.yaml",
    cli_args: list[str] | None = None,
) -> MlflowFullConfig:
    raw = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli(args_list=cli_args)
    merged = OmegaConf.merge(raw, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    return MlflowFullConfig.model_validate(data)


def save_model_card(content: str, path: str = "models/model_card.md") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
