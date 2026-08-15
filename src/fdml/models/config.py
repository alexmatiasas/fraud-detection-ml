"""Training-specific config loading.

The DagsHub/MLflow resolution helpers now live in ``fdml.config`` (shared with
the serving API) and are re-exported here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from fdml.config import load_mlflow_config, resolve_mlflow_tracking  # noqa: F401
from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.train import TrainConfig


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


def save_model_card(content: str, path: str = "models/model_card.md") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
