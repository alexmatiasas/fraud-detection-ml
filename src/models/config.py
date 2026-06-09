from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.schemas.evaluate import EvaluateConfig
from src.schemas.mlflow import MlflowFullConfig
from src.schemas.train import TrainConfig


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
