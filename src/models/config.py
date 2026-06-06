from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.schemas.train import TrainConfig


def load_train_config(
    config_path: str = "configs/train.yaml",
    cli_args: list[str] | None = None,
) -> TrainConfig:
    """Load training configuration and override with CLI arguments.

    Usage from command line::

        python -m src.models.train model.name=xgboost split.test_size=0.3

    ``cli_args`` is provided for testing — when ``None`` (default) it reads
    ``sys.argv[1:]`` so that :func:`OmegaConf.from_cli` picks up arguments
    passed to the Python process.

    The config YAML is loaded via OmegaConf (supporting ``${...}`` variable
    interpolation), merged with any CLI overrides, converted to a plain dict,
    and validated via the ``TrainConfig`` Pydantic model.

    Args:
        config_path: Path to the YAML configuration file, relative or absolute.
        cli_args: Optional list of ``key=value`` overrides (used in tests).
            When ``None``, reads from ``sys.argv[1:]``.

    Returns:
        Validated ``TrainConfig`` instance.
    """
    raw = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli(args_list=cli_args)
    merged = OmegaConf.merge(raw, overrides)
    data = OmegaConf.to_container(merged, resolve=True)
    return TrainConfig.model_validate(data)


def save_model_card(content: str, path: str = "models/model_card.md") -> Path:
    """Write the model card Markdown to disk.

    Args:
        content: Model card in Markdown format.
        path: Output file path.

    Returns:
        Path to the written file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
