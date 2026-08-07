import os
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf


def _find_project_root(
    anchor_files: tuple[str, ...] = ("pyproject.toml", "dvc.yaml", ".git"),
) -> Path:
    """Set the project root directory with an env varible, otherwise,
    navigate up from this file until you find the project root."""
    if env_root := os.getenv("PROJECT_ROOT"):
        return Path(env_root).resolve()

    current = Path(__file__).resolve().parent

    while current != current.parent:
        if any((current / anchor).exists() for anchor in anchor_files):
            return current
        current = current.parent

    raise FileNotFoundError(f"No project root found.\nAnchors searched: {anchor_files}")


def _load_config_as_dict(config_path: Path) -> Mapping[str, Any]:
    """Loads a YAML config file using OmegaConf and guarantees it returns a dictionary."""
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config_dict, dict), (
        f"{config_path.name} could not be converted to a dictionary."
    )
    return cast(Mapping[str, Any], config_dict)


PROJECT_ROOT = _find_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

data_config = _load_config_as_dict(CONFIGS_DIR / "data.yaml")
evaluate_config = _load_config_as_dict(CONFIGS_DIR / "evaluate.yaml")
features_config = _load_config_as_dict(CONFIGS_DIR / "features.yaml")
mlflow_config = _load_config_as_dict(CONFIGS_DIR / "mlflow.yaml")
train_config = _load_config_as_dict(CONFIGS_DIR / "train.yaml")


DATA_DIR = PROJECT_ROOT / data_config["base_dir"]
RAW_DIR = PROJECT_ROOT / data_config["raw_dir"]
PROCESSED_DIR = PROJECT_ROOT / data_config["processed_dir"]
FEATURES_DIR = PROJECT_ROOT / data_config["features_dir"]


DATA_FILES = {
    name: RAW_DIR / filename for name, filename in data_config.get("files", {}).items()
}

if __name__ == "__main__":
    print(features_config)
