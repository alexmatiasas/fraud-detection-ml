"""Generate JSON Schema files from Pydantic models."""

import json
from pathlib import Path

from src.schemas.data import DataConfig
from src.schemas.features import FeaturesConfig
from src.schemas.train import TrainConfig


def generate_all(output_dir: str = "schemas") -> None:
    schemas_dir = Path(output_dir)
    schemas_dir.mkdir(exist_ok=True)

    entries = [
        ("features.schema.json", FeaturesConfig),
        ("data.schema.json", DataConfig),
        ("train.schema.json", TrainConfig),
    ]

    for filename, model_cls in entries:
        schema = model_cls.model_json_schema()
        path = schemas_dir / filename
        path.write_text(json.dumps(schema, indent=2))
        print(f"  ✓ {path}")


if __name__ == "__main__":
    generate_all()
