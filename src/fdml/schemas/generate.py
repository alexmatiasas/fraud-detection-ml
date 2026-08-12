"""Generate JSON Schema files from Pydantic models."""

import json
from pathlib import Path

from fdml.schemas.data import DataConfig
from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.features import FeaturesConfig
from fdml.schemas.mlflow import MlflowFullConfig
from fdml.schemas.train import TrainConfig


def generate_all(output_dir: str = "schemas") -> None:
    schemas_dir = Path(output_dir)
    schemas_dir.mkdir(exist_ok=True)

    entries = [
        ("data.schema.json", DataConfig),
        ("features.schema.json", FeaturesConfig),
        ("train.schema.json", TrainConfig),
        ("evaluate.schema.json", EvaluateConfig),
        ("mlflow.schema.json", MlflowFullConfig),
    ]

    for filename, model_cls in entries:
        schema = model_cls.model_json_schema()
        path = schemas_dir / filename
        path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"  ✓ {path}")


if __name__ == "__main__":
    generate_all()
