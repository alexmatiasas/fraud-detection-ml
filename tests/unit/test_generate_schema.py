from __future__ import annotations

import json

from fdml.schemas.generate import generate_all


class TestGenerateAll:
    def test_creates_five_schema_files(self, tmp_path):
        generate_all(output_dir=str(tmp_path))
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 5

    def test_files_are_valid_json(self, tmp_path):
        generate_all(output_dir=str(tmp_path))
        for p in tmp_path.glob("*.json"):
            data = json.loads(p.read_text())
            assert "$schema" in data or "type" in data

    def test_expected_filenames(self, tmp_path):
        generate_all(output_dir=str(tmp_path))
        expected = {
            "data.schema.json",
            "features.schema.json",
            "train.schema.json",
            "evaluate.schema.json",
            "mlflow.schema.json",
        }
        actual = {p.name for p in tmp_path.glob("*.json")}
        assert expected == actual
