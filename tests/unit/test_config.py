import pytest
from omegaconf import OmegaConf

from fdml.models.config import (
    load_mlflow_config,
    load_train_config,
    resolve_mlflow_tracking,
)


class TestResolveMlflowTracking:
    def test_dagshub_uri_does_not_embed_token(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "DAGSHUB_TOKEN=sekret123\nDAGSHUB_USERNAME=alice\nDAGSHUB_REPO=repo\n"
        )
        for var in ("DAGSHUB_TOKEN", "DAGSHUB_USERNAME", "DAGSHUB_REPO"):
            monkeypatch.delenv(var, raising=False)
        cfg = load_mlflow_config()
        monkeypatch.chdir(tmp_path)
        cfg = resolve_mlflow_tracking(cfg)
        assert "sekret123" not in cfg.tracking.tracking_uri
        assert cfg.tracking.tracking_uri == "https://dagshub.com/alice/repo.mlflow"
        assert cfg.tracking.backend == "dagshub"

    def test_no_dotenv_leaves_uri_untouched(self, tmp_path, monkeypatch):
        for var in ("DAGSHUB_TOKEN", "DAGSHUB_USERNAME", "DAGSHUB_REPO"):
            monkeypatch.delenv(var, raising=False)
        cfg = load_mlflow_config()
        original = cfg.tracking.tracking_uri
        monkeypatch.chdir(tmp_path)
        resolve_mlflow_tracking(cfg)
        assert cfg.tracking.tracking_uri == original


class TestLoadTrainConfig:
    def test_loads_default_config(self):
        cfg = load_train_config()
        assert cfg.seed == 42
        assert cfg.split.strategy == "temporal"
        assert cfg.model.name == "lightgbm"

    def test_cli_overrides_model_name(self):
        cfg = load_train_config(cli_args=["model.name=xgboost"])
        assert cfg.model.name == "xgboost"
        assert cfg.split.strategy == "temporal"  # unchanged

    def test_cli_overrides_nested_path(self):
        cfg = load_train_config(cli_args=["split.test_size=0.3"])
        assert cfg.split.test_size == 0.3
        assert cfg.model.name == "lightgbm"  # unchanged

    def test_cli_overrides_model_params(self):
        cfg = load_train_config(
            cli_args=["model.params.learning_rate=0.1", "model.params.max_depth=6"]
        )
        assert cfg.model.params.learning_rate == 0.1
        assert cfg.model.params.max_depth == 6

    def test_cli_overrides_optuna_enable(self):
        cfg = load_train_config(cli_args=["optuna.enabled=true"])
        assert cfg.optuna.enabled is True

    def test_cli_overrides_max_train_rows(self):
        cfg = load_train_config(cli_args=["data.max_train_rows=200000"])
        assert cfg.data.max_train_rows == 200000

    def test_cli_overrides_experiment_tag(self):
        cfg = load_train_config(cli_args=["mlflow.experiment_tag=A_baseline"])
        assert cfg.mlflow.experiment_tag == "A_baseline"

    def test_cli_handles_dash_args(self):
        """OmegaConf should ignore --file-style flags."""
        cfg = load_train_config(cli_args=["--help", "split.test_size=0.25"])
        assert cfg.split.test_size == 0.25

    def test_custom_config_path(self, tmp_path: pytest.TempPathFactory):
        custom = tmp_path / "custom.yaml"
        custom.write_text(
            OmegaConf.to_yaml(
                {
                    "seed": 123,
                    "split": {"strategy": "random", "test_size": 0.3},
                    "model": {"name": "xgboost", "params": {"max_depth": 10}},
                }
            )
        )
        cfg = load_train_config(str(custom))
        assert cfg.seed == 123
        assert cfg.split.strategy == "random"
        assert cfg.split.test_size == 0.3
        assert cfg.model.name == "xgboost"
        assert cfg.model.params.max_depth == 10

    def test_invalid_config_raises_validation_error(
        self, tmp_path: pytest.TempPathFactory
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text("seed: not_a_number\nmodel: {name: lightgbm, params: {}}")
        with pytest.raises(Exception):  # pydantic.ValidationError
            load_train_config(str(bad))

    def test_invalid_model_name_raises(self, tmp_path: pytest.TempPathFactory):
        bad = tmp_path / "bad_model.yaml"
        bad.write_text(
            OmegaConf.to_yaml({"model": {"name": "invalid_model", "params": {}}})
        )
        with pytest.raises(Exception):
            load_train_config(str(bad))
