"""Validate that every YAML config matches its Pydantic schema."""

from pathlib import Path

from omegaconf import OmegaConf

from fdml.schemas import DataConfig, FeaturesConfig, TrainConfig

_CONFIGS = Path("configs")


def _load_and_validate(yaml_name: str, model):
    path = _CONFIGS / yaml_name
    assert path.exists(), f"Missing config: {path}"
    raw = OmegaConf.load(path)
    data = OmegaConf.to_container(raw, resolve=True)
    return model.model_validate(data)


class TestFeaturesConfig:
    def test_loads_and_validates(self):
        cfg = _load_and_validate("features.yaml", FeaturesConfig)
        assert cfg.transaction.target == "isFraud"
        assert "TransactionAmt" in cfg.transaction.numerical
        assert "hour_of_day" in cfg.engineered.datetime
        assert cfg.engineered.device_os is True
        assert cfg.imputation.numerical == -999

    def test_id_features_count(self):
        cfg = _load_and_validate("features.yaml", FeaturesConfig)
        assert cfg.identity is not None
        assert len(cfg.identity.id_features) == 38
        assert cfg.identity.id_features[0] == "id_01"


class TestDataConfig:
    def test_loads_and_validates(self):
        cfg = _load_and_validate("data.yaml", DataConfig)
        assert cfg.join_key == "TransactionID"
        assert cfg.target == "isFraud"
        assert cfg.files.train_transaction.endswith(".csv")

    def test_validation_rules(self):
        cfg = _load_and_validate("data.yaml", DataConfig)
        assert cfg.validation.target_values == [0, 1]
        assert cfg.validation.transaction_id_unique is True


class TestTrainConfig:
    def test_loads_and_validates(self):
        cfg = _load_and_validate("train.yaml", TrainConfig)
        assert cfg.seed == 42
        assert cfg.model.name == "lightgbm"
        assert cfg.split.strategy == "temporal"

    def test_model_params(self):
        cfg = _load_and_validate("train.yaml", TrainConfig)
        params = cfg.model.params
        assert params.learning_rate == 0.05
        assert params.scale_pos_weight == 27.6
        assert params.random_state == 42

    def test_optuna_defaults(self):
        cfg = _load_and_validate("train.yaml", TrainConfig)
        assert cfg.optuna.enabled is False
        assert cfg.optuna.n_trials == 50
