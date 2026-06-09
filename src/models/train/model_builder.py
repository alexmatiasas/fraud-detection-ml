from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

_RANDOM_STATE = 42


class ModelBuilder(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def build(self, params: dict[str, Any]) -> Any: ...

    def format_params(self, params: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def cleanup_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}


class LGBMBuilder(ModelBuilder):
    def name(self) -> str:
        return "lightgbm"

    def build(self, params: dict[str, Any]) -> lgb.LGBMClassifier:
        cleaned = self.cleanup_params(params)
        return lgb.LGBMClassifier(
            **cleaned, n_jobs=-1, verbose=-1, random_state=_RANDOM_STATE
        )

    def format_params(self, params: dict[str, Any]) -> str:
        return (
            f"leaves={params.get('num_leaves', 127)} depth={params.get('max_depth', 8)} "
            f"lr={params.get('learning_rate', 0.05):.3f} "
            f"subsample={params.get('subsample', 0.8):.1f} "
            f"colsample={params.get('colsample_bytree', 0.8):.1f} "
            f"alpha={params.get('reg_alpha', 0.1):.4f} "
            f"lambda={params.get('reg_lambda', 1.0):.1f} "
            f"scale_pos={params.get('scale_pos_weight', 27.6):.1f}"
        )

    def cleanup_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in params.items() if k not in ("n_jobs", "random_state")}


class XGBoostBuilder(ModelBuilder):
    def name(self) -> str:
        return "xgboost"

    def build(self, params: dict[str, Any]) -> xgb.XGBClassifier:
        cleaned = self.cleanup_params(params)
        return xgb.XGBClassifier(
            **cleaned, verbosity=0, random_state=_RANDOM_STATE, n_jobs=-1
        )

    def format_params(self, params: dict[str, Any]) -> str:
        return (
            f"n_est={params.get('n_estimators', 1000)} depth={params.get('max_depth', 8)} "
            f"lr={params.get('learning_rate', 0.05):.3f} "
            f"subsample={params.get('subsample', 0.8):.1f} "
            f"colsample={params.get('colsample_bytree', 0.8):.1f} "
            f"scale_pos={params.get('scale_pos_weight', 27.6):.1f}"
        )

    def cleanup_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v
            for k, v in params.items()
            if k not in ("n_jobs", "random_state", "verbosity")
        }


class RFBuilder(ModelBuilder):
    def name(self) -> str:
        return "random_forest"

    def build(self, params: dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 1000),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_child_samples", 100),
            max_samples=params.get("subsample", 0.8),
            max_features=params.get("colsample_bytree", 0.8),
            class_weight="balanced_subsample",
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )

    def format_params(self, params: dict[str, Any]) -> str:
        return (
            f"n_est={params.get('n_estimators', 1000)} depth={params.get('max_depth', 8)} "
            f"min_samples_leaf={params.get('min_child_samples', 100)} "
            f"max_samples={params.get('subsample', 0.8):.1f}"
        )


class ModelBuilderRegistry:
    _builders: dict[str, ModelBuilder] = {}

    def register(self, builder: ModelBuilder) -> None:
        self._builders[builder.name()] = builder

    def get(self, name: str) -> ModelBuilder:
        if name not in self._builders:
            msg = f"Unknown model: {name}"
            raise ValueError(msg)
        return self._builders[name]

    def build(self, name: str, params: dict[str, Any]) -> Any:
        return self.get(name).build(params)


model_builder_registry = ModelBuilderRegistry()
model_builder_registry.register(LGBMBuilder())
model_builder_registry.register(XGBoostBuilder())
model_builder_registry.register(RFBuilder())
