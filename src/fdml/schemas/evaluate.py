from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


PlotName = Literal[
    "roc_curve",
    "pr_curve",
    "calibration",
    "error_analysis",
    "learning_curves",
    "feature_importance_stability",
    "adversarial_validation",
]

EvalMetricName = Literal[
    "roc_auc", "average_precision", "f1_score", "precision", "recall"
]

BootstrapMetric = Literal["roc_auc", "average_precision"]


class ThresholdCfg(BaseModel):
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Default decision threshold"
    )
    n_thresholds: int = Field(
        default=100, ge=10, le=1000, description="Number of thresholds for tuning"
    )
    bootstrap: bool = Field(
        default=True, description="Compute bootstrap confidence intervals"
    )
    bootstrap_metric: BootstrapMetric = Field(
        default="average_precision", description="Metric for bootstrap CI"
    )
    bootstrap_iterations: int = Field(
        default=1000, ge=50, le=10000, description="Bootstrap iterations"
    )
    bootstrap_seed: int = Field(
        default=42, description="Random seed for bootstrap resampling"
    )


class SegmentsCfg(BaseModel):
    enabled: bool = Field(default=True, description="Enable per-segment analysis")
    columns: list[str] = Field(
        default_factory=lambda: ["ProductCD", "card4"],
        description="Columns for per-segment analysis",
    )
    min_samples: int = Field(
        default=50, ge=10, description="Minimum samples per segment"
    )


class LearningCurvesCfg(BaseModel):
    enabled: bool = Field(
        default=False, description="Enable learning curves (re-trains model)"
    )
    train_sizes: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        description="Training set size fractions",
    )


class StabilityCfg(BaseModel):
    feature_importance: bool = Field(
        default=False,
        description="Enable feature importance stability analysis (re-trains model N times)",
    )
    n_iterations: int = Field(
        default=10, ge=3, le=100, description="Number of re-fits for stability"
    )
    seed: int = Field(default=42, description="Random seed for stability refits")


class ErrorAnalysisCfg(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable error distribution analysis"
    )
    features: list[str] = Field(
        default_factory=lambda: ["TransactionAmt"],
        description="Features to analyse for error distribution",
    )


class AdversarialValCfg(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable adversarial validation (trains RF to detect train/val drift)",
    )
    seed: int = Field(
        default=42, description="Random seed for adversarial validation RF"
    )


class ModelCardCfg(BaseModel):
    enabled: bool = Field(default=True, description="Generate model card markdown")
    filename: str = Field(
        default="model_card.md",
        description="Output filename for model card (in plot output dir)",
    )


class PlotCfg(BaseModel):
    output_dir: str = Field(default="models/", description="Directory for plot images")
    plots: list[PlotName] = Field(
        default_factory=lambda: ["roc_curve", "pr_curve", "calibration"],
        description="Plots to generate",
    )
    error_features: list[str] | None = Field(
        default=None,
        description="Features for error analysis histograms (None = auto)",
    )


class DVCLiveCfg(BaseModel):
    enabled: bool = Field(
        default=True, description="Enable DVCLive logging for DVC metrics/plots"
    )
    dir: str = Field(default="dvclive", description="DVCLive output directory")


class ReportCfg(BaseModel):
    path: str = Field(
        default="models/report.json",
        description="Path for JSON evaluation report",
    )


class EvaluateConfig(BaseModel):
    threshold: ThresholdCfg = Field(
        default_factory=ThresholdCfg, description="Threshold tuning configuration"
    )
    segments: SegmentsCfg = Field(
        default_factory=SegmentsCfg, description="Per-segment analysis configuration"
    )
    learning_curves: LearningCurvesCfg = Field(
        default_factory=LearningCurvesCfg,
        description="Learning curves configuration",
    )
    stability: StabilityCfg = Field(
        default_factory=StabilityCfg,
        description="Feature importance stability configuration",
    )
    error_analysis: ErrorAnalysisCfg = Field(
        default_factory=ErrorAnalysisCfg, description="Error analysis configuration"
    )
    adversarial_validation: AdversarialValCfg = Field(
        default_factory=AdversarialValCfg,
        description="Adversarial validation configuration",
    )
    model_card: ModelCardCfg = Field(
        default_factory=ModelCardCfg, description="Model card configuration"
    )
    dvclive: DVCLiveCfg = Field(
        default_factory=DVCLiveCfg, description="DVCLive logging configuration"
    )
    plots: PlotCfg = Field(default_factory=PlotCfg, description="Plot configuration")
    report: ReportCfg = Field(
        default_factory=ReportCfg, description="Report output configuration"
    )
