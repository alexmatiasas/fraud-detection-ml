from __future__ import annotations


def generate_model_card(
    model_name: str,
    metrics: dict[str, float],
    ci: tuple[float, float] | None,
    best_threshold: float,
    best_f1: float,
    n_features: int,
    n_train: int,
    n_val: int,
    fraud_rate: float,
    split_strategy: str,
    feature_importance: list[tuple[str, float]] | None = None,
    auc_adv: float | None = None,
    brier: float | None = None,
    f_beta: float | None = None,
    cost_best_threshold: float | None = None,
    expected_cost: float | None = None,
    recall_at_k: dict[str, float] | None = None,
) -> str:
    lines = [
        "# Model Card",
        "",
        f"**Model**: {model_name}",
        "**Dataset**: IEEE-CIS Fraud Detection (Kaggle 2019)",
        f"**Rows**: {n_train:,} train / {n_val:,} validation",
        f"**Fraud rate**: {fraud_rate:.2%}",
        f"**Split strategy**: {split_strategy}",
        f"**Features**: {n_features}",
        "",
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ROC AUC | {metrics['roc_auc']:.4f} |",
        f"| Average Precision | {metrics['average_precision']:.4f} |",
    ]

    if ci is not None:
        lines.append(f"| AP 95% CI | [{ci[0]:.4f}, {ci[1]:.4f}] |")

    lines += [
        f"| F1 (optimal threshold={best_threshold:.2f}) | {best_f1:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
    ]

    if f_beta is not None:
        lines.append(f"| F2 (default threshold) | {f_beta:.4f} |")
    if brier is not None:
        lines.append(f"| Brier | {brier:.4f} |")
    if expected_cost is not None and cost_best_threshold is not None:
        lines.append(
            f"| Expected cost/transaction | {expected_cost:.4f} @ thr={cost_best_threshold:.2f} |"
        )
    if recall_at_k:
        for k, v in sorted(recall_at_k.items(), key=lambda kv: float(kv[0])):
            lines.append(f"| Recall@top {float(k):.0%} | {v:.4f} |")

    if auc_adv is not None:
        drift_text = (
            "possible drift detected" if auc_adv > 0.8 else "no significant drift"
        )
        lines += [
            "",
            "## Adversarial validation",
            "",
            f"AUC={auc_adv:.3f} — {drift_text}",
        ]

    if feature_importance:
        lines += [
            "",
            "## Top-10 features (split gain)",
            "",
            "| Rank | Feature | Importance |",
            "|---|---|---|",
        ]
        for i, (col, imp) in enumerate(feature_importance[:10]):
            lines.append(f"| {i + 1} | {col} | {imp:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Average Precision (PR AUC) is the primary metric due to 3.5% fraud rate",
        "- Expected cost uses FN = 10×FP (fraud lost vs. manual review)",
        "- Threshold tuned to maximize F1 on validation set",
        "- See `configs/evaluate.yaml` for evaluation settings",
        "",
    ]
    return "\n".join(lines)
