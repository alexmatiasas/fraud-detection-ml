"""Paired seed-level comparison of two experiment tags, with learning curves.

Answers "is variant X really better than the control, or is it noise?" by
pairing runs on the same seed and testing the per-seed deltas, then showing
*where* in training the gap appears (per-iteration val AUC on the shared early
stopping eval set).

Usage:
    python scripts/analyze_runs.py \\
        --control-tag A_baseline_v3 --candidate-tag B_reg \\
        --candidate-variant baseline \\
        --plot reports/learning_curves_B1_vs_baseline.png

The candidate variant defaults to the most common one in the tag (the
``baseline`` mode of compare_runs). Tracking is resolved like `make train`
(DagsHub via .env when present, otherwise local).
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from compare_runs import (  # noqa: E402
    _baseline_values,
    _get_runs,
    _run_row,
    _shorten,
    _varying_param_keys,
)
from fdml.models.config import load_mlflow_config, resolve_mlflow_tracking

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _complete_rows(runs, dedupe: bool) -> list[dict]:
    varying_keys = _varying_param_keys(runs)
    baseline = _baseline_values(
        [r for r in runs if "val/roc_auc" in r.data.metrics] or runs, varying_keys
    )
    rows = [_run_row(r, varying_keys, baseline) for r in runs]
    rows = [r for r in rows if np.isfinite(r["val/roc_auc"])]
    if dedupe:
        seen: set[tuple[str, str]] = set()
        rows = [
            r
            for r in rows
            if (r["variant"], r["seed"]) not in seen
            and not seen.add((r["variant"], r["seed"]))
        ]
    return rows


def _pick_variant(rows: list[dict], override: str | None) -> str:
    if override:
        return override
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["variant"]] += 1
    return max(
        counts,
        key=lambda v: (
            counts[v],
            sum(r["val/roc_auc"] for r in rows if r["variant"] == v),
        ),
    )


def _pair_by_seed(
    control: list[dict], candidate: list[dict]
) -> list[tuple[dict, dict]]:
    ctrl = {r["seed"]: r for r in control}
    cand = {r["seed"]: r for r in candidate}
    common = sorted(set(ctrl) & set(cand), key=int)
    if not common:
        raise SystemExit("No overlapping seeds between control and candidate")
    return [(ctrl[s], cand[s]) for s in common]


def _bagged(client: MlflowClient, rows: list[dict]) -> tuple[float, float] | None:
    proba_sum = None
    y_val = None
    used = 0
    for r in rows:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                proba_path = client.download_artifacts(
                    r["run_id"], "val_proba.npy", tmp
                )
                y_path = client.download_artifacts(r["run_id"], "y_val.npy", tmp)
            except Exception:
                continue
            proba = np.load(proba_path)
            labels = np.load(y_path)
            if y_val is None:
                y_val = labels
            elif not np.array_equal(y_val, labels):
                logger.warning("  mixed y_val across seeds — skipping bag")
                return None
            proba_sum = proba if proba_sum is None else proba_sum + proba
            used += 1
    if proba_sum is None or y_val is None or used < 2:
        return None
    proba = proba_sum / used
    return roc_auc_score(y_val, proba), average_precision_score(y_val, proba)


def _curves(
    client: MlflowClient, rows: list[dict]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for r in rows:
        hist = client.get_metric_history(r["run_id"], "val/auc")
        if not hist:
            continue
        iters = np.array([m.step for m in hist])
        vals = np.array([m.value for m in hist])
        out[r["seed"]] = (iters, vals)
    return out


def _mean_curve(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    grid_values = np.full((len(curves), len(grid)), np.nan)
    for i, (_seed, (iters, vals)) in enumerate(curves.items()):
        grid_values[i, np.searchsorted(grid, iters)] = vals
    mean = np.nanmean(grid_values, axis=0)
    std = np.where(
        np.sum(~np.isnan(grid_values), axis=0) >= 2,
        np.nanstd(grid_values, axis=0, ddof=1),
        np.nan,
    )
    n = np.sum(~np.isnan(grid_values), axis=0)
    return mean, std, max(n, default=0)


def _plot_learning_curves(
    client: MlflowClient,
    ctrl_rows: list[dict],
    cand_rows: list[dict],
    control_label: str,
    candidate_label: str,
    out_path: Path,
) -> None:
    ctrl = _curves(client, ctrl_rows)
    cand = _curves(client, cand_rows)
    if not ctrl or not cand:
        logger.warning("  no per-iteration val/auc curves available — skipping plot")
        return
    grid = np.arange(10, 201, 10)
    max_iter = max(
        *(v[0].max() for v in ctrl.values()),
        *(v[0].max() for v in cand.values()),
    )
    grid = grid[grid <= max_iter]
    c_mean, c_std, _ = _mean_curve(ctrl, grid)
    k_mean, k_std, k_n = _mean_curve(cand, grid)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(
        grid,
        c_mean - c_std,
        c_mean + c_std,
        alpha=0.15,
        label=f"{control_label} ± 1σ (N={len(ctrl)})",
    )
    ax.fill_between(
        grid,
        k_mean - k_std,
        k_mean + k_std,
        alpha=0.15,
        label=f"{candidate_label} ± 1σ (N={len(cand)})",
    )
    ax.plot(grid, c_mean, label=control_label)
    ax.plot(grid, k_mean, label=candidate_label)

    ctrl_iter = np.mean(
        [
            r["val/best_iteration"]
            for r in ctrl_rows
            if np.isfinite(r["val/best_iteration"])
        ]
    )
    cand_iter = np.mean(
        [
            r["val/best_iteration"]
            for r in cand_rows
            if np.isfinite(r["val/best_iteration"])
        ]
    )
    ax.axvline(ctrl_iter, color="C0", ls="--", lw=1, alpha=0.5)
    ax.axvline(cand_iter, color="C1", ls="--", lw=1, alpha=0.5)

    ax.set_xlabel("iteration (ES eval set, 20k rows, mlflow_every=10)")
    ax.set_ylabel("val AUC")
    ax.set_title("Learning curves — early-stopping eval set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-tag",
        required=True,
        help="control experiment tag (e.g. A_baseline_v3)",
    )
    parser.add_argument(
        "--candidate-tag", required=True, help="candidate experiment tag (e.g. B_reg)"
    )
    parser.add_argument(
        "--control-variant", default=None, help="control variant (default: most common)"
    )
    parser.add_argument(
        "--candidate-variant",
        default=None,
        help="candidate variant (default: most common)",
    )
    parser.add_argument(
        "--plot",
        default="reports/learning_curves.png",
        help="path for the learning-curve plot",
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="keep one run per (variant, seed)"
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    mlflow_cfg = resolve_mlflow_tracking(load_mlflow_config())
    experiment = args.experiment or mlflow_cfg.tracking.experiment_name
    tracking_uri = args.tracking_uri or mlflow_cfg.tracking.tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    control = _complete_rows(
        _get_runs(client, experiment, args.control_tag), args.dedupe
    )
    candidate = _complete_rows(
        _get_runs(client, experiment, args.candidate_tag), args.dedupe
    )
    if not control or not candidate:
        raise SystemExit("No complete runs found for one of the tags")

    ctrl_var = _pick_variant(control, args.control_variant)
    cand_var = _pick_variant(candidate, args.candidate_variant)
    control_rows = [r for r in control if r["variant"] == ctrl_var]
    cand_rows = [r for r in candidate if r["variant"] == cand_var]
    ctrl_label = f"{args.control_tag}:{_shorten(ctrl_var)}"
    cand_label = f"{args.candidate_tag}:{_shorten(cand_var)}"

    pairs = _pair_by_seed(control_rows, cand_rows)
    deltas = [(c, k, k["val/roc_auc"] - c["val/roc_auc"]) for c, k in pairs]

    print(f"\nControl   : {ctrl_label}  (N={len(control_rows)})")
    print(f"Candidate : {cand_label}  (N={len(cand_rows)})")
    print(f"Paired seeds: {', '.join(c['seed'] for c, _, _ in deltas)}\n")

    print("  seed   control_auc   cand_auc      delta")
    for c, k, d in deltas:
        print(
            f"  {c['seed']:>4}   {c['val/roc_auc']:.5f}   {k['val/roc_auc']:.5f}   {d:+.5f}"
        )

    deltas_auc = np.array([d for _, _, d in deltas])
    n = len(deltas_auc)
    mean_d = deltas_auc.mean()
    std_d = deltas_auc.std(ddof=1)
    t_stat, p_two = stats.ttest_rel(
        np.array([k["val/roc_auc"] for _, k, _ in deltas]),
        np.array([c["val/roc_auc"] for c, _, _ in deltas]),
    )
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    pos = int(np.sum(deltas_auc > 0))
    sign_p = stats.binomtest(pos, n, alternative="greater").pvalue

    print("\nPaired deltas (candidate − control, by seed):")
    print(f"  mean Δ = {mean_d:+.5f}  std = {std_d:.5f}  SE = {std_d / np.sqrt(n):.5f}")
    print(
        f"  paired t({n - 1}) = {t_stat:.2f}   p(1-sided) = {p_one:.4f}  p(2-sided) = {p_two:.4f}"
    )
    print(f"  sign test: {pos}/{n} seeds improved → p(1-sided) = {sign_p:.4f}")
    print(
        f"  noise floor σ (control across seeds) = {np.std([c['val/roc_auc'] for c, _, _ in deltas], ddof=1):.5f}"
    )
    print(
        f"  mean Δ in σ units = {mean_d / np.std([c['val/roc_auc'] for c, _, _ in deltas], ddof=1):+.2f}σ"
    )

    ctrl_bag = _bagged(client, control_rows)
    cand_bag = _bagged(client, cand_rows)
    if ctrl_bag and cand_bag:
        print("\nSeed-bagged (mean of val_proba, N>=2):")
        print(f"  {ctrl_label:34s} AUC={ctrl_bag[0]:.5f}  AP={ctrl_bag[1]:.5f}")
        print(f"  {cand_label:34s} AUC={cand_bag[0]:.5f}  AP={cand_bag[1]:.5f}")
        print(f"  Δ AUC = {cand_bag[0] - ctrl_bag[0]:+.5f}")

    _plot_learning_curves(
        client, control_rows, cand_rows, ctrl_label, cand_label, Path(args.plot)
    )


if __name__ == "__main__":
    main()
