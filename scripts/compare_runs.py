"""Compare MLflow runs grouped by the `experiment` tag (e.g. A_baseline, B_reg).

Usage:
    python scripts/compare_runs.py --tag A_baseline
    python scripts/compare_runs.py --tag B_reg --seed-bag
    python scripts/compare_runs.py --tag B_reg --seed-bag --dedupe --output models/compare_B_reg.csv

The tracking URI is resolved the same way as `make train` (DagsHub via .env
when present, otherwise local).

Concepts:
- **variant**: runs in a group are told apart by their *configuration* — the
  hyperparameters (and model name) that differ across the group. A run whose
  params match the group-wide value for every varying key is labelled
  ``baseline``. This lets one tag hold several variants (e.g. the B_reg
  regularization screen) without confusing them for repeated seeds.
- **dedupe**: keeps one run per (variant, seed) — useful when the same
  experiment was run twice (e.g. to check reproducibility).
- **aborted runs**: runs tagged ``status=aborted`` (killed with Ctrl-C /
  SIGTERM) are excluded unless ``--include-aborted`` is passed.
- **seed-bag**: averages the ``val_proba.npy`` artifacts per variant (all runs
  of a variant share the same validation split) and re-computes AUC / AP on the
  bagged probabilities. Only variants with >=2 distinct seeds are bagged.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fdml.models.config import load_mlflow_config, resolve_mlflow_tracking

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

METRICS = ["val/roc_auc", "val/average_precision", "val/f1_best", "val/best_iteration"]

# Params that identify a run's *experiment* rather than its *configuration*.
_NON_CONFIG_PARAMS = {
    "seed",
    "experiment",
    "random_state",
    "train_dt_min",
    "train_dt_max",
    "val_dt_min",
    "val_dt_max",
}


def _get_runs(client: MlflowClient, experiment: str, tag: str) -> list:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment '{experiment}' not found")
    return client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.experiment = '{tag}'",
        order_by=["attributes.start_time ASC"],
    )


def _model_name(run) -> str:
    return run.data.tags.get("model_name", run.data.params.get("model", ""))


def _varying_param_keys(runs) -> list[str]:
    """Hyperparameter/model keys whose value differs across complete runs."""
    complete = [r for r in runs if "val/roc_auc" in r.data.metrics]
    keys: set[str] = set()
    model_names = {_model_name(r) for r in complete}
    if len(model_names) > 1:
        keys.add("model_name")
    for r in complete:
        for k, v in r.data.params.items():
            if k in _NON_CONFIG_PARAMS:
                continue
            if any(other.data.params.get(k) != v for other in complete):
                keys.add(k)
    return sorted(keys)


def _variant(run, varying_keys: list[str]) -> str:
    parts = []
    for k in varying_keys:
        v = _model_name(run) if k == "model_name" else run.data.params.get(k)
        if v is not None and str(v) != "":
            parts.append(f"{k}={v}")
    return ",".join(parts) if parts else "baseline"


def _run_row(run, varying_keys: list[str]) -> dict:
    m, p, t = run.data.metrics, run.data.params, run.data.tags
    best_iter = m.get("val/best_iteration")
    return {
        "run_name": p.get("run_name", run.info.run_name),
        "variant": _variant(run, varying_keys),
        "seed": p.get("seed", ""),
        "n_train": t.get("n_train", ""),
        "n_val": t.get("n_val", ""),
        "val/roc_auc": round(m.get("val/roc_auc", float("nan")), 5),
        "val/average_precision": round(m.get("val/average_precision", float("nan")), 5),
        "val/f1_best": round(m.get("val/f1_best", float("nan")), 5),
        "val/best_iteration": (
            float(best_iter) if best_iter is not None else float("nan")
        ),
        "training_elapsed_s": round(m.get("training_elapsed_s", float("nan")), 1),
    }


def _print_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))


def _summary(rows: list[dict]) -> None:
    complete = [r for r in rows if np.isfinite(r["val/roc_auc"])]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in complete:
        by_variant[r["variant"]].append(r)
    for variant in sorted(by_variant):
        group = by_variant[variant]
        print(f"\n[{variant}]  mean +/- std, N={len(group)}")
        for metric in METRICS:
            vals = [r[metric] for r in group if np.isfinite(r[metric])]
            if vals:
                print(f"  {metric:<24} {np.mean(vals):.5f} +/- {np.std(vals):.5f}")


def _seed_bag(
    client: MlflowClient, pairs: list[tuple[dict, object]], output: Path | None
) -> None:
    """Average val_proba.npy per variant; report bagged AUC/AP for N>=2 seeds."""
    by_variant: dict[str, list[tuple[dict, object]]] = defaultdict(list)
    for row, run in pairs:
        if np.isfinite(row["val/roc_auc"]) and row["seed"]:
            by_variant[row["variant"]].append((row, run))

    lines: list[str] = []
    any_reported = False
    for variant in sorted(by_variant):
        pairs = by_variant[variant]
        seen_seeds: set[str] = set()
        seeds: set[str] = set()
        bagged = None
        y_val = None
        used = 0
        for row, run in pairs:
            if row["seed"] in seen_seeds:
                continue
            seen_seeds.add(row["seed"])
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    proba_path = client.download_artifacts(
                        run.info.run_id, path="val_proba.npy", dst_path=tmp
                    )
                    yval_path = client.download_artifacts(
                        run.info.run_id, path="y_val.npy", dst_path=tmp
                    )
                except Exception:
                    continue
                proba = np.load(proba_path)
                labels = np.load(yval_path)
                if y_val is None:
                    y_val = labels
                elif not np.array_equal(y_val, labels):
                    logger.warning(
                        "  [%s] run %s has a different y_val — skipping bag "
                        "(mixed splits?)",
                        variant,
                        run.info.run_name,
                    )
                    break
                bagged = proba if bagged is None else bagged + proba
                seeds.add(row["seed"])
                used += 1
        if bagged is None or y_val is None or len(seeds) < 2:
            continue
        bagged /= used
        auc = roc_auc_score(y_val, bagged)
        ap = average_precision_score(y_val, bagged)
        any_reported = True
        msg = (
            f"Seed-bagged [{variant}] (N={used} runs, seeds={sorted(seeds)}): "
            f"val/roc_auc {auc:.5f}  val/average_precision {ap:.5f}"
        )
        print(f"\n{msg}")
        lines.append(f"{variant},{auc:.5f},{ap:.5f}")

    if not any_reported:
        print("\nSeed-bag: no variant with >=2 distinct seeds and val_proba artifacts")
    if output is not None and lines:
        output.write_text(
            "variant,seed_bagged_auc,seed_bagged_ap\n" + "\n".join(lines) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True, help="experiment tag value, e.g. A_baseline"
    )
    parser.add_argument(
        "--experiment", help="experiment name (default: from configs/mlflow.yaml)"
    )
    parser.add_argument(
        "--tracking-uri",
        help="override tracking URI (default: from configs/mlflow.yaml)",
    )
    parser.add_argument("--output", help="optional CSV path for the run table")
    parser.add_argument(
        "--seed-bag",
        action="store_true",
        help="average val_proba.npy per variant and report bagged AUC/AP",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="keep one run per (variant, seed); drop exact reruns",
    )
    parser.add_argument(
        "--include-aborted",
        action="store_true",
        help="include runs tagged status=aborted (excluded by default)",
    )
    args = parser.parse_args()

    mlflow_cfg = resolve_mlflow_tracking(load_mlflow_config())
    experiment = args.experiment or mlflow_cfg.tracking.experiment_name
    tracking_uri = args.tracking_uri or mlflow_cfg.tracking.tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    runs = _get_runs(client, experiment, args.tag)
    if not runs:
        raise SystemExit(f"No runs with tag 'experiment={args.tag}' in '{experiment}'")

    if not args.include_aborted:
        runs = [r for r in runs if r.data.tags.get("status") != "aborted"]

    varying_keys = _varying_param_keys(runs)
    pairs: list[tuple[dict, object]] = [(_run_row(r, varying_keys), r) for r in runs]
    if args.dedupe:
        seen: set[tuple[str, str]] = set()
        # keep complete runs over killed/aborted ones sharing (variant, seed)
        pairs.sort(key=lambda p: "val/roc_auc" in p[1].data.metrics, reverse=True)
        pairs = [
            pair
            for pair in pairs
            if (pair[0]["variant"], pair[0]["seed"]) not in seen
            and not seen.add((pair[0]["variant"], pair[0]["seed"]))
        ]
    rows = [row for row, _ in pairs]

    print(f"{len(rows)} run(s) with tag 'experiment={args.tag}'")
    print(f"varying keys: {', '.join(varying_keys) or 'none (single config)'}\n")
    _print_table(rows)
    _summary(rows)

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nTable saved to {args.output}")

    if args.seed_bag:
        out = Path(args.output).with_suffix(".seed_bag.csv") if args.output else None
        _seed_bag(client, pairs, out)


if __name__ == "__main__":
    main()
