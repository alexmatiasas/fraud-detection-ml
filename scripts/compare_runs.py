"""Compare MLflow runs grouped by the `experiment` tag (e.g. A_baseline, B_reg).

Usage:
    python scripts/compare_runs.py --tag A_baseline
    python scripts/compare_runs.py --tag A_baseline --seed-bag
    python scripts/compare_runs.py --tag B_reg --output models/compare_B_reg.csv

The tracking URI is resolved the same way as `make train` (DagsHub via .env
when present, otherwise local). Each row shows the run metrics, and the summary
section reports mean +/- std across the group. `--seed-bag` additionally
averages the val_proba.npy artifacts (all runs share the same validation split)
and re-computes AUC / AP on the bagged probabilities.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

from fdml.models.config import load_mlflow_config, resolve_mlflow_tracking

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

METRICS = ["val/roc_auc", "val/average_precision", "val/f1_best", "val/best_iteration"]


def _get_runs(client: MlflowClient, experiment: str, tag: str) -> list:
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experiment '{experiment}' not found")
    return client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.experiment = '{tag}'",
        order_by=["attributes.start_time ASC"],
    )


def _run_row(run) -> dict:
    m, p, t = run.data.metrics, run.data.params, run.data.tags
    return {
        "run_name": p.get("run_name", run.info.run_name),
        "seed": p.get("seed", ""),
        "n_train": t.get("n_train", ""),
        "n_val": t.get("n_val", ""),
        "val/roc_auc": round(m.get("val/roc_auc", float("nan")), 5),
        "val/average_precision": round(m.get("val/average_precision", float("nan")), 5),
        "val/f1_best": round(m.get("val/f1_best", float("nan")), 5),
        "val/best_iteration": int(m.get("val/best_iteration", 0)),
        "training_elapsed_s": round(m.get("training_elapsed_s", float("nan")), 1),
    }


def _print_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c in rows[0]]
    with pd.option_context("display.width", 160):
        print(df[cols].to_string(index=False))


def _summary(rows: list[dict]) -> None:
    print("\nSummary (mean +/- std, N=%d):" % len(rows))
    for metric in METRICS:
        vals = [r[metric] for r in rows if np.isfinite(r[metric])]
        if vals:
            print(f"  {metric:<24} {np.mean(vals):.5f} +/- {np.std(vals):.5f}")


def _seed_bag(client: MlflowClient, runs, output: Path | None) -> None:
    bagged = None
    y_val = None
    seeds: set[str] = set()
    used = 0
    for run in runs:
        seed = run.data.params.get("seed", "")
        seeds.add(seed)
        if not seed:
            continue
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
                    "  Run %s has a different y_val — skipping bag (mixed splits?)",
                    run.info.run_name,
                )
                return
            bagged = proba if bagged is None else bagged + proba
            used += 1
    if bagged is None or y_val is None:
        print("\nSeed-bag: no val_proba.npy artifacts found")
        return
    if len(seeds) < 2:
        print(f"\nSeed-bag: only {len(seeds)} distinct seed(s) — need >=2")
        return
    bagged /= used
    auc = roc_auc_score(y_val, bagged)
    ap = average_precision_score(y_val, bagged)
    print(
        f"\nSeed-bagged (N={used} runs, distinct seeds={sorted(seeds)}): "
        f"val/roc_auc {auc:.5f}  val/average_precision {ap:.5f}"
    )
    if output is not None:
        output.write_text(f"seed_bagged_auc,{auc:.5f}\nseed_bagged_ap,{ap:.5f}\n")


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
        help="average val_proba.npy across runs and report bagged AUC/AP",
    )
    args = parser.parse_args()

    mlflow_cfg = resolve_mlflow_tracking(load_mlflow_config())
    experiment = args.experiment or mlflow_cfg.tracking.experiment_name
    tracking_uri = args.tracking_uri or mlflow_cfg.tracking.tracking_uri
    client = MlflowClient(tracking_uri=tracking_uri)

    runs = _get_runs(client, experiment, args.tag)
    if not runs:
        raise SystemExit(f"No runs with tag 'experiment={args.tag}' in '{experiment}'")

    rows = [_run_row(r) for r in runs]
    print(f"{len(rows)} run(s) with tag 'experiment={args.tag}'\n")
    _print_table(rows)
    _summary(rows)

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nTable saved to {args.output}")

    if args.seed_bag:
        out = Path(args.output).with_suffix(".seed_bag.csv") if args.output else None
        _seed_bag(client, runs, out)


if __name__ == "__main__":
    main()
