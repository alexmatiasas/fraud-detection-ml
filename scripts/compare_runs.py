"""Compare MLflow runs grouped by the `experiment` tag (e.g. A_baseline, B_reg).

Usage:
    python scripts/compare_runs.py --tag A_baseline
    python scripts/compare_runs.py --tag B_reg --seed-bag
    python scripts/compare_runs.py --tag B_reg --seed-bag --dedupe --output models/compare_B_reg.csv

The tracking URI is resolved the same way as `make train` (DagsHub via .env
when present, otherwise local). When stdout is a terminal the results open in
an interactive, scrollable table (Textual DataTable — arrows to move, ``a`` to
sort by AUC, ``s`` to toggle runs/summary, ``q`` to quit); pass ``--plain`` to
force the non-interactive rich tables, or ``--interactive`` to force the TUI.

Concepts:
- **variant**: runs in a group are told apart by their *configuration* — the
  hyperparameters (and model name) that differ across the group. A run whose
  params match the most common value for every varying key is labelled
  ``baseline``; every other variant is shown compactly as the keys where it
  differs from that baseline (long values like ``features_hash`` are
  truncated in the table; the full value is kept in the CSV). This lets one
  tag hold several variants (e.g. the B_reg regularization screen) without
  confusing them for repeated seeds.
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
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import MlflowClient
from rich.console import Console
from rich.table import Table
from sklearn.metrics import average_precision_score, roc_auc_score

from fdml.models.config import load_mlflow_config, resolve_mlflow_tracking

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Download artifacts silently (no tqdm "Downloading artifacts" bars).
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

console = Console(width=160)

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


def _baseline_values(complete_runs, varying_keys: list[str]) -> dict[str, str]:
    """Most common value per varying key — used to label variants compactly.

    A variant is shown as the set of keys where it *differs* from the baseline
    (the most common config in the group), so ``baseline`` means "all defaults".
    """
    baseline: dict[str, str] = {}
    for k in varying_keys:
        if k == "model_name":
            values = [_model_name(r) for r in complete_runs]
        else:
            values = [r.data.params.get(k, "") for r in complete_runs]
        baseline[k] = Counter(values).most_common(1)[0][0]
    return baseline


def _shorten(value: str, limit: int = 16) -> str:
    """Shorten long values (e.g. the 40-char ``features_hash``) for display."""
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _variant_parts(run, varying_keys: list[str], baseline: dict[str, str]) -> list:
    parts = []
    for k in varying_keys:
        v = _model_name(run) if k == "model_name" else run.data.params.get(k, "")
        if v != baseline[k]:
            parts.append((k, v))
    return parts


def _variant(run, varying_keys: list[str], baseline: dict[str, str]) -> str:
    """Full variant key — unique per config, used for grouping."""
    parts = _variant_parts(run, varying_keys, baseline)
    return ",".join(f"{k}={v}" for k, v in parts) if parts else "baseline"


def _variant_label(run, varying_keys: list[str], baseline: dict[str, str]) -> str:
    """Short human-readable variant for the table (hashes truncated)."""
    parts = _variant_parts(run, varying_keys, baseline)
    return ",".join(f"{k}={_shorten(v)}" for k, v in parts) if parts else "baseline"


def _run_row(run, varying_keys: list[str], baseline: dict[str, str]) -> dict:
    m, p, t = run.data.metrics, run.data.params, run.data.tags
    best_iter = m.get("val/best_iteration")
    return {
        "run_id": run.info.run_id,
        "run_name": p.get("run_name", run.info.run_name),
        "variant": _variant(run, varying_keys, baseline),
        "variant_label": _variant_label(run, varying_keys, baseline),
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
    table = Table(
        header_style="bold",
        show_lines=False,
        pad_edge=False,
        title=f"runs — {len(rows)} total",
    )
    table.add_column("run", no_wrap=True)
    table.add_column("variant", no_wrap=True)
    table.add_column("seed", justify="right")
    table.add_column("auc", justify="right")
    table.add_column("ap", justify="right")
    table.add_column("f1", justify="right")
    table.add_column("iter", justify="right")
    table.add_column("secs", justify="right")

    def _fmt(value: float, ndigits: int) -> str:
        return f"{value:.{ndigits}f}" if np.isfinite(value) else "—"

    for r in rows:
        table.add_row(
            r["run_name"],
            r["variant_label"],
            str(r["seed"]),
            _fmt(r["val/roc_auc"], 5),
            _fmt(r["val/average_precision"], 4),
            _fmt(r["val/f1_best"], 4),
            _fmt(r["val/best_iteration"], 0),
            _fmt(r["training_elapsed_s"], 1),
        )
    console.print(table)


def _summary_stats(rows: list[dict]) -> list[dict]:
    """Per-variant mean ± std stats, sorted by mean AUC desc."""
    complete = [r for r in rows if np.isfinite(r["val/roc_auc"])]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in complete:
        by_variant[r["variant"]].append(r)
    labels = {
        variant: by_variant[variant][0]["variant_label"] for variant in by_variant
    }

    stats = []
    for variant, group in by_variant.items():
        mean_auc = np.mean([r["val/roc_auc"] for r in group])
        stats.append(
            {
                "variant": variant,
                "label": labels[variant],
                "n": len(group),
                "mean_auc": mean_auc,
            }
        )
    stats.sort(key=lambda s: s["mean_auc"], reverse=True)

    best = next((s["variant"] for s in stats if s["n"] >= 2), None)
    for s in stats:
        s["is_best"] = s["variant"] == best
        group = by_variant[s["variant"]]

        def _fmt(metric: str, ndigits: int) -> str:
            vals = [r[metric] for r in group if np.isfinite(r[metric])]
            if not vals:
                return "—"
            return f"{np.mean(vals):.{ndigits}f} ± {np.std(vals):.{ndigits}f}"

        s["roc_auc"] = _fmt("val/roc_auc", 5)
        s["average_precision"] = _fmt("val/average_precision", 4)
        s["f1_best"] = _fmt("val/f1_best", 4)
        s["best_iteration"] = _fmt("val/best_iteration", 0)
    return stats


def _summary(rows: list[dict]) -> None:
    stats = _summary_stats(rows)
    table = Table(
        header_style="bold",
        show_lines=False,
        pad_edge=False,
        title="summary — mean ± std per variant",
    )
    table.add_column("variant", no_wrap=True)
    table.add_column("N", justify="right")
    for metric in METRICS:
        table.add_column(metric.removeprefix("val/"), justify="right")

    for s in stats:
        marker = "*" if s["is_best"] else ""
        table.add_row(
            f"{s['label'].replace(',', ', ')} {marker}".rstrip(),
            str(s["n"]),
            s["roc_auc"],
            s["average_precision"],
            s["f1_best"],
            s["best_iteration"],
        )
    console.print(table)
    best = next((s for s in stats if s["is_best"]), None)
    if best is not None:
        console.print(f"  * best candidate among N≥2 variants ({best['label']})")


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


class CompareApp:
    """Interactive, scrollable table browser (Textual DataTable).

    Arrows navigate (up/down/left/right scrolls columns), ``a`` sorts by AUC,
    ``s`` toggles between the runs and the per-variant summary, ``q`` quits.
    """

    def __init__(self, tag: str, run_rows: list[dict], summary_stats: list[dict]):
        from textual.app import App, Binding, ComposeResult
        from textual.widgets import DataTable, Footer, Header

        class _App(App[None]):
            CSS = """
            DataTable { height: 1fr; }
            """
            BINDINGS = [
                Binding("q", "quit", "Salir"),
                Binding("a", "sort_auc", "Sort AUC"),
                Binding("s", "toggle_view", "Runs/Summary"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.title = "compare_runs"
                self.show_runs = True

            def compose(self) -> ComposeResult:
                yield Header(show_clock=False)
                yield DataTable(id="table", cursor_type="row")
                yield Footer()

            def on_mount(self) -> None:
                self._show_runs()

            def _rebuild(
                self,
                title: str,
                columns: list[tuple[str, str]],
                rows: list[list],
            ) -> None:
                table = self.query_one(DataTable)
                table.clear(columns=True)
                for label, key in columns:
                    table.add_column(label, key=key)
                for cells in rows:
                    table.add_row(*cells)
                self.sub_title = title

            def _show_runs(self) -> None:
                self.show_runs = True
                fmt = lambda v, n: f"{v:.{n}f}" if np.isfinite(v) else "—"  # noqa: E731
                columns = [
                    ("run", "run"),
                    ("variant", "variant"),
                    ("seed", "seed"),
                    ("auc", "auc"),
                    ("ap", "ap"),
                    ("f1", "f1"),
                    ("iter", "iter"),
                    ("secs", "secs"),
                ]
                rows = [
                    [
                        r["run_name"],
                        r["variant_label"],
                        str(r["seed"]),
                        fmt(r["val/roc_auc"], 5),
                        fmt(r["val/average_precision"], 4),
                        fmt(r["val/f1_best"], 4),
                        fmt(r["val/best_iteration"], 0),
                        fmt(r["training_elapsed_s"], 1),
                    ]
                    for r in run_rows
                ]
                rows.sort(
                    key=lambda cells: float(cells[3]) if cells[3] != "—" else -1,
                    reverse=True,
                )
                self._rebuild(f"{tag} — runs ({len(rows)})", columns, rows)
                self.query_one(DataTable).sort("auc", reverse=True)

            def _show_summary(self) -> None:
                self.show_runs = False
                columns = [
                    ("variant", "variant"),
                    ("N", "n"),
                    ("roc_auc", "auc"),
                    ("average_precision", "ap"),
                    ("f1_best", "f1"),
                    ("best_iteration", "iter"),
                ]
                rows = [
                    [
                        f"{s['label'].replace(',', ', ')} {'*' if s['is_best'] else ''}".rstrip(),
                        str(s["n"]),
                        s["roc_auc"],
                        s["average_precision"],
                        s["f1_best"],
                        s["best_iteration"],
                    ]
                    for s in summary_stats
                ]
                self._rebuild(f"{tag} — summary", columns, rows)

            def action_toggle_view(self) -> None:
                self._show_summary() if self.show_runs else self._show_runs()

            def action_sort_auc(self) -> None:
                self.query_one(DataTable).sort("auc", reverse=True)

        self._app = _App()

    def run(self) -> None:
        self._app.run()


def _use_tui(interactive: bool | None) -> bool:
    """Whether to use the Textual TUI instead of plain tables."""
    if interactive is False:
        return False
    if interactive is True:
        return True
    return bool(sys.stdout.isatty())


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
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=None,
        help="open the results in a scrollable Textual table (auto when TTY)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="force plain rich tables (disable the Textual browser)",
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
    complete = [r for r in runs if "val/roc_auc" in r.data.metrics]
    baseline = _baseline_values(complete or runs, varying_keys)
    pairs: list[tuple[dict, object]] = [
        (_run_row(r, varying_keys, baseline), r) for r in runs
    ]
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
    baseline_desc = ", ".join(
        f"{k}={_shorten(str(v))}" for k, v in baseline.items() if str(v)
    )
    if varying_keys:
        print(f"baseline config: {baseline_desc}")

    summary_stats = _summary_stats(rows)
    use_tui = _use_tui(False if args.plain else args.interactive)

    if not use_tui:
        _print_table(rows)
        _summary(rows)
    else:
        try:
            app = CompareApp(args.tag, rows, summary_stats)
            app.run()
        except Exception as exc:  # noqa: BLE001
            print(f"\nTextual UI failed ({exc}) — falling back to plain tables")
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
