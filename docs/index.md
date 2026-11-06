# Documentation

Map of the project's documentation. Code and configs are the source of truth;
these documents explain the contracts that are hard to see from a single file.

## Reading order

| If you are… | Read in this order |
|---|---|
| Lost / retaking the project | [project map](project-map.md) → `configs/*.yaml` → the 8 files listed there |
| New to the project | [README](../README.md) → [results](results.md) → [architecture](architecture.md) → [pipeline](pipeline.md) |
| Reviewing the ML work | [results](results.md) → [decisions](decisions.md) → [evaluation](implementation/evaluation.md) |
| Contributing code | [architecture](architecture.md) → [implementation](#implementation) → `configs/*.yaml` |
| An agent / future maintainer | [AGENTS.md](../AGENTS.md) → this page → `configs/*.yaml` → code |

## Documents

| Document | Answers |
|---|---|
| [project map](project-map.md) | One-page model of the whole repo: flow, module map, reading order, current state, gaps. |
| [README](../README.md) | What the project is, quickstart, API reference, roadmap. |
| [architecture](architecture.md) | Package layout, serving contract, config flow, MLflow resolution, project-root discovery. |
| [pipeline](pipeline.md) | DVC stages, temporal split, feature engineering stage, CRISP-DM mapping. |
| [decisions](decisions.md) | Why the important choices were made (ADR-style log). |
| [results](results.md) | Measured metrics, provenance of each number, ablation snapshot, how to regenerate. |
| [implementation/features](implementation/features.md) | Feature pipeline: step order, column contract, invariants, how to add a feature. |
| [implementation/training](implementation/training.md) | Training phases, split, HPO, early stopping, artifacts, MLflow, registry, ablation. |
| [implementation/evaluation](implementation/evaluation.md) | Metrics and thresholds, segments, drift, stability, reporters, outputs. |
| [implementation/api](implementation/api.md) | FastAPI app, model resolution, endpoints, scoring flow, security, deployment. |

## Implementation

The `docs/implementation/` deep dives describe how each subsystem is built and
where its seams are. They are intentionally narrower than
[architecture](architecture.md): architecture explains how the pieces fit,
implementation explains how each piece works.

## Regenerating the artifacts behind the docs

| Command | Writes |
|---|---|
| `just evaluate` | `models/report.json`, `models/model_card.md`, plots, `dvclive/` |
| `just train model.name=lightgbm` | MLflow run, `models/pipeline.joblib`, `models/report.json` |
| `just train ablation.enabled=true` | `models/feature_ablation.csv` + one MLflow run per variant × seed |
| `just schemas` | `schemas/*.schema.json` from the Pydantic models |

`models/` is git-ignored except for `models/.gitkeep`. The numbers in
[results](results.md) are a committed snapshot of those local artifacts; re-run
the commands above to refresh them.

## Documentation rules

- A document explains a contract, a decision, or a measured result. It does not
  restate code line by line.
- If a document and the code disagree, the code wins. Fix the document.
- Any reported metric names its source (artifact path, command, or API endpoint)
  and whether it was measured locally or reported by a deployed service.
- `docs/tasks/*`, `AGENTS.md`, `PLANS.md`, and `.github/**/*.md` are local-only
  planning material, excluded from git via `.git/info/exclude`.
