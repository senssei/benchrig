# Specification: BenchRig

What must stay true (invariants), what happens when things go wrong (failure modes), and behavior that is planned but not built.
It does not repeat wire formats or command references: name the normative doc (for example `docs/api.md`) and change it in the
same commit as the behavior.

## 1. Components

`benchrig.cli:main` parses flags, picks a runtime (`ollama`, `foundry`, `onnx-gpu`, `prism`, `all`), and hands a `BenchmarkRunner` (`benchrig.core.runner`) the model list, suite and `--runs`. The runner:

1. Asks `benchrig.core.hardware` for the host's telemetry provider (Darwin Apple Silicon, Linux NVIDIA).
2. Calls the matching client in `benchrig.core.client` (`OllamaClient`, `FoundryClient`, `PrismClient`, ONNX direct) — each one implements `BaseRuntimeClient`.
3. Runs each requested suite (`speed`, `coding`, `reasoning`, `polish`, `context`) by way of methods on the runner that map to scenario JSON in `benchrig/data/scenarios/`.
4. Aggregates per-scenario records into per-model scorecards, pre-filled with the notes from `benchrig.core.runtimes`.
5. Hands the scorecards to `benchrig.reporting`: terminal (`display`), Markdown (`markdown`), and (Phase 1) CSV (`csv_export`) + PNG chart (`charts`).

Each scorecard is a `dict`; the JSON in `results/runs/<ts>.json` is the same shape. Architecture: `README.md` §"System Architecture".

## 2. Invariants

Tests and reviews cite these by number. Changing one needs operator approval.

| # | Invariant |
|---|---|
| I1 | Every scorecard emitted by the CLI is also representable as one CSV row in the `--csv` export and one bar in the `--chart` PNG, in that order. |
| I2 | A run without `--csv` or `--chart` produces no CSV and no PNG; the Markdown report does not reference paths that were not produced. |
| I3 | `matplotlib` is imported lazily; the CLI still loads and exits 0 when the `[charts]` extra is not installed and no `--chart` is requested. |

## 3. Failure modes (current behavior)

| Situation | Behavior | Where specified |
|---|---|---|
| `--chart path.png` requested but `matplotlib` is not installed | CLI prints a clear error and exits non-zero without producing a half-written PNG | `benchrig/reporting/charts.py` |
| `--csv path.csv` path is not writable | CLI prints a clear error and exits non-zero; no partial CSV left on disk | `benchrig/reporting/csv_export.py` |

## 4. Planned behavior (not implemented)

### Phase 1: Charts and CSV export

A run with `--csv <path>` produces a CSV at `<path>` with one row per scorecard and a stable header defined as `SCORECARD_CSV_COLUMNS` in `benchrig/reporting/csv_export.py`. The columns are:

`model`, `runtime`, `engine`, `composite_score`, `coding_pass_rate`, `reasoning_accuracy`, `avg_eval_tok_sec`, `avg_ttft_sec`, `peak_vram_mb`, `total_runs`.

A run with `--chart <path>` produces a PNG at `<path>` with one bar per scorecard. The bar height is `composite_score` (0–100); the bar label is `<model> (<runtime>)`. `matplotlib` is imported only inside the chart function; the rest of the CLI must load without it.

The Markdown report links to the chart (as `![](chart_path)`) and to the CSV (as `[results](csv_path)`) **only** when those files were produced this run. When neither was produced, the report is unchanged.

These three behaviors move up into §2 (I1, I2, I3) and §3 once Phase 1 ships.
