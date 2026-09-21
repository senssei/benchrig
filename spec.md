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
| I4 | The CLI never exits non-zero solely because the user picked a slow execution provider. The provider warning is printed once per run, before any model is benchmarked, and the run proceeds. |
| I5 | When benchrig sends `stream_options={"include_usage": True}` to a Prism server, the result record carries `usage_estimated=False`, `usage={"prompt_tokens", "completion_tokens", "total_tokens"}`, and `telemetry={"device", ...}` exactly as the server reported; absent or malformed usage still sets `usage_estimated=True`. |
| I6 | A 503 from prism-local with `Retry-After` is retried up to 3× with the server-provided delay (capped at 60 s); after that the run fails with `PrismBusyError` naming the reason from the JSON body. Other 5xx responses (4xx, 500, 502, 504) are NOT retried. |

## 3. Failure modes (current behavior)

| Situation | Behavior | Where specified |
|---|---|---|
| `--chart path.png` requested but `matplotlib` is not installed | CLI prints a clear error and exits non-zero without producing a half-written PNG | `benchrig/reporting/charts.py` |
| `--csv path.csv` path is not writable | CLI prints a clear error and exits non-zero; no partial CSV left on disk | `benchrig/reporting/csv_export.py` |
| `--runtime onnx-gpu` selects a `generic-cpu` execution provider on a CUDA host | CLI prints a one-line warning naming the model and the measured slowdown ("generic-cpu on CUDA: 2–22 tok/s for qwen; Phi-3.5-mini may not finish in 5 min"); the run continues and exits 0 (I4) | `benchrig/core/runtimes.py` `warning_for_provider()` |
| VRAM baseline contains foreign GPU processes (other model, MCP server, IDE daemon) | `vram_baseline_dirty_mb` records the foreign-process VRAM (rounded to MB); `vram_baseline_mb` stays as the whole-GPU used memory (own + foreign); existing `vram_baseline_dirty` boolean keeps its current semantics | `benchrig/core/runner.py` `measure_vram_baseline()` |
| Prism server returns 503 with `Retry-After` (queue full or load lock contention) | Benchrig retries up to 3× with the server-provided delay (capped at 60 s); after that `PrismBusyError` is raised naming the reason from the JSON body (I6) | `benchrig/core/client.py` retry helper |

## 4. Planned behavior (not implemented)

### Phase 1: Charts and CSV export

A run with `--csv <path>` produces a CSV at `<path>` with one row per scorecard and a stable header defined as `SCORECARD_CSV_COLUMNS` in `benchrig/reporting/csv_export.py`. The columns are:

`model`, `runtime`, `engine`, `composite_score`, `coding_pass_rate`, `reasoning_accuracy`, `avg_eval_tok_sec`, `avg_ttft_sec`, `peak_vram_mb`, `total_runs`.

A run with `--chart <path>` produces a PNG at `<path>` with one bar per scorecard. The bar height is `composite_score` (0–100); the bar label is `<model> (<runtime>)`. `matplotlib` is imported only inside the chart function; the rest of the CLI must load without it.

The Markdown report links to the chart (as `![](chart_path)`) and to the CSV (as `[results](csv_path)`) **only** when those files were produced this run. When neither was produced, the report is unchanged.

These three behaviors move up into §2 (I1, I2, I3) and §3 once Phase 1 ships.

### Phase 2: Runtime warnings (`generic-cpu` on CUDA, dirty VRAM baseline)

A new helper `benchrig.core.runtimes.warning_for_provider(provider: str | None, host_gpu_type: str) -> str | None` returns a one-line warning string when:

- `provider` is `"generic-cpu"` (case-insensitive) and
- `host_gpu_type` is `"nvidia"` (a CUDA host)

otherwise `None`. The warning text is:

> `generic-cpu execution provider on a CUDA host is very slow (2–22 tok/s for qwen; Phi-3.5-mini may not finish 4500 tokens in 5 min). Use the cuda provider or benchmark on CPU-only hardware.`

The CLI prints the warning once per run, before any model is benchmarked, and exits 0 (I4). Apple Silicon (`gpu_type="apple_silicon"`) and other providers return `None`.

A new scorecard field `vram_baseline_dirty_mb: float | None` records the foreign-process GPU memory in MB, rounded to one decimal. It is set by `BenchmarkRunner.measure_vram_baseline()` before the model is loaded. When no foreign memory is present it is `0.0`; when the measurement is unavailable (e.g. macOS, or nvidia-smi missing) it is `None`. The existing `vram_baseline_mb` and `vram_baseline_dirty` keep their semantics; `vram_baseline_dirty_mb` is purely additive.

These behaviors move up into §2 (I4) and §3 once Phase 2 ships.

### Phase 4: Integration with `prism-local` HEAD (`~/03-foundy-local`)

Benchrig already implements items 4.1 and 4.2; the items below pin the contract with regression tests and add new behavior for 4.5. Behavior is grounded in `~/03-foundy-local` `main` (8 commits past `v0.2.0`; tag `v0.2.0` is on disk).

**4.1 stream usage (verification, no new code).** When benchrig sends `stream_options={"include_usage": True}` to a Prism server (`benchrig.core.client.OpenAICompatibleChatClient._chat_complete_openai`, also used by Foundry), the parser at `benchrig.core.client.OpenAICompatibleChatClient._parse_stream_response` extracts the last chunk's `usage` and `telemetry` and stamps `usage_estimated=False` on the result record (I5). A regression test in `tests/test_prism_runtime.py::test_prism_stream_records_usage_and_telemetry` pins the exact chunk shape from `prism-local` HEAD `tests/test_prism_server_api.py:307-317`.

**4.2 device label (verification, no new code).** Each `/v1/models` entry has `device` (resolved at serve time, e.g. `"CUDA (GPU)"` or `"CPU"`) and `exported_for` (the label the user passed). Benchrig's Markdown report derives the runtime label via `runtime_label(sc.get("runtime"))` which is fed from `telemetry.device` on each result (client.py:867-874). A regression test in `tests/test_prism_runtime.py::test_prism_models_response_device_overrides_exported_for` pins that the scorecard picks the resolved device, not the user-supplied label.

**4.3 env-var documentation (no new code).** `benchrig --help` and `docs/runtimes.md` document `PRISM_PREFILL_CHUNK` (default `1024`, `"off"` or `0` disables chunking), `PRISM_THREADS` (default `None`, positive int), and `PRISM_DEVICE` (default `"auto"`, choices `auto|cuda|cpu`). Trade-off for `PRISM_PREFILL_CHUNK` from `scratch/TODO.md` §2 row 8: peak −44/−54/−5 %, TTFT +4/+110/+33 %. Stays opt-in (user sets the env when starting `prism serve`). Test: `tests/test_docs.py::test_prism_env_vars_are_documented`.

**4.5 503 retry (new code).** Prism returns 503 with `Retry-After: 30` when the queue is full (`server_busy`) or load-lock contention is detected (`insufficient_resources`). Benchrig's `OpenAICompatibleChatClient` (and the Prism `chat_complete_openai` path through it) gets a `_post_with_503_retry(url, payload, headers)` helper: on 503 it waits the server-provided `Retry-After` (capped at 60 s) and retries up to 3 times; on the 4th 503 it raises `PrismBusyError(reason=...)` with the reason from the JSON body (`error.code` / `error.type`). Other 5xx (500, 502, 504) and all 4xx are NOT retried. The helper logs one line per retry attempt at the `console.print("[dim]…")` level so the user sees what's happening during long waits. (I6.)

These behaviors move up into §2 (I5, I6) once Phase 4 ships.
