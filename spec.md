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
| I7 | `PrismClient.unload_model(model)` always sends `POST /v1/unload` to the Prism server and always returns `True`; a failing or missing endpoint is logged, never raised, so `evaluate_model`'s per-model teardown never fails a run because of it. |
| I8 | `FoundryClient.generate()` forwards `top_k`, `repetition_penalty` and `stop` from `options` verbatim into the `/v1/chat/completions` payload when present, and omits them when absent; a malformed value (from these or the pre-existing `temperature`/`max_tokens`/`top_p`/`seed` options) fails that one scenario via `_failure_result`, never the whole run. |
| I9 | `FoundryClient.generate()` reads `reasoning_content` from the response (streaming delta or non-streaming message) and sets `thinking_chars` on the result when non-empty; `ttft_sec` reflects the first token of either kind and `answer_ttft_sec` the first content token (omitted when no content token arrives). Behavior for a response with no `reasoning_content` is unchanged. |

## 3. Failure modes (current behavior)

| Situation | Behavior | Where specified |
|---|---|---|
| `--chart path.png` requested but `matplotlib` is not installed | CLI prints a clear error and exits non-zero without producing a half-written PNG | `benchrig/reporting/charts.py` |
| `--csv path.csv` path is not writable | CLI prints a clear error and exits non-zero; no partial CSV left on disk | `benchrig/reporting/csv_export.py` |
| `--runtime onnx-gpu` selects a `generic-cpu` execution provider on a CUDA host | CLI prints a one-line warning naming the model and the measured slowdown ("generic-cpu on CUDA: 2–22 tok/s for qwen; Phi-3.5-mini may not finish in 5 min"); the run continues and exits 0 (I4) | `benchrig/core/runtimes.py` `warning_for_provider()` |
| VRAM baseline contains foreign GPU processes (other model, MCP server, IDE daemon) | `vram_baseline_dirty_mb` records the foreign-process VRAM (rounded to MB); `vram_baseline_mb` stays as the whole-GPU used memory (own + foreign); existing `vram_baseline_dirty` boolean keeps its current semantics | `benchrig/core/runner.py` `measure_vram_baseline()` |
| Prism server returns 503 with `Retry-After` (queue full or load lock contention) | Benchrig retries up to 3× with the server-provided delay (capped at 60 s); after that `PrismBusyError` is raised naming the reason from the JSON body (I6) | `benchrig/core/client.py` retry helper |
| A scenario's `options` has a sampling value the server rejects, or one that isn't even parseable client-side (e.g. `top_k: "many"`, `repetition_penalty: "high"`) | `FoundryClient.generate()`/`PrismClient.generate()` catches the `TypeError`/`ValueError` from building the payload and returns a normal failed-scenario result (I8); the rest of the run continues | `benchrig/core/client.py` `FoundryClient.generate()` |
| A `coding` scenario fails (`passed=False`) | The result record additionally carries `extracted_code` (first `sandbox.EXTRACTED_CODE_PREVIEW_CHARS` chars) and `response_excerpt` (first `CODING_RESPONSE_EXCERPT_CHARS` chars of the raw response — head-truncated, since a coding prompt asks for code first), so the failure is diagnosable from the saved JSON alone (Phase 9, item 9.1, shipped). A passing scenario carries neither field. | `benchrig/core/runner.py` `run_coding_suite`'s `score()` |

## 4. Planned behavior (not implemented)

### Phase 1: Charts and CSV export

A run with `--csv <path>` produces a CSV at `<path>` with one row per scorecard and a stable header defined as `SCORECARD_CSV_COLUMNS` in `benchrig/reporting/csv_export.py`. The columns are:

`model`, `runtime`, `engine`, `composite_score`, `coding_pass_rate`, `reasoning_accuracy`, `avg_eval_tok_sec`, `avg_ttft_sec`, `peak_vram_mb`, `total_runs`.

A run with `--chart <path>` produces a PNG at `<path>` with one bar per scorecard. The bar height is `composite_score` (0–100); the bar label is `<model> (<runtime>)`, rotated 30° with right alignment (`ax.set_xticklabels(labels, rotation=30, ha="right")`) so labels do not overlap into an unreadable strip when there are more than a couple of scorecards or the labels are long. `matplotlib` is imported only inside the chart function; the rest of the CLI must load without it.

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

### Phase 5: Real model unload for Prism (`POST /v1/unload`) — shipped, see I7

Grounded in `~/03-foundy-local` `main` commit `6d6467a` (post-`v0.2.0`; not yet tagged). Contract (`docs/api.md`, `spec.md` P8
in that repo): `POST /v1/unload` drops the ONNX model held by `ActiveEngineManager`, is idempotent, takes an optional
JSON-object body (any object accepted and ignored; a non-empty non-object body is `400`), requires the API key when
`--api-key` is set, and replies `200 {"unloaded": bool, "model": string|null}`. `manager.unload()` holds the engine lock,
so the call is synchronous and can take seconds if a generation is in flight. Ollama models served through Prism are not
affected (only ONNX models go through `ActiveEngineManager`).

**Gap today:** `benchrig/cli.py::evaluate_model` already calls `client.unload_model(model)` after each model's suites
("Unload after the test so the next model starts from clean memory", gated on `unload_after_test`), and
`BaseRuntimeClient.unload_model` is part of the client contract. `PrismClient` does not override it, so it inherits
`FoundryClient.unload_model`, which only acts `if self._managed_by_foundry_cli()` — always `False` for Prism
(`auto_detect_port=False`). The call is a silent no-op: the console prints "Unloading memory for model X (Prism)..." but
nothing happens, so the VRAM baseline noise described in `intent.md` §1 and the capacity-exhausted short-circuit (plan.md
Phase 4 item 4.6) both see whatever the *previous* model left resident.

**New behavior:** `PrismClient.unload_model(model_name)` calls `POST /v1/unload` on the Prism server (with the bearer
token from `_request_kwargs()` when `api_key` is set) and returns `True` regardless of outcome — unload is best-effort
and must never fail a benchmark run (matching `FoundryClient.unload_model`'s existing contract and
`BaseRuntimeClient.unload_model`'s docstring). Any transport error, non-2xx status (including `404` from a prism-local
build older than `6d6467a`, which has no `/v1/unload` route), or unexpected body is logged at `_log.info` and swallowed;
`model_name` is used only for the log line, since the server's own response already names the released model id.

Shipped (plan.md Phase 5); the invariant is I7 in §2.

### Phase 6: Sampling parameter passthrough and reasoning content (`FoundryClient`/`PrismClient`) — shipped, see I8/I9

Grounded in `~/03-foundy-local` `main` commit `e01569c` (`top_k`/`repetition_penalty` on `/v1/chat/completions`, both
ONNX and Ollama backends) and `4d8567d` (reasoning content separated out of `<think>...</think>` server-side into a
dedicated `reasoning_content` field), plus the pre-existing `stop` parameter (`4a77b5c`, already in `v0.2.0`) that
Prism has accepted since before `v0.2.0` but `benchrig` has never sent.

**Gap today:** `FoundryClient.generate()` (`benchrig/core/client.py`, used directly by the `foundry` runtime and
inherited by `PrismClient`) only forwards `temperature`, `num_predict`/`max_tokens`, `top_p` and `seed` from the
caller's `options` dict into the JSON payload. A scenario that sets `options: {"top_k": 40, "repetition_penalty": 1.1,
"stop": ["\n\n"]}` sends those keys to `OllamaClient` (which merges `options` wholesale) but they are silently
dropped for `foundry`/`prism` — the server never sees them, and no error is raised. Separately, the response's
`reasoning_content` (delta in streaming, message field in non-streaming) is not read at all: only `content` is
accumulated, so `thinking_chars` (the field `benchrig.core.runner.BenchmarkRunner._base_record` already forwards
generically into scorecards, `client.py:310`) is never populated for Foundry/Prism, unlike `OllamaClient` which sets
it from `thinking`. A side effect: today's `ttft_sec` for Foundry/Prism is measured to the first *content* token,
because reasoning deltas are invisible to the client — for a reasoning model that thinks before answering, this
already reports something closer to "answer TTFT" than "first token of any kind" (`OllamaClient`'s contract,
`client.py:371-372`); Phase 6 makes the two runtimes consistent.

**New behavior:**

1. `FoundryClient.generate()` forwards, when present in `options`:
   - `top_k` → `top_k` (`int(...)`) in the payload.
   - `repetition_penalty` → `repetition_penalty` (`float(...)`) in the payload.
   - `stop` → `stop` in the payload, passed through unchanged (a string or a list of up to 4 non-empty strings —
     Prism's own limit, `prism/server.py` `MAX_STOP_SEQUENCES`). BenchRig does not validate `stop` client-side; a
     malformed value gets the server's `400`, which surfaces through the client's normal error handling
     (`raise_for_status()` → `_failure_result`), not a client-side exception.
   These join the existing whitelist; a key absent from `options` is still omitted from the payload (no defaults are
   invented). Building the payload from `options` (this step, and the pre-existing `temperature`/`max_tokens`/`top_p`/
   `seed` conversions) is wrapped in `try/except (TypeError, ValueError)`: a value that cannot be coerced (e.g.
   `options={"top_k": "many"}`) returns a normal failed-scenario result via `_failure_result`, the same shape as a
   request-level failure, instead of raising out of `generate()` and aborting the whole `--runs N` benchmark (review
   finding, fixed 2026-09-22).
2. `FoundryClient.generate()` accumulates `reasoning_content` separately from `content`:
   - Streaming: `choices[0].delta.reasoning_content`, chunk by chunk, alongside the existing `content` accumulation;
     an empty-string chunk is a no-op (falsy), and pieces across multiple chunks are joined in order.
   - Non-streaming: `choices[0].message.reasoning_content`.
   - The result gains `thinking_chars` (`len` of the accumulated reasoning text) **only when non-empty**. This is
     not byte-for-byte `OllamaClient`'s shape: `OllamaClient` always includes `thinking_chars` (0 when there was
     none) and `answer_ttft_sec` (`None` when there was no reasoning), while `FoundryClient` omits both keys
     entirely when there is no reasoning content. Every consumer (`runner.py:310` `resp.get("thinking_chars")`,
     `runner.py:312-313` `resp.get("answer_ttft_sec")`, `runner.py:444` `resp.get("thinking_chars", 0)`) reads both
     shapes identically, so this is a documented difference, not a bug.
   - `ttft_sec` is measured to the first token of *either* kind (reasoning or content), matching `OllamaClient`;
     `answer_ttft_sec` records the first *content* token's time separately when reasoning preceded it, and is
     omitted (not `None`) when no content token ever arrives (e.g. truncated at `max_tokens` while still
     reasoning). When no `reasoning_content` is present in the response, `ttft_sec` is computed exactly as today
     (first content token) — no behavior change for non-reasoning models.
3. `PrismClient` needs no override: it inherits the fixed `FoundryClient.generate()`. The plain `foundry` runtime
   gets the same fix, since Foundry Local's own OpenAI-compatible endpoint accepts the same OpenAI parameter names.

Shipped (plan.md Phase 6, reviewed); the invariants are I8 and I9 in §2.

### Phase 8: Real-time per-scenario feedback and an overall progress bar — shipped, retroactively documented

Found by the operator: a suite with many scenarios (e.g. 12-scenario `reasoning`) gave no terminal output at all
between "Running reasoning & logic tests..." and a block of per-scenario `PASS`/`FAIL` lines dumped all at once —
`BenchmarkRunner._run_suite` (and `run_context_suite`) built the whole suite's result list in memory and
`benchrig/cli.py::evaluate_model` only called `display_scenario_result` after `getattr(runner, method_name)(...)`
returned, so a long-running suite looked stalled even though scenarios were completing one by one.

**New behavior:** `BenchmarkRunner.__init__` takes an optional `on_result: Callable[[dict], None] | None` (alongside
the pre-existing `progress_callback`); `_run_suite` and `run_context_suite` call it with each scenario's finished
record immediately after appending it to the suite's result list, before moving to the next scenario. `on_result`
defaults to `None` (no behavior change for existing callers, e.g. `tests/test_runner_suites.py`).
`benchrig/cli.py::evaluate_model` wires `on_result` to a closure that calls `display_scenario_result` (moved out of
the old post-suite loop, so each line prints as soon as that scenario finishes) and an optional `on_progress` hook.
`run_benchmarks` wraps the per-target loop in a `rich.progress.Progress` bar (model:runtime label, bar,
completed/total count, elapsed, ETA) sized to `len(targets) * args.runs * sum(len(scenarios) per suite)`, and passes
`on_progress=lambda _record: progress.advance(task)` into `evaluate_model`. A model skipped for capacity, or a suite
with fewer scenarios than requested, leaves the bar short of 100% for that run instead of raising — the bar is a UX
aid, not a completion invariant.

Shipped (plan.md Phase 8); no new numbered invariant (cosmetic CLI feedback, not a correctness contract).

### Phase 9: Coding-suite failure diagnostics (investigation of `results/runs/runs/benchmark_20260922_214037.json`)

**Investigation, not yet spec'd as a fix.** The operator flagged that run's `coding` suite: `Phi-4-mini-instruct-cuda-gpu`,
`Phi-4-mini-instruct-generic-cpu-5:v5` and `Phi-3.5-mini-instruct-generic-cpu-2:v2` scored 0 `passed_tests` on every
scenario (`IndentationError`/`SyntaxError` in `sandbox_error`); `mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32`
passed 2 of 4. All had `finish_reason: "stop"` (not truncated) and plausible `eval_count`s — the model finished
normally and still produced code the sandbox could not run.

**Leading hypothesis going in** (`extract_python_code`, `benchrig/core/sandbox.py:17-31`, only calls `.strip()` on
the whole extracted/joined block, never dedents internal lines, and joins multiple `def`/`class` blocks with
`"\n\n".join(...)` with no per-block cleanup) was **not confirmed**: replaying the exact failing
`(model, scenario)` pairs — same prompt, same `temperature: 0.1`/`num_predict` from `benchrig/data/scenarios/coding.json`,
same live Prism server, same `extract_python_code`/`run_code_with_tests` — passed 15/15 times across
`code_flatten_dict`, `code_merge_intervals`, `code_lru_cache` and `code_balanced_parentheses` for
`Phi-4-mini-instruct-cuda-gpu` (2026-09-22, this session). Every replayed response was flush-left, single-block,
valid Python. This does not rule out the dedent gap as a real latent bug — it is still worth hardening — but it is
not shown to be *this run's* cause.

**Working alternative hypothesis:** `execution_alignment_1to1.seed: 42` (`benchrig/data/config.yaml`) is applied to
every run via `BenchmarkRunner._sampling_defaults()`/`_with_sampling()` (not just `--pair` 1:1 comparisons), on the
assumption that a fixed seed plus `temperature: 0.1` makes suites reproducible — this is intent.md Constraint 4
("Suites are deterministic … No flaky-by-design benchmarks"). ONNX Runtime GenAI on CUDA does not guarantee
bit-identical output for a fixed seed under different GPU occupancy (cuBLAS/cuDNN algorithm selection can vary with
free memory and concurrent kernels); the flagged run had tested several models back-to-back in the same Prism
process (which has no automatic unload between models pre-Phase-5, and multiple `generic-cpu` models forced onto
CUDA — `AGENTS.md`'s slow-provider note) shortly before the Phi-4-mini coding suite ran, a GPU-load profile this
session's clean replay did not reproduce. This would mean Constraint 4 is not currently upheld for the `coding`
(and by extension `reasoning`) suites on this backend, and Constraint 5 ("numbers reported are honest … never
hidden") is also not met, since nothing today flags or captures evidence of it happening.

**Not confirmed either** — there is no captured evidence (raw response, GPU occupancy at request time) from the
original run to prove the alternative hypothesis; only that the leading hypothesis failed 15/15 reproduction
attempts and this alternative is consistent with what *is* known (intent.md's existing, unresolved ONNX-int4
reasoning-quality gap open item is a different but related symptom of the same runtime).

**Proposed next step (needs operator decision on scope):** before any behavior change, capture enough to diagnose
the *next* occurrence without live reproduction: persist `extracted_code` (already computed, currently discarded)
and a bounded raw-response excerpt into the `coding` suite's result record whenever `passed` is `False` — see
plan.md Phase 9 item 9.1. Whether to also do something about the suspected non-determinism (retry-on-failure,
GPU-occupancy sampling at request time, a determinism warning, or nothing beyond documenting it) is an open
question for the operator; no invariant or new behavior is specified for it yet.

**Shipped 2026-09-22 (both items approved by the operator):**

- **Item 9.1** (§3 table has the current behavior): `extracted_code`/`response_excerpt` are now kept on a failing
  `coding` record.
- **Item 9.2** (hardening — still not a confirmed fix for the flagged run, see above): `extract_python_code`
  (`benchrig/core/sandbox.py`) dedents (`textwrap.dedent`) a fenced block **only when it is the single fence in
  the response**, before the `code_blocks_with_def` filter and the final `.strip()`. Independent review found
  that the first version of this fix dedented *every* matched block independently, including when
  `code_blocks_with_def` joins more than one — which broke a real, different pattern: a model showing a class in
  one fence and a method continuation (indented relative to that class, not to markdown) in a second fence. Since
  there is no reliable way to tell "spurious markdown-list margin" from "intentional continuation indent" once
  there is more than one fence, dedenting is now scoped to the unambiguous case (exactly one fence) only.
