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
| I10 | In the coding benchmark sandbox, candidate stdout and stderr never determine the test verdict or expected test count. Expected test count is strictly `len(test_assertions)`. Test execution status is communicated out-of-band (via a dedicated temporary result file verified with a secret token), and any premature process termination or forged markers fail against the caller's assertions. |
| I11 | Scorecards and records distinguish `requested_device` from `observed_device` across Ollama, Foundry Local, direct ONNX Runtime GenAI and Prism; when observed placement is unavailable, it is recorded as `"unknown"`, and a GPU request running on CPU is flagged with `cpu_fallback=True`. |
| I12 | TTFT, prefill duration, and decode duration are reported as separate metrics with explicit units and provenance tags (`prefill_provenance`: `"engine"` \| `"unavailable"`); TTFT is never labelled as engine prefill duration. |
| I13 | Peak host RSS (in MB) is sampled and reported alongside platform GPU/UMA memory, explicitly qualified by process coverage (`rss_coverage`: `"client_only"` \| `"client_and_server"`) and fixed workload context parameters (`context_tokens`, `prompt_tokens`, `max_output_tokens`). |
| I14 | Warm-up runs (`--warmup-runs N`, default 1) are executed before steady-state measurement and strictly excluded from benchmark summary averages; repeated identical runs record `cache_mode` (`"cold"`, `"warm"`, `"prefix_cached"`, `"unverified"`), and prefix cache hits require backend verification. |
| I15 | Coding suite metrics report both task-level pass counts (`coding_tasks_passed / coding_task_count`) and assertion-level pass counts (`coding_assertions_passed / coding_assertion_count`) separately, with per-task outcomes preserved in run records. |

## 3. Failure modes (current behavior)

| Situation | Behavior | Where specified |
|---|---|---|
| Candidate code in coding sandbox attempts to forge test verdict or exits early (`SystemExit(0)`, forged stdout markers, forged totals) | The sandbox ignores candidate stdout for verdict determination, enforces `total_tests = len(test_assertions)`, requires out-of-band completion evidence, and marks `passed=False` with error detail when assertions fail or did not run (I10) | `benchrig/core/sandbox.py` `run_code_with_tests()` |
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

### Phase 10: Display honesty for floor-driven decode-speed ceilings

`FoundryClient`/`PrismClient` floor the eval window at 0.001 s (`eval_duration_sec = max(0.001, raw)`) so a
sub-millisecond generation cannot divide by zero. The resulting `eval_tok_per_sec` (for example `3 tokens / 0.001 s =
3000.0 t/s`) is a ceiling produced by the floor, not a measurement. Intent Constraint 5 (report what was measured) is
upheld by *labelling* such values, never by changing them.

- **Per-response flag.** `FoundryClient`/`PrismClient.generate()` set `eval_tok_sec_floored = True` when `eval_count > 0`
  and the raw eval window is below `EVAL_DURATION_FLOOR_THRESHOLD_SEC` (0.0015 s), including a raw window of exactly
  0.0, on both the streaming and the non-streaming path; otherwise `False`. `OllamaClient` reports its own
  `eval_duration` and applies no floor, so it always sets `False`.
- **Record and scorecard.** `BenchmarkRunner._base_record` copies the flag onto the per-scenario record when true (the key
  is absent otherwise, so raw run JSON stays unchanged for unaffected records). `compute_model_scorecard` sets the
  scorecard's `eval_tok_sec_floored` to true if any token-producing record is floored.
- **Markdown.** The leaderboard speed cell renders `~<value> t/s` for a floored scorecard, plain `<value> t/s` otherwise.
  Forward-only: existing reports are not rewritten.
- **CSV.** `SCORECARD_CSV_COLUMNS` ends with `eval_tok_sec_floored` (`True`/`False`; a scorecard from a run that predates the
  flag exports an empty cell — unknown, not `False` — and the column is never missing — I1).
- **Known limit.** One floored answer among many real ones marks the whole scorecard `~` (tracked in `plan.md` Backlog).

### Phase 11: Structured logging on stderr (no stdout UX change)

Operator request after `v0.2.0`: make benchrig's diagnostic events machine-parseable
without changing the colored terminal UX that `rich.console.print` provides today.
Stdlib `logging` only — no OpenTelemetry, no file output, no log aggregation. Stderr
gets one JSON object per record; stdout keeps the rich UX untouched.

**Surface:**

- New module `benchrig/core/logging.py`:
  - `JsonFormatter(logging.Formatter)` — emits one JSON object per record with the
    stable shape `{"ts": "<ISO 8601 UTC, ms precision>", "level": "<DEBUG|INFO|...|CRITICAL>",
    "event": "<dotted name>", "run_id": "<uuid4 or absent>", "message": "<formatted>"}`,
    plus any per-event extras (`model`, `runtime`, `engine`, `attempt`,
    `duration_sec`, `ttft_sec`, `eval_count`, `ok`, `error`, `reason`, `delay_sec`,
    `url`, `total_duration_sec`, `models_ok`, `models_skipped`, `argv`, `num_models`,
    `runs`). Keys are always strings; numeric values are JSON numbers; booleans JSON
    booleans.
  - `RunIdFilter(logging.Filter)` — reads `run_id` from a `contextvars.ContextVar`
    that `cli.py::main` sets at startup; absent when unset (tests).
  - `setup_logging(level: str = "WARNING", run_id: str | None = None) -> logging.Logger`
    — installs a `StreamHandler(sys.stderr)` with the formatter + filter on the benchrig
    logger (idempotent: replaces any existing handler, never duplicates).
- New CLI flag `--log-level {DEBUG,INFO,WARNING,ERROR}` (default `WARNING`) and env
  var `BENCHRIG_LOG_LEVEL` (overrides the default; `--log-level` overrides the env).
  Invalid values clear-message + non-zero exit.

**Events (level = when):**

| event | level | where | fields |
|---|---|---|---|
| `run.started` | INFO | `cli.py::run_benchmarks` top | `argv`, `runtime`, `num_models`, `runs` |
| `run.completed` | INFO | `cli.py::run_benchmarks` finally | `total_duration_sec`, `models_ok`, `models_skipped` |
| `http.request_started` | INFO | every `requests.post` in `benchrig/core/client.py` | `model`, `runtime`, `engine`, `url`, `attempt` |
| `http.response_completed` | INFO | on 2xx | `model`, `runtime`, `engine`, `attempt`, `status_code`, `duration_sec` |
| `http.request_failed` | WARNING | on non-2xx (every attempt, so each 503 of a `PrismBusyError` run) or a transport error | `model`, `runtime`, `engine`, `attempt`, `error`, `status_code` (non-2xx) |
| `retry.attempted` | INFO | per retry in `_post_with_503_retry` | `attempt`, `delay_sec`, `reason` |
| `retry.exhausted` | WARNING | budget run out | `attempt`, `reason` |
| `capacity.exhausted` | WARNING | `BenchmarkRunner._capacity_exhausted_reason` triggers | `model`, `reason` |
| `unload.completed` | DEBUG | `PrismClient.unload_model` returns | `model`, `ok`, `error` (when failed) |

`ttft_sec` and `eval_count` are not on `http.response_completed`: a streamed response is only parsed after the HTTP layer
returns, so they are not known there (they are in the JSON report). `http.request_failed` for a GET (health/version/list
probes such as `is_reachable`) is logged at DEBUG, not WARNING: a stopped daemon is an expected answer to a probe and
must not print JSON on stderr at the default level. `--check` and `--pull-recommended` are not benchmark runs and emit
no `run.*` events.

**What is NOT logged:**

- Per-scenario PASS/FAIL (`on_result` already prints; the JSON report and Markdown
  scorecard carry the same information). Adding a duplicate event would 4× volume on
  a 12-scenario reasoning run with `--runs 3`.
- Scorecard aggregation (progress bar + Markdown report already cover it).
- Anything emitted on stdout by `rich.console.print` (UX). The colored output stays
  where the operator sees it today.

**Out of scope (defer to a later phase):** file output (`--log-file`), log rotation,
OpenTelemetry OTLP export, `trace_id`/`span_id` propagation, integration with the
external `~/03-foundy-local` server. Re-scope into a numbered phase when needed.


## Adversarial remediation: Sandbox verdict integrity and trust boundary

- **Trust boundary:** In `benchrig/core/sandbox.py` (`run_code_with_tests`), candidate code and any output emitted on `stdout` or `stderr` is untrusted data. Candidate output MUST NEVER determine the test verdict (`passed`), `passed_tests`, or `total_tests`.
- **Expected test count:** `total_tests` is strictly derived from the caller's test assertions list (`len(test_assertions)`). The runner never allows candidate code or child process output to override or forge `total_tests`.
- **Out-of-band result communication:** The test execution harness writes execution results (whether all assertions completed, number of assertions passed, failure messages) via a dedicated temporary result file verified with a secret token, completely separate from candidate `stdout`.
- **Early exit and failure behavior:** If candidate code terminates early (e.g. `raise SystemExit`, `sys.exit(0)`, `os._exit`, syntax error, uncaught exception) before all assertions in `test_assertions` have been executed, or fails to produce the expected out-of-band result verification, the verdict is marked as `passed=False`, with `total_tests` matching the caller's count and `passed_tests` reflecting only assertions genuinely executed and passed.
- **Pre-emitted or forged markers:** Any string printed to `stdout` (such as `__RESULT__:passed=...`) by candidate code is ignored by the parent evaluator. All tests must execute against the caller's original assertions.


## Workspace SDLC unification — 2026-10-02

Scope authorized by the operator's request to unify SDLC across projects 01–08. The workflow is intent → spec → plan → test (red) → code → independent review. Existing domain invariants and adversarial findings remain in force.

- Kit-owned runner, five skills, pre-commit hook and Cursor rule come from the sibling `local-sdlc-kit`; install/update with its installer, never maintain project forks of those files.
- `AGENTS.md` carries the same kit process section in every project; project-specific language, hardware, privacy and execution rules stay outside that section. Harness adapters point to `AGENTS.md`.
- `sdlc.toml` declares the actual checks, red command, timeout and changelog paths. The public gate command is `python3 scripts/sdlc_check.py`; selection uses `--only NAME`, red uses `--red ID`. Gate tooling requires Python 3.11+ independently of product runtime support.
- Migration preserves existing verification controls. Project-specific changed-line lint in Prism remains a separate helper; common runner logic must not absorb language-specific behavior. Pytest collection errors and missing unittest ids must be NOT RED.
- Missing/invalid gate configuration or unknown checks fail explicitly. Missing optional documentation tooling may only skip where the prior gate allowed it. A skipped or unavailable check is reported, not presented as verified.
- 01 gains static Python compilation and JSON/configuration checks; live Windows probes remain manual. 04 gains its existing Astro build as the gate; neither project claims a behavioral test suite that does not exist. Their red interfaces are configured for future unittest/Node test ids and reject missing tests.
- Workspace consistency checks compare installed kit-owned files and process sections with the kit source. Project check sets stay distinct; no common lowest-denominator test suite is imposed.
- Implementation status is recorded separately from verification. Plan boxes remain open until the complete project gate passes; unrelated pre-existing failures are preserved and reported. No commits, pushes, real engine calls or automatic hook activation are part of this change.

### Codex execution contract

Codex reads project AGENTS.md and routes through `.agents/skills/sdlc`. A natural-language request is sufficient; `$sdlc` is an explicit entry. Gate checks receive the selected comparison base through `SDLC_BASE`. Full `scripts/sdlc_check.py` is also configured in `.github/workflows/sdlc.yml`; existing CI jobs remain. A sandbox-blocked check remains unverified. Operator authorization persists within the requested scope.

## Planned behavior: Phase 14 — Interpretable benchmark comparisons

Scope added at the operator's request on 2026-10-03; implementation begins once the operator approves this plan.

### 14.1 Execution placement and CPU fallback (I11)
- **Requested vs Observed Placement:**
  - `requested_device`: Target device requested by caller (`"cuda"`, `"generic-cpu"`, `"metal"`, `"cpu"`, or `"auto"`).
  - `observed_device`: Actual execution hardware reported by backend telemetry or confirmed runtime inspect:
    - Prism: `telemetry.device` from model metadata / response stream (`"CUDA (GPU)"`, `"CPU"`).
    - Ollama: `/api/ps` processor info (`"GPU"`, `"CPU"`), or `model_info` execution providers.
    - ONNX Direct: query `og.is_cuda_available()` and provider registration.
    - When backend provides no device evidence: `observed_device = "unknown"`.
  - `cpu_fallback`: Boolean flag set to `True` when `requested_device` indicates GPU (`"cuda"`, `"metal"`, `"gpu"`) but `observed_device` is `"CPU"`.
- **Reporting:**
  - Leaderboard table in Markdown renders `Device / Provider` column showing observed device, and flags fallback visibly as `⚠️ CPU Fallback`.
  - CSV export appends `requested_device`, `observed_device`, `cpu_fallback` columns.
  - Updates `docs/runtimes.md`.

### 14.2 Timing breakdown and provenance (I12)
- **Independent Metrics:**
  - `ttft_sec`: Client wall time (seconds) from request dispatch until receipt of the first token chunk.
  - `prefill_duration_sec`: Duration spent purely evaluating the prompt tokens before token generation begins.
    - Ollama: `prompt_eval_duration` (converted from ns to seconds). `prefill_provenance = "engine"`.
    - Foundry / Prism / ONNX: If backend reports no separate prompt evaluation time, `prefill_duration_sec = None`, `prefill_provenance = "unavailable"`.
    - `prompt_tok_per_sec`: If `prefill_duration_sec` is available, `prompt_eval_count / prefill_duration_sec`. Otherwise, client effective prefill speed is reported separately as `prefill_eff_tok_sec = prompt_eval_count / ttft_sec` with `prefill_provenance = "client_ttft"`. TTFT is never labelled or stored as engine prefill duration.
  - `decode_duration_sec`: Duration spent in token generation phase (`end_wall_time - first_token_time`).
  - `eval_tok_per_sec`: Token generation speed. For generator loop clients (`OnnxGenAiClient`), `(eval_count - 1) / decode_duration_sec` when `eval_count > 1` and `decode_duration_sec > 0`, else `eval_count / total_time_sec`. For HTTP streaming clients (`FoundryClient`, `PrismClient`), `eval_count / eval_duration_sec` (preserving the Phase 10 floor measurement contract and regression test suite).
- **Reporting:**
  - Markdown report displays `TTFT (s)`, `Prefill (t/s, eff.)`, and `Decode (t/s)` distinctly.
  - CSV export includes `ttft_sec`, `prefill_duration_sec`, `decode_duration_sec`, `prefill_provenance`.
  - Updates `tests/test_measurement_methodology.py`, `tests/test_report_markdown.py`, `tests/test_report_csv.py`.

### 14.3 Peak RSS at fixed context (I13)
- **Process Memory Measurement:**
  - Measure host memory Resident Set Size (RSS) in MB using `HardwareSampler` sampling at `interval_sec`.
  - Track `peak_rss_mb` across the run.
  - `rss_coverage`:
    - `"client_only"`: Host memory sampled for the BenchRig process (`os.getpid()`) when runtime daemon PID is not discovered or resides across container boundaries.
    - `"client_and_server"`: Host memory sampled as the sum of client process and discovered backend server daemon process (e.g. Ollama daemon, Foundry daemon, Prism process).
- **Fixed Context Attribution:**
  - Memory measurements in run records are tagged with workload parameters: `context_tokens`, `prompt_tokens`, `max_output_tokens`.
  - Disclose that host RSS is separate from GPU VRAM / Apple Silicon UMA memory.
- **Reporting:**
  - Scorecards and JSON results include `peak_rss_mb`, `rss_coverage`, `context_tokens`.
  - Updates `benchrig/core/hardware.py`, `benchrig/core/runner.py`, `tests/test_hardware.py`, `docs/hardware-telemetry.md`.

### 14.4 Warm-up and KV reuse protocol (I14)
- **Execution Protocols:**
  - `cold_run`: First request after model loading. Measures cold-start latency (model loading, pipeline preparation).
  - `warm_run`: Steady-state evaluation with model loaded into memory, without assuming KV prefix cache reuse.
  - `prefix_cached`: Workload reusing a fixed prompt prefix on backends that support prefix caching.
- **Warm-up Flag & Aggregation:**
  - `--warmup-runs N` (default 1): Runs `N` warm-up iterations per scenario before benchmark measurement.
  - Warm-up runs are recorded with `phase="warmup"` and strictly excluded from benchmark summary averages (`avg_eval_tok_sec`, `avg_ttft_sec`, `composite_score`).
- **Cache Reuse Verification:**
  - `cache_mode`: Stamped as `"cold"`, `"warm"`, `"prefix_cached"`, or `"unverified"`.
  - A run is only labelled `"prefix_cached"` when verified via backend telemetry (an explicit boolean prefix cache hit/miss field in server telemetry; zero prompt count alone is insufficient). Unsupported or unconfirmed reuse is recorded as `"unverified"`.
- **Reporting:**
  - Updates `benchrig/cli.py`, `benchrig/core/runner.py`, `tests/test_measurement_methodology.py`, `docs/tutorials/cross-engine-benchmarking.md`.

### 14.5 Executable-test evidence and coding metrics (I15)
- **Disaggregated Coding Metrics:**
  - `coding_task_count`: Total coding scenarios evaluated.
  - `coding_tasks_passed`: Number of scenarios where 100% of unit test assertions passed.
  - `coding_task_pass_rate`: Percentage of fully solved tasks (`(coding_tasks_passed / coding_task_count) * 100`).
  - `coding_assertion_count`: Total unit test assertions executed across all scenarios.
  - `coding_assertions_passed`: Total unit test assertions that passed.
  - `coding_pass_rate`: Percentage of passed assertions (`(coding_assertions_passed / coding_assertion_count) * 100`, preserving backwards compatibility).
- **Per-Task Evidence:**
  - Every coding result record retains `task_id`, `passed` (task-level boolean), `passed_tests`, `total_tests`, `pass_ratio`, and failure diagnostics.
- **Reporting:**
  - Markdown report shows `Tasks Passed: X/Y (Z%)` alongside `Assertions: A/B (C%)`.
  - CSV export includes `coding_task_pass_rate`, `coding_tasks_passed`, `coding_task_count`, `coding_assertions_passed`, `coding_assertion_count`.
  - Updates `benchrig/core/runner.py`, `benchrig/reporting/`, `tests/test_report_markdown.py`, `tests/test_report_csv.py`, `docs/benchmark-suites.md`, `CHANGELOG.md`.

## Planned behavior: Phase 12 — Repo-quality hardening (Items 12.1, 12.2, 12.9, 12.10)

### 12.1 Sandbox isolation level and residual risk disclosure
- Module docstrings and method comments in `benchrig/core/sandbox.py`, CLI `--help`, and documentation define the sandbox accurately as "isolated subprocess and process-group execution" rather than OS-level containerization.
- Disclose residual risk: while processes run in dedicated process groups with strict timeouts (10s) and out-of-band JSON token verification, candidate code executes under the user's Python interpreter and can attempt system calls (`socket`, `subprocess`, `ctypes`).
- Architecture docs and CLI descriptions are aligned to use "isolated subprocess execution" consistently.

### 12.2 Coding suite language scope (Python-only)
- Explicitly document across `README.md`, `docs/benchmark-suites.md`, and a new sibling `benchrig/data/scenarios/coding.README.md` that the coding suite currently tests Python algorithms only.
- State clearly that multi-language execution (JS/TS, Rust, Go) is not bundled in default scenarios.

### 12.9 Continuous integration indicator
- Ensure `README.md` features clear indicators for GitHub Actions CI build status, linking directly to workflow runs.

### 12.10 Operational playbook migration
- Move the runtime-by-runtime operational debugging guide (§1–§4) from `AGENTS.md` into `docs/agent-debugging.md`.
- Register `docs/agent-debugging.md` in `mkdocs.yml` navigation under Project / Agent Debugging.
- In `AGENTS.md`, retain high-level instructions, mandatory gates, and a prominent link pointing coding agents to `docs/agent-debugging.md`.

### 12.6 Surface composite weights in Markdown report
- Render active composite scoring weights in the metadata header of `results/LATEST_SUMMARY.md`:
  `**Composite weights:** coding={w_code:.2f}, reasoning={w_reas:.2f}, performance={w_perf:.2f}  `
- Resolve weights from `config["benchmark"]["composite_weights"]` (or direct argument / scorecard), falling back to `DEFAULT_COMPOSITE_WEIGHTS` (`coding: 0.40, reasoning: 0.30, performance: 0.30`).
- Updates `benchrig/reporting/markdown.py`, `benchrig/core/runner.py`, `tests/test_report_markdown.py`.

### 12.8 Deduplication and dedent regression tests for code extraction
- Pin `extract_python_code` single-fence vs multi-fence dedent behavior with regression tests in `tests/test_sandbox.py`:
  1. Single ```python fence with uniform indentation margin is dedented so that unindented top-level statements do not raise `IndentationError`.
  2. Multiple fences where subsequent blocks are continuation fragments are NOT individually dedented (relative structure preserved).
  3. Plain text with zero fences returns stripped raw text.

### 12.3 Numeric-equivalence layer for reasoning ground truth
- `benchrig/core/reasoning_parser.py::evaluate_reasoning_answer` gains an optional keyword argument `evaluator: str | None = None`. When set, the function runs a normalization layer on the model's answer before delegating to the underlying `check_type`.
- `evaluate_reasoning_answer` also gains an optional `scenario_id: str | None = None` keyword argument, propagated by `BenchmarkRunner._evaluate_answer` from `sc.get("id")`. It is included in the `evaluator.unknown` structured-log event payload (see Failure modes).
- For both `evaluator` values, the model's answer is preprocessed the same way: if a `Final answer:` marker is present, the text after the marker is used; otherwise the full answer text is used. Leading and trailing whitespace is then trimmed. This means scenarios whose prompt asks for `Final answer: <integer>` are matched by both evaluators (the marker and reasoning prose don't trip the numeric parse).
- `evaluator` values (all opt-in per scenario; absent means behavior is exactly as today):
  - `"numeric"` — after the preprocessing above, if both sides parse as a `Fraction`, compare exactly; otherwise fall back to the underlying `check_type` (so `42` and `42.0` and `  42  ` all match `42`; word answers like `forty-two` do not silently match a numeric `42`).
  - `"numeric_text"` — same as `"numeric"` plus an English text-to-number conversion on both sides before the comparison. Supported forms: `zero`, `one`, ..., `nineteen`, `twenty`, `thirty`, ..., `ninety`, `hundred`, `thousand`, `million`, and hyphenated forms like `forty-two`, `twenty-one`. The conversion is case-sensitive: `Forty-Two` does NOT match `forty-two`.
- If only one side parses as a number (after normalization), the numeric layer is a no-op and the underlying `check_type` decides. This is the invariant that keeps word-puzzle answers (`expected_answer` is itself a word) from being silently accepted as numeric.
- `benchrig/core/runner.py::_evaluate_answer` passes `evaluator=sc.get("evaluator")` and `scenario_id=sc.get("id")` through to the evaluator.
- `benchrig/data/scenarios/reasoning.json` opts scenarios in by adding an `"evaluator"` field. Numeric scenarios (`reasoning_letter_count`, `reasoning_bat_ball`, `reasoning_multiples`, `reasoning_distinct_digits`, `reasoning_recurrence`, `reasoning_price_chain`, `reasoning_strawberry`) opt in to `"numeric"`; word-puzzle scenarios (the three-boxes regex) keep the current string/regex path. `polish.json`, `coding.json`, `speed.json`, and `context_scaling.json` are out of scope for this item.
- New tests live in `tests/test_reasoning_evaluator.py::NumericEquivalenceTests` (per `plan.md` Phase 12.3):
  - `42` matches `42`, `42.0`, and `  42  ` (whitespace-trimmed) when the scenario uses `evaluator: "numeric"`.
  - `forty-two` matches `42` only when the scenario uses `evaluator: "numeric_text"`.
  - `Forty-Two` does NOT match `forty-two` without an explicit case-insensitive opt-in (out of scope for this item).
  - Non-numeric ground truth (e.g. word puzzles, regex-based answers) keeps the current string/regex path unchanged.
  - `evaluator="numeric"` short-circuits to `correct=True` even when the legacy `check_type="final_answer"` path with empty `accepted_patterns` would reject (proves the numeric layer is on the critical path, not a no-op for `final_answer` scenarios).
  - `evaluator="numeric_text"` rejects `response="Final answer: Forty-Two"` against `expected_answer="42"` directly (not via the no-op fallback).

Failure modes:
- Unknown `evaluator` value: treated as `None` and emits a structured-log event `evaluator.unknown` with `scenario.id` (when provided) and `evaluator.value` (the rejected value). The numeric layer does not run; the underlying `check_type` decides. This is a typo-guard so a misspelled field does not silently disable comparison.
- Numeric layer is a no-op when the model's answer has no digits and the scenario's `expected_answer` has no digits (status unchanged, exit code unchanged). This is pinned by a test for both `evaluator="numeric"` and `evaluator="numeric_text"`.
- Numeric layer is a no-op when only one side parses as a number after normalization (status unchanged, exit code unchanged).

Normative doc updates: `docs/benchmark-suites.md` gains a short "Reasoning ground-truth comparison" subsection listing the supported normalizations and the per-scenario `evaluator` field; `CHANGELOG.md` gains a one-line entry under `[Unreleased]`.

### 12.4 Split `benchrig/cli.py` into per-command modules
- The current `benchrig/cli.py` (1028 lines) becomes a thin dispatcher; the implementation moves into a new `benchrig/cli/` package with one module per command group:
  - `benchrig/cli/__init__.py` — `build_parser()` (unchanged), `main()` (unchanged behavior, dispatches by arg), and **backward-compat re-exports** of every public symbol the tests currently import via `from benchrig.cli import X` or patch via `patch("benchrig.cli.X", ...)` (`run_benchmarks`, `run_system_check`, `pull_recommended_models`, `run_compare_mode`, `resolve_target_models`, `_resolve_log_level`, `_warn_slow_provider`, `_suite_skip_notice`, `_bootstrap_cuda_env`, `evaluate_model`, `build_scorecards`, `save_outputs`, `show_1to1_comparison`, `load_baseline`, `display_system_banner`, `display_leaderboard`, `display_token_savings`, `get_system_specs`, plus the module-level `console`, `glob`, `logging`, and `time` modules — verified by `grep "benchrig\\.cli" tests/`).
  - `benchrig/cli/_common.py` — module-level constants (`SUITES`, `RUNTIME_CHOICES`, `_VALID_LOG_LEVELS`, `CONFIG_FILENAME`, `CONFIG_ENV_VAR`, `BUNDLED_DATA`, `SCENARIOS_DIR`) and the cross-command helpers (`_resolve_log_level`, `resolve_config_path`, `load_config`, `resolve_scenarios_dir`, `load_scenario_file`, `_warn_slow_provider`, `_suite_skip_notice`, `_total_scenario_steps`, `load_json_or_exit`, `_bootstrap_cuda_env`).
  - `benchrig/cli/check.py` — `--check` and `--pull-recommended` paths (`run_system_check`, `pull_recommended_models`, `resolve_target_models`, `resolve_pair_targets`, `_installed_names`, `_recommended_models_for`, `_print_pull_progress`, `is_ollama`, `_check_ollama`, `_check_foundry`, `_check_prism`, `_check_onnx`, `_check_accelerator`).
  - `benchrig/cli/compare.py` — `--compare` path (`run_compare_mode`, `load_baseline`, `show_1to1_comparison`, `records_for`, `select_pair`, `_model_key`).
  - `benchrig/cli/run.py` — main benchmark orchestration (`run_benchmarks`, `evaluate_model`, `build_scorecards`).
  - `benchrig/cli/report.py` — output writers (`save_outputs`; the Markdown / CSV / chart producers from `benchrig.reporting.*` stay where they are).
- The single-file `benchrig/cli.py` is **replaced** by the package: `benchrig/cli.py` is removed and `benchrig/cli/__init__.py` becomes the new thin dispatcher. This is a package restructure, not a rename; `pyproject.toml`'s `benchrig = "benchrig.cli:main"` entry point keeps resolving unchanged because Python imports `benchrig.cli` (the package) instead of the old `benchrig/cli.py` file.
- `benchrig --help` renders **byte-identically** to today (the `argparse.ArgumentParser` configuration is moved as-is, not rewritten).
- `benchrig/__main__.py`'s `from benchrig.cli import main` keeps working because `main` is re-exported in `benchrig/cli/__init__.py`.
- Tests pinned in `tests/test_packaging.py` (`assertEqual(module, "benchrig.cli")` for the entry-point string) keep passing without changes.

Failure modes:
- New tests can patch either `benchrig.cli.X` (the re-export, current pattern) or `benchrig.cli.run.<name>` (the source module). Both work because command dispatch and shared-helper calls resolve at call time: an overridden package export takes precedence; otherwise the callable is read from its owning module.
- If a re-export is forgotten, a test that imports it directly via `from benchrig.cli import X` will raise `ImportError`. `tests/test_packaging.py` already pins the entry-point string; a new test in `tests/test_cli_dispatch.py` pins the re-exports by importing each symbol listed above.

Normative doc updates: none (no public CLI flag changes).


### 12.5 Installation documentation: PyPI first

Operator authorized the revised scope on 2026-10-04: defer Linux/WSL setup
scripts and make PyPI the primary installation path in README and the quickstart.
This is documentation only; product requirements and runtime behavior are unchanged.

- Main instructions for macOS Apple Silicon and Linux/WSL2 use Python 3.10+,
  a virtual environment, `python -m pip install benchrig`, and `benchrig --check`.
- Optional `benchrig[charts]` and `benchrig[onnx-gpu]` installation commands use
  the same environment. Describe direct ONNX as the optional NVIDIA CUDA path;
  runtime setup remains documented in the existing runtime guides.
- Repository cloning and `python -m pip install -e ".[dev]"` appear separately
  under development installation, requiring Python 3.11+ for the dev tools.
  This requirement does not raise the product's Python 3.10+ floor.
- The existing macOS setup script may remain linked as an optional source-install
  helper, with its scope distinguished from the development installation.
- No Linux/WSL setup scripts or setup-script tests are added. The previous
  script proposal is deferred; revisit only on a separate operator request.

Normative docs: `README.md` Quick Start and `docs/tutorials/quickstart.md` Step 1.
Verification: documentation diff review and the full SDLC gate. This documentation
change does not claim a fresh PyPI install or real-hardware runtime verification.


### 12.7 Historical analyses in documentation

Operator chose relocation on 2026-10-04 ("przeneisc"). Move the existing local,
Git-ignored `results/RUN_ANALYSIS.md` and `results/ANOMALY_VERIFICATION.md` into
`docs/analysis/2026-09-23/`, matching their recorded authorship date rather than
individual benchmark timestamps. Preserve findings, numbers and code excerpts.
Add a historical/manual-analysis notice explaining that recommendations and line
numbers describe the reviewed September state, not current runtime guarantees.

Add `docs/analysis/index.md` with links to both dated analyses and their provenance.
Register all three pages in MkDocs navigation and link the index from README and
the docs home. Add a relative link between the two analyses. Historical source
artifact paths remain code-formatted repository-relative provenance: ignored
JSON/CSV/PNG results are not bundled and must not become broken website links.
Update the active historical reference in plan.md and record the relocation in
CHANGELOG.md. No CLI flag, automatic analysis writer, anomaly detector, runtime
change, raw-result migration, or publication is part of this scope.

Verification: existing docs navigation coverage, Markdown link review, full SDLC
gate and independent read-only review. Source documents must be present before
moving; missing source files stop the relocation instead of inventing content.


## Planned behavior: Phase 13 — Tool-use benchmark

Status: implementation and additive I1 CSV schema change approved by operator
2026-10-04 ("zatwierdzam"), scope 13.1–13.4. Fits local-only benchmark intent; tool execution,
retrieval quality and changes to composite weights are outside scope.

### 13.1 Scenario contract

`tool_use.json` is a JSON array with unique `id`, `name`, `category`, `prompt`,
`tools` (OpenAI-style function schemas), `options`, `check_type: "tool_call"`, and
`turns`. Each turn has `expect` containing `tool` (name or null), `args` (object),
and, for null-tool expectations, mandatory `answer` (exact expected text after
trimming outer whitespace). A non-final tool turn has a literal `tool_result`
string. No filesystem, shell, network or retrieval function is actually executed.
The runner supplies this fixture only after a correct preceding call, using the
actual assistant call ID/name to construct protocol-appropriate tool messages.

Bundle at least 18 distinct scenarios: at least three in each category of tool
selection, typed argument extraction, abstention/plain answers, two-step calls,
Polish prompts and retrieval-shaped calls (`rag_search`, `read_file`, `grep`).
One expected call maximum per turn; additional/parallel calls fail in this release.
Schemas use object properties, required, additionalProperties=false, primitive
JSON types, enum and array items. Validate scenarios before any model request;
invalid custom scenarios produce a diagnostic naming file/id/field and CLI exit 2.
No new third-party schema dependency is required for this bounded schema subset.

Argument expectations are literal JSON values or an explicit matcher object
`{"type": "string", "contains": "text"}`. Literal object key order is irrelevant;
array order is significant, case/whitespace inside strings are preserved, extra
arguments fail, and there is no type coercion. Booleans are distinct from numbers;
integer arguments require integer JSON values (strings and float values fail).
Reject malformed expectations, undeclared tools and missing continuation results.

### 13.2 Client transport and result contract

Add `BaseRuntimeClient.chat_tools(model, messages, tools, options)` independently
of existing `generate()`. Initial release is **non-streaming**: Ollama `/api/chat`
and Foundry/Prism `/chat/completions` with stream=false. Reuse configured endpoint,
timeouts, auth, Foundry loading behavior and existing Prism 503 retry policy.
Normalize Ollama object arguments and OpenAI-compatible JSON-string arguments;
retain response content, raw calls, normalized calls and per-call parse errors.
Decode argument JSON strictly: reject non-object, duplicate keys, non-finite
numbers and malformed JSON. Never repair content JSON into structured calls.

Base/direct ONNX returns unsupported without generation because its current
adapter has no tool channel. Ollama, Foundry and Prism attempt native tools;
explicit capability rejection (recognized unsupported-tools response/metadata)
is unsupported. Generic HTTP 400/404, transport errors, malformed responses and
exhausted retries are errors, not unsupported. Record reason and HTTP status when
available. Stop later tool requests for a (model,runtime) only on confirmed
unsupported; emit unsupported records for all remaining scenario repetitions.
Do not suppress other suites. Persistent capacity errors retain existing behavior.

Result distinguishes transport status `ok|error|unsupported`, structured-channel
presence, valid/malformed calls, and `json_in_content` (whole content or one JSON
fence parses as a tool-call-shaped object/array). Text-only JSON never passes a
call-required turn. Refusal is recorded through failed expected-answer matching
and retained text, not inferred universally through a keyword heuristic.

Record request wall time and server usage/telemetry with existing estimation and
placement provenance rules. TTFT/decode duration/speed are unavailable (null)
when not reported by the server; never substitute total non-streaming latency.
Report tool-specific timing separately from legacy speed-suite statistics.
Implementation references: Ollama official tool-calling documentation
(https://docs.ollama.com/capabilities/tool-calling) and Foundry tool-calling guide
(https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-use-tool-calling-with-foundry-local),
checked 2026-10-04. Local Prism checkout referenced by older plans was unavailable;
verify deployed/local contract with fixtures before claiming live compatibility.

### 13.3 Runner, scoring and repetitions

`--suite tool_use` is opt-in; `--suite all` retains the existing five suites.
Use existing `--runs N` (default unchanged at 1); documentation recommends
`--runs 3`. Bundled scenarios explicitly set temperature=0 and seed=42 where
supported. Record effective settings; no guarantee of model determinism. A
nonzero-temperature variant is deferred to a separately specified experiment.
Each measured repetition starts a fresh conversation. Warm-ups use the same
conversation/fixture protocol, carry phase=warmup and are excluded from metrics.

A turn passes only on transport success, no truncation, exactly the expected
single structured call and matching typed arguments. Abstention passes only on
no structured call, no detected content-call JSON, and the required answer.
Stop a multi-step scenario after the first failed turn, marking later turns
not_run_due_to_prior_failure. Such turns cannot contribute successes. Record
all expected turn counts, attempted counts, per-turn evidence and first failure.
Scenario success requires every planned turn to pass. Progress advances once
per scenario repetition; warm-ups do not advance measured progress.

Aggregate per (model,runtime), with integer numerator/denominator alongside rates:
- task success: fully passed scenarios / eligible measured scenarios;
- selection accuracy: exactly correct single tool / planned call-required turns;
- argument accuracy: matching arguments / planned call-required turns (wrong
  tool, malformed call, error or skipped turn contributes no success);
- abstention accuracy: successful expected-answer no-call / planned null turns;
- false-call rate: null turns with any structured or detected content call /
  planned null turns; record errors/skipped counts alongside this diagnostic;
- structured-call rate: call-required turns emitting a nonempty tool_calls
  channel / planned call-required turns (malformed arguments remain observable).
Unsupported records are excluded from eligibility and counted separately.
Error and skipped turns remain in denominators to avoid inflating scores.
Zero denominator yields null/n/a, never 0%. Report partial/unsupported/error
counts so incomplete coverage cannot imply full benchmark success.

### 13.4 Reporting and compatibility

Add separate tool-use tables to terminal/Markdown with the rates above, pass
counts, coverage, errors, content-JSON observations, and mean request wall time
for completed successful requests. Preserve existing suite outcome rendering.
Append CSV columns: tool_status, tool_tasks_passed, tool_task_count,
tool_task_pass_rate, tool_selection_accuracy, tool_argument_accuracy,
tool_abstention_accuracy, tool_false_call_rate, tool_structured_call_rate,
tool_unsupported_count, tool_error_count, tool_request_latency_sec. JSON holds
all numerator/denominator pairs and per-turn evidence. Null rates produce empty
CSV cells and n/a Markdown/terminal cells. This additive I1 CSV contract change
was explicitly approved by operator 2026-10-04; existing columns stay in order.

Tool records do not enter existing coding/reasoning/performance composite inputs
or legacy aggregate timing/placement/memory fields. Tool-only scorecards expose
legacy scores/metrics as unavailable, not fabricated zero performance; existing
reporters must safely render null fields. Charts use separate tool-task-success
panels when tool evidence exists; unsupported/zero-denominator entries display
n/a, not zero-height failures. Existing composite panels remain unchanged when
no tool evidence is present; omit composite panel for tool-only results.

Normative updates: docs/scenarios.md, docs/benchmark-suites.md, docs/cli.md,
docs/runtimes.md, README.md and CHANGELOG.md. No new default suite execution,
composite weight, cloud traffic, external tool execution, or shipping action.

### 13.5 Live results and harness decision — deferred

Live model experiments are a separate operator-authorized step after hermetic
implementation/review. Before the run, agree on installed model/runtime pairs,
repetitions, temperature, coverage requirements and acceptance thresholds. Do
not infer production harness suitability from the earlier n=1 probe. A real
results note must report denominators/errors/unsupported pairs and platform/
runtime versions. No automatic change to dark-factory or rag repositories.


### CI Python 3.10 development dependency compatibility — 2026-10-04

Operator requested a fix for CI / test (3.10), failing during dependency install.
Keep the product/test matrix at Python 3.10–3.13. The development dependency on
`local-sdlc-kit` is selected only for Python >=3.11 using a PEP 508 environment
marker, matching the kit's interpreter requirement. Python 3.10 still installs
the remaining test/lint/build/chart dependencies and runs the same CI checks.
Full SDLC kit tooling requires a newer interpreter as documented; no gate or
matrix entry is removed. Regression evaluates the actual declared requirement
against Python 3.10 and 3.11 marker environments.


## Phase 16: Controlled runtime/prefix comparison

Implementation and cache classification semantics authorized 2026-10-04. Normative documentation:
`docs/tutorials/cross-engine-benchmarking.md` and `docs/cli.md`.
Implemented and independently reviewed; no remaining actionable findings.

Model/runtime initialization and prefix reuse are independent evidence axes.
New records expose `runtime_state` (`cold`, `warm`, `unverified`),
`prefix_cache_state` (`hit`, `miss`, `unverified`) and raw evidence/provenance.
Cold requires explicit backend initialization evidence (`runtime_initialized: true`)
for that request; positive load/setup duration alone does not establish initialization.
Warm requires a successful
runtime warm-up with no subsequently observed reload. Missing evidence is unverified.
A successful warm-up does not establish prefix cache absence. Neither request order,
zero prompt token count alone nor faster TTFT proves a cache hit. Legacy `cache_mode`
is derived conservatively: verified hit takes precedence, otherwise unverified
prefix state yields unverified; cold/warm requires verified prefix miss and runtime
state. Existing historical records are not reclassified.

Engine load, prefill and decode durations remain distinct from client TTFT and
startup request wall time. Absent timing is null with unavailable provenance, not
zero. Keep runtime-provided raw telemetry; direct ONNX request load timing is
unavailable when model loading occurs outside the measured request.

`--cache-probe` is opt-in, requires explicit `--suite speed`, one model and one
runtime; pair/baseline/compare/check/pull modes are incompatible. Invalid mode,
nonpositive runs or missing explicit positive `num_ctx` in the selected speed
scenario options fails with exit 2 before
model requests. Probe data is a separate record category, excluded from legacy
scorecards and score denominators. Existing normal CLI defaults remain unchanged.

After runtime warm-up, each pair submits the identical prompt twice, sequentially,
with identical effective context, generation options and session scope. Record
pair identity and position, prompt identity, runtime/model versions where available,
requested and observed context/cache type, durations and evidence. Do not reset
cache between requests or call the first request a miss. A pair is comparable only
when both requests succeed and context/cache type are verified unchanged; otherwise
report unverified with reasons, retaining both records. Runtime settings are observed,
not modified. Transport failures retain diagnostic records and do not imply unsupported
cache capability. Backends without cache evidence remain runnable but unverified.

Markdown renders paired evidence without combining these timings into legacy speed
or composite scores. No CSV header extension in this phase. Tests cover all runtime
adapters; live probes and published conclusions remain separately authorized work.

Phase 16 failure/artifact clarification: startup failures retain a diagnostic record
and produce the probe report. Probe artifacts use `runs/cache_probe_<timestamp>.json`,
`cache_probe_latest.json` and `CACHE_PROBE_SUMMARY.md`, preserving existing legacy
latest/summary artifacts. Native final server metrics are retained even without a
verified hit. TTFT without an observed first token is null/unavailable.
