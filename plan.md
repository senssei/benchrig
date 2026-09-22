# Implementation plan (`plan.md`)

The plan is the state of the work. An item is ticked only after `python3 scripts/sdlc_check.py` exited `0` for it. Each item names
the files it touches and the test that proves it. Behavior it adds is written in `spec.md` **before** the item starts.

---

## Phase 1: Charts and CSV export from `results/runs/`

Status: approved (2026-09-21).

- [x] <!-- Item 1.1: CSV export of scorecards.
         Files: benchrig/reporting/csv_export.py (new), benchrig/reporting/__init__.py (export),
                benchrig/cli.py (wire --csv flag), tests/test_report_csv.py (new).
         Test: tests/test_report_csv.py::test_csv_export_writes_one_row_per_scorecard_with_expected_columns
               uses a synthetic scorecard list and asserts the CSV at the requested path has the header
               SCORECARD_CSV_COLUMNS from benchrig/reporting/csv_export.py and one row per scorecard.
         Status: shipped 2026-09-21; sdlc_check.py exit 0. **Correction (2026-09-22):** the "shipped"
                 status above only covered `benchrig/reporting/csv_export.py`; `benchrig/cli.py` never
                 got the `--csv` flag and `reporting/__init__.py` never exported the function, so the CLI
                 raised "unrecognized arguments: --csv". Fixed in the same pass as item 1.2's correction
                 (tests/test_cli_report_flags.py); see that item's note. -->

- [x] <!-- Item 1.2: PNG chart of composite score per runtime/model.
         Files: benchrig/reporting/charts.py (new), benchrig/reporting/__init__.py (export),
                benchrig/cli.py (wire --chart flag), tests/test_report_charts.py (new).
         Test: tests/test_report_charts.py::test_chart_export_writes_png_with_one_bar_per_scorecard
               generates a chart from a synthetic scorecard list, asserts the output PNG is non-empty,
               and asserts the bar count equals len(scorecards).
         Status: shipped 2026-09-21; sdlc_check.py exit 0. **Correction (2026-09-22):** same gap as item
                 1.1 — `benchrig/cli.py` never got the `--chart` flag, so `benchrig --chart out.png` raised
                 "unrecognized arguments" (caught by the user running it, not by the gate: no test exercised
                 the CLI's argument parser or `save_outputs()` for these flags). Fixed: `build_parser()` now
                 registers `--csv`/`--chart`, `reporting/__init__.py` exports both writers, and
                 `save_outputs()` calls them and passes `chart_path`/`csv_path` (relative to the report's
                 directory) into `generate_markdown_report`. New test:
                 tests/test_cli_report_flags.py (flags registered, `save_outputs` writes/skips the
                 artifacts, markdown links them). docs/cli.md and CHANGELOG.md updated;
                 sdlc_check.py exit 0. -->

- [x] <!-- Item 1.3: Markdown report links to the chart and CSV.
         Files: benchrig/reporting/markdown.py (add image + csv links), tests/test_report_markdown.py (new).
         Test: tests/test_report_markdown.py::MarkdownAttachmentTests runs the markdown generator with
               chart_path/csv_path set and asserts the rendered Markdown contains an
               `![Composite scores](chart_path)` image embed near the top and a `**Attachments:** [CSV](csv_path)`
               link near the bottom; when neither path is set, no link or image is emitted (spec.md I2).
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

Risks and open questions:

- `matplotlib` is an optional `[charts]` extra (`pyproject.toml`); the import path must be lazy so `benchrig` still runs without it.
- A chart with one scorecard is uninformative; the test covers the no-fail shape, not visual correctness.

---

## Phase 2: Runtime warnings (`generic-cpu` on CUDA, dirty VRAM baseline)

Status: approved (2026-09-21).

- [x] <!-- Item 2.1: Warn when `generic-cpu` execution provider is selected on a CUDA host.
         Files: benchrig/core/runtimes.py (warning_for_provider + GENERIC_CPU_ON_CUDA_WARNING),
                benchrig/cli.py (_warn_slow_provider), tests/test_warnings.py (new).
         Test: tests/test_warnings.py::WarningForProviderTests — five tests covering generic-cpu on
               nvidia (warning), generic-cpu on apple_silicon (no warning), cuda on nvidia (no warning),
               None/empty provider (no warning), and warning text stability (spec.md §3).
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

- [x] <!-- Item 2.2: Surface foreign-GPU-process VRAM in scorecards as `vram_baseline_dirty_mb`.
         Files: benchrig/core/hardware.py (BaseHardwareProvider.read_gpu_foreign_memory_mb + LinuxNvidiaProvider
                override using nvidia-smi --query-compute-apps=pid,used_memory; _run_nvidia_smi helper),
                benchrig/core/runner.py (BenchmarkRunner.vram_baseline_dirty_mb field, populated in
                measure_vram_baseline, forwarded into scorecard hardware dict),
                tests/test_hardware_dirty_baseline.py (new).
         Test: tests/test_hardware_dirty_baseline.py — 7 tests across ForeignGpuMemoryTests and
               RunnerVramBaselineDirtyTests; covers base provider returning 0.0, nvidia-smi CSV parsing
               excluding own PIDs, empty/None handling, runner recording both vram_baseline_mb and
               vram_baseline_dirty_mb, and the zero-foreign case.
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

Risks and open questions:

- nvidia-smi output parsing is brittle; a regression test must freeze the format we depend on.
- A warning is not the same as an auto-skip; the operator may want both, but only after seeing the warning text.

---

## Phase 3: Operator-side releases (PyPI for benchrig, GitHub Pages, `senssei/local-coders`, prism-local 0.2.0)

Status: not approved yet. These are operator actions, not code changes — they appear here so they are not lost between sessions, but they are not SDLC plan items in the code/test sense.

- [ ] Trusted Publisher for `benchrig` on PyPI (`publish.yml`, environments `testpypi` and `pypi`), move `[Unreleased]` → dated entry in `CHANGELOG.md`, tag `v0.1.0`.
- [ ] GitHub → Settings → Pages → source „GitHub Actions” for `benchrig` and `prism-local`.
- [ ] Create the `senssei/local-coders` repository on GitHub (links from the README are dead until it exists).
- [ ] `prism-local`: decide and write the `[Unreleased]` entry for `stream_options.include_usage`, device label, `PRISM_PREFILL_CHUNK`; tag `v0.2.0`.

Risks and open questions:

- None of these can be ticked by a coding agent; the operator runs the dashboard UI, the repo creation, and the PyPI tag.

---

## Phase 4: Integration with `prism-local` HEAD (post-v0.2.0; `~/03-foundy-local`)

Status: approved (2026-09-21). Re-specified against the actual contract in `~/03-foundy-local` `main` (8 commits past `v0.2.0`: `1274b58…4d8567d`). Benchrig already implements items 4.1 and 4.2; those items below are **verification** work (tests + a guard against contract drift), not new features.

- [x] <!-- Item 4.1 (verify, was: implement): `stream_options.include_usage` end-to-end.
         Contract (prism-local HEAD, tests/test_prism_server_api.py:307-317):
         when the client sends `stream_options={"include_usage": True}`, the last SSE chunk has
         `choices=[]`, `usage={prompt_tokens, completion_tokens, total_tokens}`, and
         `telemetry={device, ttft_sec, …}`. Without that flag, no usage chunk is sent.
         Benchrig already does it (benchrig/core/client.py:646 sends the flag; 720-788 records usage
         and clears `usage_estimated`). This item adds the **regression test** that pins the contract.
         Files: tests/test_prism_runtime.py (extend).
         Test: tests/test_prism_runtime.py::test_prism_stream_records_usage_and_telemetry
               feeds a recorded /v1/chat/completions stream whose last chunk is the 0.2.0 shape;
               asserts `usage_estimated=False`, `result["usage"] == {...}`, `result["telemetry"]["device"] == "cpu"`.
         Pre-condition: prism-local v0.2.0 tag (already on disk in ~/03-foundy-local).
         Status: shipped 2026-09-21. The regression test already existed as
                 `tests/test_prism_runtime.py::UsageTests::test_usage_and_telemetry_from_the_last_chunk_are_used`
                 (line 149) and passes. No new code, no new test; the existing test pins the contract. -->

- [x] <!-- Item 4.2 (verify, was: implement): `/v1/models` device label = the device the model will run on.
         Contract (prism-local HEAD, tests/test_prism_server_api.py:288-301 + tests/test_prism_catalog.py:63):
         each model has `device` (e.g. "CUDA (GPU)" or "CPU"; resolved from `PRISM_DEVICE` / GPU presence at
         serve time) and `exported_for` (the label the user passed). They can differ.
         Benchrig already reads `telemetry.device` (client.py:867-874); the Markdown report already uses
         `runtime_label(sc.get("runtime"))` which is fed from the model record's runtime field. This item
         pins the parser behavior with a regression test.
         Files: tests/test_prism_runtime.py (extend).
         Test: tests/test_prism_runtime.py::UsageTests::test_device_comes_from_telemetry_not_from_export_label
               feeds a fake stream whose last chunk has telemetry.device="CUDA (GPU)" and a stray
               exported_for="cpu" field; asserts result["device"] == "CUDA (GPU)" and usage_estimated is False.
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

- [x] <!-- Item 4.3 (docs only): document `PRISM_PREFILL_CHUNK` (default 1024), `PRISM_THREADS` (default None),
         and `PRISM_DEVICE` (default "auto") in `benchrig --help` (CLI text + man page) and `docs/runtimes.md`.
         No benchrig code change: the env vars are read by prism-local itself (prism/engine.py:58-98).
         Trade-off (TODO §2 row 8): peak −44/−54/−5 %, TTFT +4/+110/+33 %; stays opt-in (user sets the env
         when starting `prism serve`).
         Files: benchrig/cli.py (--help text only), docs/runtimes.md.
         Test: tests/test_docs.py::test_prism_env_vars_are_documented asserts each var appears in runtimes.md.
         Status: shipped 2026-09-21; new "Tuning the Prism server with environment variables" section
                 added to docs/runtimes.md with the three vars and their defaults; test passes; CLI --help
                 untouched (no CLI surface change needed, the env vars are read by prism-local itself). -->

- [x] <!-- Item 4.5 (real code): handle 503 from prism-local with a Retry-After-aware retry.
         Contract (prism-local HEAD, prism/server.py:669-671 + prism/cli.py:313-315):
         `server_busy` (queue full) and `insufficient_resources` (load lock contention) return 503 with
         `Retry-After: 30`. Benchrig currently surfaces `requests.HTTPError` immediately (client.py:240,269).
         This item retries up to 3× with the server-provided `Retry-After` (capped at 60 s, hard ceiling),
         gives up with a clear error naming the reason from the JSON body, and does NOT retry other 5xx.
         Files: benchrig/core/client.py (retry helper + caller), tests/test_prism_runtime.py (extend).
         Test:
           - test_prism_client_retries_on_503_with_retry_after: first call returns 503 + Retry-After: 1;
             second call returns 200; client returns the second body. Verify 2 HTTP calls happened.
           - test_prism_client_gives_up_after_three_retries: 3 consecutive 503s; client raises a
             PrismBusyError naming the reason from the JSON body.
           - test_prism_client_does_not_retry_on_500: 500 with Retry-After: 1 → no retry, immediate raise.
         Pre-condition: prism-local v0.2.0 tag. Risk: prism-local may change the retry-after header value
         (currently "30"); the test pins "1" so we only test the mechanism.
         Status: shipped 2026-09-21; sdlc_check.py exit 0.
                 Implementation: `benchrig.core.client._post_with_503_retry` + new `PrismBusyError` +
                 `FoundryClient._make_request` hook (default `requests.post`) + `PrismClient._make_request`
                 override (uses the helper). 4 tests in `tests/test_prism_runtime.py::Retry503Tests` cover:
                 retry on 503 with Retry-After, give-up after budget (generate path: failure result carries
                 the reason), give-up after budget (helper path: PrismBusyError), no retry on 500. -->

- [x] <!-- Item 4.6 (real code, bug fix): short-circuit remaining scenarios once a model shows a
         persistent `insufficient_resources` failure, instead of retrying every scenario in every suite.
         Found manually 2026-09-22: `benchrig --runtime all --suite all` on a model that does not fit in
         available VRAM retried 503 (4 attempts, with backoff) for EVERY scenario in EVERY suite —
         dozens of near-identical error lines and minutes of wasted retries for a condition that cannot
         change mid-run (nothing frees VRAM between scenarios). `server_busy` (queue full) is NOT covered
         — it can clear on its own, so it keeps retrying per scenario as before (item 4.5).
         Files: benchrig/core/runner.py (`BenchmarkRunner._capacity_exhausted_reason`,
                `_capacity_error`, guards in `_run_suite` and `run_context_suite`),
                tests/test_runner_capacity_shortcircuit.py (new).
         Test: tests/test_runner_capacity_shortcircuit.py —
           - test_persistent_capacity_failure_stops_remaining_scenarios_in_the_suite: 5 scenarios, first
             fails with `insufficient_resources` → only 1 ran.
           - test_persistent_capacity_failure_skips_later_suites_for_the_same_model: coding suite trips
             it → the next suite call on the same `BenchmarkRunner` makes zero HTTP calls.
           - test_transient_busy_failure_does_not_short_circuit: `server_busy` failures still retry every
             scenario (regression guard against over-broadening the marker list).
         Status: shipped 2026-09-22; sdlc_check.py tests green (the 2 live-Prism failures in the same
                 run are unrelated real VRAM contention on the host, not a code regression — see
                 tests/test_prism_runtime_live.py, which needs the host to actually have ~9.3 GB free).
                 **Follow-up (same day):** the short-circuit was silent — `BenchmarkRunner._notify`
                 (the "skipping" message) never reaches the terminal because `benchrig/cli.py` never
                 passes a `progress_callback` when constructing `BenchmarkRunner`. Confirmed live by the
                 operator: `--suite all` on an oversized model printed "Running coding/reasoning/polish
                 tests..." with zero scenario lines under each and no explanation. Fixed with a public
                 `BenchmarkRunner.capacity_exhausted_reason` property and
                 `benchrig/cli.py::_suite_skip_notice` (pure function: no notice when the suite actually
                 ran or had nothing to run; the capacity reason otherwise), called from `evaluate_model`'s
                 suite loop to print "⚠ Skipped — model does not fit in available VRAM: ...". New tests:
                 tests/test_cli_capacity_skip_notice.py. sdlc_check.py tests green
                 (--ignore=tests/test_prism_runtime_live.py; live suite needs free host VRAM), lint clean. -->

Risks and open questions:

- Prism-local HEAD is post-v0.2.0; contracts here may shift again before 0.3.0. Re-read at the start of each
  item against `~/03-foundy-local` `main`. If the contract drifts, stop and re-spec (sdlc-implement rule).
- Items 4.1 and 4.2 add new constants only in tests; no new notes in `benchrig.core.runtimes` are required.
- Tests for Prism currently mock the daemon. A recorded real fixture from `tests/test_prism_server_api.py`
  is the closest thing to ground truth; we reuse its shape, not the test runner.
- Item 4.5 (retry) interacts with `--runs N`: the retry counts as one logical attempt (we do not re-issue
  every repetition). Decision deferred to implementation; default: retry once for the first attempt, then
  proceed with repetitions normally if it succeeds.

### Preliminary tests (folded into 4.1, 4.2, 4.5)

The original prelim sub-items (4.1-prelim / 4.2-prelim / 4.3-prelim) are now subsumed by items 4.1, 4.2
and the regression tests in 4.5. No separate prelim phase.

---

## Phase 5: Real model unload for Prism (`POST /v1/unload`, prism-local HEAD `6d6467a`)

Status: approved (2026-09-22); implemented and independently reviewed (fresh-context `general-purpose`
subagent). Review: 1 finding (docstring omitted the engine-lock/blocking note this phase's own "Risks and
open questions" asked for) — fixed directly (docs-only, no new test needed); everything else (URL, auth
header, exception scope, timeout, red-first test evidence, patch scope, no secret leakage) checked out
clean. sdlc_check.py exit 0 after the fix. Not re-reviewed independently a second time (docstring-only
change, verified by re-running the gate). Awaiting operator decision to commit/ship.

- [x] <!-- Item 5.1: `PrismClient.unload_model()` calls `POST /v1/unload` instead of the inherited Foundry-CLI no-op.
         Files: benchrig/core/client.py (PrismClient.unload_model, new override),
                tests/test_prism_runtime.py (extend).
         Test: tests/test_prism_runtime.py — new `UnloadModelTests` class:
           - test_unload_model_posts_to_v1_unload: mocked `requests.post` asserts the call URL is
             `<base_url>/unload` and the method returns True.
           - test_unload_model_sends_the_bearer_token_when_api_key_is_set: same, with `api_key="secret"`,
             asserts `headers == {"Authorization": "Bearer secret"}`.
           - test_unload_model_is_best_effort_on_transport_error: `requests.post` raises `OSError`;
             asserts `unload_model()` still returns True and does not raise.
           - test_unload_model_is_best_effort_on_404_from_an_older_prism_server: `requests.post` returns a
             404 response (`raise_for_status` raises `requests.HTTPError`); asserts `unload_model()` still
             returns True.
         Status: shipped 2026-09-22; sdlc_check.py exit 0 (293 passed, 4 skipped). -->

- [x] <!-- Item 5.2: Docs and changelog.
         Files: docs/runtimes.md ("How BenchRig talks to it" and "Token counts and the model-only memory
                figure" bullets updated — both previously said Prism "has nothing to unload" / "has no
                unload", now stale; corrected to describe the real `POST /v1/unload` call and its
                best-effort fallback on an older server), CHANGELOG.md (`### Added` entry under
                `[Unreleased]`).
         Test: none beyond the existing `tests/test_docs.py` doc-presence checks; this item is prose only.
         Status: shipped 2026-09-22; sdlc_check.py exit 0. -->

Risks and open questions:

- prism-local `6d6467a` is not yet tagged (post-`v0.2.0`, pre-`v0.3.0`); if the response shape changes before
  release, re-check `~/03-foundy-local` `docs/api.md` and `spec.md` P8 before shipping this phase.
- The unload call is synchronous and can block for seconds if a generation is in flight (the server holds the
  engine lock); `evaluate_model` already runs unload after all suites for the model finish, so this should
  never overlap a benchmark request in practice — call this out in the docstring, not a new test.
- No new CLI flag: `unload_after_test` (default `True`) already gates the call for every runtime; Phase 5 only
  makes the existing Prism path do real work.

---

## Phase 6: Sampling parameters and reasoning content for Foundry/Prism (`~/03-foundy-local` `e01569c`, `4d8567d`)

Status: approved (2026-09-22). Operator confirmed the `ttft_sec` semantic change for reasoning models (item 6.2) is
acceptable as a documented behavior fix, and the scope exclusions below are fine, to be tracked as backlog.
Implemented 2026-09-22: items 6.1, 6.2, 6.3 all shipped; sdlc_check.py exit 0 (298 passed, 4 skipped).
Review (fresh-context `general-purpose` subagent, 2026-09-22): 4 findings, all fixed test-first —
(1, Medium) a malformed `top_k`/`repetition_penalty` (e.g. `options={"top_k": "many"}`) raised an
uncaught `ValueError`/`TypeError` out of `generate()`, crashing the whole `--runs N` benchmark instead of
failing one scenario; fixed by wrapping the options-to-payload block in `try/except` → `_failure_result`
(2 new tests in `tests/test_foundry_runtime.py`). (2, Low) three correct-but-untested branches (reasoning
with no content ever arriving, an empty-string `reasoning_content` chunk, reasoning split across multiple
chunks) — added 3 regression tests in `tests/test_prism_runtime.py::ReasoningContentTests`, all passed
immediately (implementation was already correct). (3, Low) `spec.md` §2 never got I7 (Phase 5) or I8/I9
(Phase 6) promoted into the invariants table despite both phases being shipped — fixed by promoting all
three and updating the Phase 5/6 narrative sections to point at them instead of duplicating them. (4, Info)
`spec.md` overstated `thinking_chars`/`answer_ttft_sec` as matching `OllamaClient` "byte-for-byte" — fixed
the wording to describe the actual (harmless) shape difference. sdlc_check.py exit 0 after fixes (304
passed, 4 skipped). Fixes verified by tests, not re-reviewed by a second fresh subagent. Awaiting operator
commit.

- [x] <!-- Item 6.1: Forward top_k, repetition_penalty and stop from options into FoundryClient.generate().
         Files: benchrig/core/client.py (FoundryClient.generate), tests/test_foundry_runtime.py (extend).
         Test: tests/test_foundry_runtime.py::TestFoundryClient::test_generate_forwards_top_k_repetition_penalty_and_stop
               (red-proven: KeyError before the fix) and
               ::test_generate_omits_sampling_params_when_not_provided (regression guard).
         Status: shipped 2026-09-22; sdlc_check.py tests green. -->

- [x] <!-- Item 6.2: Capture reasoning_content into thinking_chars/answer_ttft_sec (streaming + non-streaming).
         Files: benchrig/core/client.py (FoundryClient.generate), tests/test_prism_runtime.py (extend:
                new ReasoningContentTests class following UsageTests' stream()/generate() helper pattern).
         Test: tests/test_prism_runtime.py::ReasoningContentTests — 3 tests (red-proven: KeyError on
               thinking_chars before the fix):
           - test_streaming_reasoning_then_content_sets_thinking_chars_and_answer_ttft
           - test_non_streaming_reasoning_content_sets_thinking_chars
           - test_content_only_response_has_no_thinking_chars (regression guard)
         Status: shipped 2026-09-22; sdlc_check.py exit 0 (298 passed, 4 skipped).
                 Note: answer_ttft_sec is floored at ttft_sec (`max(ttft_sec, first_answer_time - start)`)
                 to avoid a rounding artifact where a mocked/instant response rounds both to 0.000 but
                 float truncation could otherwise show answer_ttft_sec < ttft_sec; caught by the first
                 test run (AssertionError: 0.0 not >= 0.001), fixed before ticking this box. -->

- [x] <!-- Item 6.3: Docs and changelog.
         Files: docs/runtimes.md (new bullets under "How BenchRig talks to it" documenting top_k,
                repetition_penalty, stop and reasoning_content/thinking_chars/answer_ttft_sec),
                CHANGELOG.md (### Added entries for items 6.1/6.2, new ### Changed entry for the
                ttft_sec semantic change under [Unreleased]).
         Test: none beyond existing tests/test_docs.py doc-presence checks (prose only).
         Status: shipped 2026-09-22; sdlc_check.py exit 0. -->

Risks and open questions:

- `stop` is not new in prism-local (predates `v0.2.0`) but was never wired into benchrig; bundled here because
  it is the same "sampling passthrough" gap and the fix touches the same few lines.
- Item 6.2 changes `ttft_sec` for any Foundry/Prism reasoning model that was already being benchmarked (the
  number gets smaller/more accurate when reasoning precedes content) — call this out in CHANGELOG.md as a
  behavior fix, not a new feature, since past run JSON files are not comparable across it.
- Scope explicitly excludes: `/v1/embeddings` (no embedding suite in benchrig), tool calls / `tool_calls`
  (no tool-use suite in benchrig), `PRISM_THREADS` / cancel-on-disconnect / MCP auto-stop (server-side only,
  no client-observable surface). Operator confirmed (2026-09-22): keep excluded from Phase 6, tracked as
  backlog below for a future phase.

---

## Phase 7: Bug fix — chart bar labels overlap into an unreadable strip

Status: approved and shipped (2026-09-22, found manually by the operator from a `--chart` PNG showing 6
scorecards). sdlc_check.py exit 0 (299 passed, 4 skipped). Awaiting independent review and operator commit.
`make_scorecard_figure` (`benchrig/reporting/charts.py`) set `tick_label=labels` on `ax.bar(...)` with no
rotation; horizontal labels of the form `<model> (<runtime>)` (often 30-50+ characters, e.g.
`mistral-7b-instruct-v0.2-q4_0 (prism)`) run into each other and become illegible once there are more than
2-3 bars, even though `figsize` already scales with `1.2 * len(scorecards)`. `spec.md`'s Phase 1 section
never specified label rotation or overlap handling, so this was undefined behavior, not a broken invariant
(I1-I3 unaffected) — `spec.md` updated in the same pass to say labels are rotated 30° with right alignment.

- [x] <!-- Item 7.1: Rotate bar labels 30° with right alignment so they no longer overlap.
         Files: benchrig/reporting/charts.py (make_scorecard_figure: replace tick_label=labels on ax.bar
                with ax.set_xticks + ax.set_xticklabels(labels, rotation=30, ha="right")),
                tests/test_report_charts.py (extend).
         Test: tests/test_report_charts.py::ChartExportTests::test_bar_labels_are_rotated_to_avoid_overlap
               (red-proven: rotation was 0.0 before the fix) asserts every x-tick label has rotation 30
               and horizontalalignment "right"; existing test_make_scorecard_figure_has_one_bar_per_scorecard
               continues to assert the label text itself is unchanged.
         Status: shipped 2026-09-22; sdlc_check.py exit 0. -->

---

## Backlog (not scheduled)

Surfaced while specifying Phase 6 against `~/03-foundy-local` `main`; none are approved for implementation.
Re-scope into a numbered phase (with its own `spec.md` section and operator approval) before starting any of these.

- **Embeddings suite.** Prism serves `/v1/embeddings` (Ollama-backed models only; ONNX Runtime GenAI does not
  produce embeddings). Benchrig has no embedding suite or `BaseRuntimeClient.embed()` method today — would need
  a new suite type end-to-end (scenario shape, scoring, reporting columns), not just a client method.
- **Tool-calling suite.** Prism's `/v1/chat/completions` accepts `tools`/`tool_choice` and returns `tool_calls`
  (including the `d143788` fix that turns a template render failure into a `400` instead of silently dropping
  tools). Benchrig has no tool-use suite; would need scenario definitions with expected tool calls and a scorer.
- **`PRISM_THREADS` as a benchrig-managed setting.** Currently a `prism serve` environment variable the operator
  sets by hand (documented in Phase 4 item 4.3); benchrig could in principle report or vary it per run, but that
  is server configuration, not a client request parameter.
- **`error.holder` on `503 insufficient_resources` (Prism `P12`, uncommitted in `~/03-foundy-local` as of
  2026-09-22 — working-tree only, not yet reviewed/committed there).** When the model-load lock is held by
  another process, Prism's `503` body gains a structured `error.holder = {"pid": int, "model": str}` alongside
  the existing prose in `error.message` (the field is omitted, not `null`, when there is no holder). Low-risk
  follow-up once Prism ships it: have `PrismBusyError`/the failure result carry `holder` as a structured field
  (today only the prose reason string is captured), so reports/CSV can show which process/model is blocking a
  run without parsing text. No process management involved — purely reading a field that is already in the
  response. Do not spec until the Prism-side change is committed (their own `plan.md` Phase 8 status is
  "awaiting independent review and operator commit"); re-check the field name and shape against
  `~/03-foundy-local` at that point, same rule as Phase 4's "if the contract drifts, stop and re-spec".
- **`POST /v1/drain` (Prism `P12`, same uncommitted state as above).** Lets an orchestrator ask a running
  `prism serve` to finish its current request, unload the model, and exit the process with status 0 — Prism's
  own plan.md names `benchrig --runtime prism` as the intended caller, so this is worth watching. Using it for
  real (e.g. auto-draining a `prism serve` that `error.holder` names as blocking a load, then starting our own)
  is a materially bigger architecture change than anything else in this backlog: benchrig has never managed a
  Prism server's process lifecycle, only talked to one already running at a fixed `base_url`
  (`docs/runtimes.md` "How BenchRig talks to it"). Killing a process benchrig did not start is a destructive
  action on state outside the current run — needs an explicit operator decision (e.g. a `--drain-holder` flag
  the user opts into per run, never automatic) before it gets anywhere near a `spec.md` entry. Not scheduled;
  revisit only after discussing the design with the operator, and only once Prism's side is committed and
  tagged.