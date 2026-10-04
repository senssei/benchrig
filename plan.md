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

## Phase 8: Real-time per-scenario feedback and an overall progress bar

Status: shipped, retroactively documented (2026-09-22). Requested by the operator ("dodaj do pipeline jakiś
progressbar") after a long `reasoning` suite gave no terminal output for its whole duration and then dumped every
`PASS`/`FAIL` line at once — `_run_suite`/`run_context_suite` built the full suite result list before
`benchrig/cli.py::evaluate_model` printed any of it. This item was implemented directly (skipping `/sdlc-plan`,
against `CLAUDE.md`); the operator asked for it to be written up here after the fact rather than reverted, since it
was already green. `spec.md` Phase 8 has the full before/after. Gate run manually: `ruff check .`, `ruff format
--check .`, full `pytest` (305 passed, 2 skipped, 2 deselected — see note below), `python3 -m build`, all green.
Review (2026-09-22, together with Phase 9, one independent general-purpose agent, fresh context): 1 finding
(missing test coverage for item 8.2), fixed — see item 8.2 below. sdlc_check.py exit 0 after the fix.

- [x] <!-- Item 8.1: BenchmarkRunner.on_result callback, fired per scenario record.
         Files: benchrig/core/runner.py (__init__: new on_result param; _run_suite and run_context_suite call
                self.on_result(record) right after results.append(record)),
                tests/test_runner_suites.py (RunnerSuiteTests.test_on_result_fires_per_scenario_before_the_suite_finishes).
         Test: tests/test_runner_suites.py::RunnerSuiteTests::test_on_result_fires_per_scenario_before_the_suite_finishes
               (red-proven: HEAD's BenchmarkRunner.__init__ has no on_result parameter at all, so constructing it
               with on_result=... raises TypeError) asserts on_result sees each record, in scenario order, before
               run_speed_suite returns, and that the seen list equals the returned results list.
         Status: shipped 2026-09-22. -->
- [x] <!-- Item 8.2: Real-time per-scenario printing + an overall rich.progress.Progress bar in the CLI.
         Files: benchrig/cli.py (evaluate_model: new on_progress param, on_result closure wired into
                BenchmarkRunner(on_result=...) that calls display_scenario_result immediately instead of the old
                post-suite `for r in suite_results: display_scenario_result(r)` loop; run_benchmarks: wraps the
                per-target loop in a rich.progress.Progress bar sized via the new _total_scenario_steps() helper
                — extracted during review so the arithmetic is directly testable — advanced via on_progress;
                tests/test_benchmark_cli.py: TotalScenarioStepsTests, EvaluateModelProgressTests).
         Test: tests/test_benchmark_cli.py::TotalScenarioStepsTests::test_counts_one_unit_per_target_run_and_scenario,
               ::test_zero_scenarios_is_zero_not_a_crash, and
               EvaluateModelProgressTests::test_on_progress_fires_once_per_scenario_across_suites_and_runs (all
               red-proven: neither _total_scenario_steps nor evaluate_model's on_progress param existed at HEAD
               — "not found"/would-be TypeError); ::test_on_progress_is_optional is a guard, not red-proven (it
               also passes with on_progress omitted at HEAD, since that param is additive).
         Status: shipped 2026-09-22. Independent review (general-purpose agent, fresh context) flagged this item
               had zero test coverage; fixed by extracting _total_scenario_steps and adding the tests above.
               sdlc_check.py exit 0. Re-verified by tests only, not re-sent to the reviewer agent. -->

**Note on the deselected live tests:** `tests/test_prism_runtime_live.py::PrismRuntimeLiveTests::test_device_comes_from_telemetry_not_from_exported_for`
and `::PrismLoadLockLiveTests::test_concurrent_loads_trigger_503_with_retry_after` fail regardless of this change —
confirmed by running them against `HEAD` (via `git stash`) before this item started: first a VRAM-contention 503
(another process holding the load lock), then `Connection refused` once the operator's manually-started
`prism serve --port 0` (ephemeral port, not the tests' hardcoded 5272) was no longer reachable at that port. Pre-existing
environment/live-server flakiness, unrelated to items 8.1/8.2; not fixed here.

---

## Phase 9: Coding-suite failure diagnostics (from a flagged 0-passed run)

Status: approved by the operator (2026-09-22, both 9.1 and 9.2; the other two open questions below —
retry/determinism-warning design and re-running the old flagged benchmark — were not addressed and stay open,
not implemented). Items 9.1 and 9.2 both shipped 2026-09-22. Review (2026-09-22, together with Phase 8, one
independent general-purpose agent, fresh context): 4 findings — 1 high (9.2's dedent broke a multi-fence
class+method-continuation pattern: a real, demonstrated regression, not a false alarm), 2 medium (9.1's
response_excerpt used the wrong truncation direction for coding; item 8.2 had no test coverage — recorded
under Phase 8), 1 low (9.1's own test never exercised the truncation branch it claimed to prove) — all 4
fixed, each reproduced with its own test first. Awaiting operator commit; the fixes were verified by tests
only, not re-sent to the reviewer agent for a second pass. Triggered by the operator flagging that in
`results/runs/runs/benchmark_20260922_214037.json`, `Phi-4-mini-instruct-cuda-gpu`,
`Phi-4-mini-instruct-generic-cpu-5:v5` and `Phi-3.5-mini-instruct-generic-cpu-2:v2` scored 0 `passed_tests` on every
`coding` scenario (`IndentationError`/`SyntaxError`), which is implausible for that model class. Full investigation
and both competing hypotheses are in `spec.md` Phase 9. Summary: the leading hypothesis (an indentation/dedent bug
in `extract_python_code`, `benchrig/core/sandbox.py:17-31`) failed to reproduce 15/15 times against the exact
failing `(model, scenario, options)` triples on the live Prism server this session — so **item 9.2 below is a
hardening measure, not a confirmed bug fix**, and must not be described as "fixing" that run's failures. The more
evidence-consistent alternative — `execution_alignment_1to1.seed: 42` does not make the `coding`/`reasoning` suites
actually deterministic on ONNX Runtime GenAI + CUDA under GPU-load conditions this session's replay did not
reproduce, which would mean intent.md Constraints 4 and 5 are not currently upheld for these suites — has **no
captured evidence either** (the original run kept no raw response), which is exactly what item 9.1 fixes going
forward.

- [x] <!-- Item 9.1: Persist coding-suite failure diagnostics so the next 0-passed run is debuggable without live
         reproduction.
         Files: benchrig/core/runner.py (run_coding_suite's score(): when test_res["passed"] is False, add
                extracted_code (read from test_res, already computed by run_code_with_tests/_sandbox_result,
                truncated to sandbox.EXTRACTED_CODE_PREVIEW_CHARS) and response_excerpt (first
                CODING_RESPONSE_EXCERPT_CHARS=400 chars of the raw response — corrected during review, see
                below) to the returned dict), tests/test_runner_suites.py (three tests next to
                test_coding_suite_scores_sandbox_result).
         Test: tests/test_runner_suites.py::RunnerSuiteTests::test_coding_suite_failure_keeps_extracted_code_and_response_for_diagnosis
               (red-proven: KeyError 'extracted_code', the field did not exist),
               ::test_coding_suite_success_omits_diagnostic_fields (guards against bloating every record), and
               ::test_coding_suite_response_excerpt_keeps_the_code_over_long_trailing_prose (red-proven against
               the first version of this fix, see below).
         Status: shipped 2026-09-22; sdlc_check.py exit 0. Independent review (general-purpose agent, fresh
               context) found response_excerpt copied reasoning's tail-truncation shape (last 400 chars), which
               is wrong for coding: the code fence comes first in the prompt's own instructions, so long
               trailing prose after the fence could push the code itself out of the excerpt — exactly the
               material a future diagnosis needs. Fixed to head-truncation (first 400 chars, new
               CODING_RESPONSE_EXCERPT_CHARS constant instead of a bare literal, also flagged by the review).
               Re-verified by tests only, not re-sent to the reviewer agent. -->
- [x] <!-- Item 9.2 (hardening, not a confirmed fix — see status note above): dedent a fenced code block in
         extract_python_code before stripping/joining, so a model that wraps a single ```python fence inside
         indented markdown (numbered lists, nested bullets) is not miss-extracted. Defense-in-depth for a real
         latent gap; not shown to be the cause of the flagged run's failures.
         Files: benchrig/core/sandbox.py (extract_python_code: apply textwrap.dedent() ONLY when there is
                exactly one matched fence (`len(matches) == 1`), before the `code_blocks_with_def` filter and
                before the final .strip()/join — corrected during independent review, see below),
                tests/test_sandbox.py (ExtractCodeTests::test_dedents_a_fence_indented_under_a_markdown_list,
                ::test_does_not_dedent_a_continuation_fence_that_shares_indentation_with_an_earlier_block).
         Test: tests/test_sandbox.py::ExtractCodeTests::test_dedents_a_fence_indented_under_a_markdown_list
               (red-proven: a fence for `def add(a, b): return a + b` plus a same-level `print(add(1, 2))`,
               indented 3 spaces under a markdown list item, raised `IndentationError: unindent does not match
               any outer indentation level` on compile() before the fix — the exact error string seen in the
               flagged run — and now round-trips to flush-left, correctly-indented code).
         Status: shipped 2026-09-22; sdlc_check.py exit 0. Independent review (general-purpose agent, fresh
               context) found the first version dedented every matched block independently (including when
               `code_blocks_with_def` joins more than one), which is a real regression: a class shown in one
               fence with a method continuation in a second, indented fence lost the method's indentation and
               it silently became a free function instead of staying nested (confirmed by hand: 2/2 assertions
               passed before the first version of this fix, 1/2 after, with `'Calculator' object has no
               attribute 'subtract'`). Fixed by scoping dedent to the unambiguous single-fence case only,
               red-proven via test_does_not_dedent_a_continuation_fence_that_shares_indentation_with_an_earlier_block
               before the fix, green after. Re-verified by tests only, not re-sent to the reviewer agent. -->

**Open questions for the operator (do not implement without a decision):**
1. Approve 9.1 (diagnostics) and 9.2 (hardening) both, or only 9.1 for now?
2. Should anything be done about the suspected non-determinism itself (e.g. a coding/reasoning suite retry-on-fail,
   sampling GPU occupancy at request time, a "seed did not guarantee determinism" warning) — or is documenting the
   gap in spec.md Phase 9 enough for now? This is a bigger design change (cost/time tradeoffs of retries; risk of
   masking a genuine model weakness as "just flakiness") and needs an explicit decision, not an assumption.
3. Should `results/runs/runs/benchmark_20260922_214037.json`'s existing scorecards be re-run once 9.1 ships, to get
   a diagnosable data point the next time this happens — or leave it as an unexplained historical run?

---

## Phase 10: Display honesty for floor-driven decode-speed ceilings

Status: shipped 2026-09-23. Review (2026-09-23, fresh-context `general-purpose` subagent, together with
Phase 11): 16 findings in total; the Phase 10 ones — (1, High) `_base_record` never copied the flag, so the tilde and
the CSV `True` could not appear in a real run (the tests set it on hand-built scorecards); (2, Medium) a raw eval
window of exactly 0.0 was not flagged; (3, Medium) Ollama was flagged although it has no floor (operator: floor path
only); (9, Medium) no Phase 10 text in `spec.md`, wrong scorecard key in CHANGELOG, no legend; (16, Info) untested
boundaries, non-streaming path, Ollama, runner propagation, older run JSON — all fixed test-first (13, the
whole-scorecard `~` over-warning, is deferred with the operator's yes: see Backlog). Surface surfaced during
the Sept 22 anomaly verification (`docs/analysis/2026-09-23/ANOMALY_VERIFICATION.md`). A measurement floor
in `benchrig/core/client.py` (`eval_duration_sec = max(0.001, end_wall_time -
first_token_time)`) prevents divide-by-zero on sub-millisecond generations, but the
resulting displayed speed is a meaningless ceiling (`eval_count=3 / 0.001s = 3000.0 t/s`
for Phi-4-mini on every 4-digit-number context-suite answer; same artifact at
`6000.0 t/s` for Phi-3.5 on the same answers). The Markdown report shows the bare number
— no annotation that the floor was applied, no distinction from a real `3000 t/s`
measurement. Intent.md Constraint 5 is technically upheld (the speed is what the client
measured) but the **display** is misleading when the answer is short enough that the
floor engages.

**Operator decisions (2026-09-23):**

1. **Display form**: tilde prefix on the speed cell — `~3000.0 t/s` in the Markdown table. Minimal,
   readable, no new column. Markdown only — no separate column / footnote.
2. **Retroactive**: forward-only. Existing `LATEST_SUMMARY.md` files are left untouched.
   The annotation ships with the next run that hits the floor.
3. **CSV**: yes — add `eval_tok_sec_floored: bool` column to `SCORECARD_CSV_COLUMNS` so operators
   parsing CSV programmatically can filter out floor-driven values.

### Plan items

- [x] <!-- Item 10.1: Annotate floored decode-speed values in Markdown (tilde prefix) and CSV
         (new `eval_tok_sec_floored` boolean column). One per-record flag from `client.py` is
         carried through to scorecards; the reporting layer reads the flag.
         Files: benchrig/core/client.py (set eval_tok_sec_floored when the floor engages on
         the streaming path; same for non-streaming eval_duration_sec < 0.0015 and small eval_count),
                benchrig/reporting/markdown.py (render ~3000.0 t/s when floored),
                benchrig/reporting/csv_export.py (carry eval_tok_sec_floored through the row),
                tests/test_floor_speed_annotation.py (new; holds the client, Markdown and CSV
                tests below in one file — corrected 2026-09-23, the plan originally named
                tests/test_prism_runtime.py, tests/test_report_markdown.py and tests/test_report_csv.py).
         Test (red-first):
           - test_eval_tok_sec_floored_true_when_floor_engages: a streamed response with
             eval_count=3 and eval_duration_sec < 0.0015 produces a record with
             eval_tok_sec_floored=True.
           - test_eval_tok_sec_floored_false_when_real: a response with
             eval_duration_sec=0.005 and eval_count=10 has eval_tok_sec_floored=False.
           - test_markdown_renders_tilde_when_floored: a scorecard with
             eval_tok_sec_floored=True renders the cell as ~<value> t/s.
           - test_markdown_renders_plain_when_real: a scorecard with
             eval_tok_sec_floored=False renders the cell as <value> t/s (no tilde).
           - test_csv_carries_eval_tok_sec_floored_column: SCORECARD_CSV_COLUMNS ends with the
             new boolean column and it round-trips through the exporter.
         Risk: the tilde prefix is a minor textual change to existing Markdown columns. Operators
         piping the Markdown through grep may need to update their regex. Documented in
         docs/cli.md or docs/benchmark-suites.md (whichever carries the scorecard column legend).
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (354 passed, 4 skipped). -->

---

## Phase 11: Structured logging on stderr

Status: shipped 2026-09-23 (items 11.1-11.7 all green). Review (2026-09-23, fresh-context `general-purpose`
subagent, together with Phase 10): 16 findings, triaged with the operator. Fixed test-first: (4) Prism's
`_make_request` bypassed `_logged_request`, so Prism emitted no `http.*` events; (5) `response_completed` was logged
for non-2xx, Foundry/Prism records had `model: null` (spec text corrected: `ttft_sec`/`eval_count` are not known at
the HTTP layer); (6, operator: probes at DEBUG) a stopped Ollama daemon printed JSON at the default WARNING level;
(7, operator: remove) `--check`/`--pull-recommended` emitted unplanned `run.started`; (8) `run.completed` carried
`scorecards_produced`/`targets_planned` instead of the spec's `models_ok`/`models_skipped`; (10) `unload.completed`
failure was INFO, not DEBUG; (11) `taskName` leaked into every JSON record; (12) `--log-level debug` was rejected while
the env var accepted it, and the invalid-level message went to stdout; `CRITICAL` was accepted although spec lists
four levels. Not defects: (15) the positional `perf_counter` mocks in `tests/test_ollama_think.py` are brittle but
reshuffled, not weakened; (11, part) `NaN`/reserved-key extras/`setup_logging("BOGUS")` have no call site. Deferred
with the operator's yes: (13), (14) — see Backlog. Whole-suite gate exit 0; the fixes were verified by tests, not
re-reviewed by a second fresh subagent.

Surfaced by the operator ("dodaj logowanie i trace-yy") the same day `v0.2.0` shipped.
Scope: stdlib `logging` emitting JSON-shaped records to stderr, configurable via
`--log-level` flag or `BENCHRIG_LOG_LEVEL` env var. Default level `WARNING` (silent
stderr, matching today's UX). Out of scope: OpenTelemetry, file output, log
rotation, OTLP export. Stdout UX (rich.console.print, progress bar, Markdown
scorecard) is unchanged — every event documented below is *additional* to the UX,
not a duplicate.

### Why

- Phase 9 captured per-record `extracted_code`/`response_excerpt` so 0-passed runs
  can be diagnosed from the JSON alone. Same need, broader surface: today there is no
  machine-parseable trail of HTTP retries, capacity waits, or run lifecycle events.
- `benchrig/core/client.py` already has a `_log` (`logging.getLogger(__name__)`) used
  3 times (503 retries, unload failures). That logger was never configured; an INFO
  record went nowhere visible. Phase 11 wires it.

### Plan items

- [x] <!-- Item 11.1: Logging module (formatter + filter + setup).
         Files: benchrig/core/logging.py (new), tests/test_logging.py (new).
         Test:
           - test_json_formatter_emits_required_fields: a record logged through
             JsonFormatter serialises to one JSON object with keys ts/level/event/message.
           - test_run_id_filter_injects_contextvar: a record logged after
             setup_logging(run_id=…) carries that run_id; outside, field is absent.
           - test_setup_logging_replaces_existing_stderr_handler: idempotent (no dup
             stderr lines across re-calls of setup_logging in the same process).
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (326 passed, 4 skipped). -->

- [x] <!-- Item 11.2: CLI surface for log level.
         Files: benchrig/cli.py (`build_parser`, `main`, `run_benchmarks` top-level),
                benchrig/core/logging.py (already shipped in 11.1),
                docs/cli.md (§5 Logging),
                tests/test_cli_logging.py (new).
         Test:
           - test_log_level_flag_default_is_warning: `--log-level` absent → WARNING.
           - test_log_level_flag_accepts_debug: `--log-level DEBUG` round-trips through parser.
           - test_env_var_overrides_flag_default: BENCHRIG_LOG_LEVEL=DEBUG → DEBUG.
           - test_flag_overrides_env_var: explicit flag wins over env.
           - test_invalid_level_exits_non_zero: garbage value → clear error + exit 2.
           - test_run_started_event_fires_with_argv_runtime_num_models_runs.
           - test_run_completed_event_fires_with_total_duration (in finally, even on exit).
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (333 passed, 4 skipped). -->

- [x] <!-- Item 11.3: HTTP lifecycle log events on every requests.post from
         benchrig.core.client (OllamaClient, FoundryClient, PrismClient).
         Files: benchrig/core/client.py (instrument the `_make_request` hook),
                tests/test_client_logging.py (new).
         Test:
           - test_request_started_logged_on_post_with_attempt (FoundryClient hook).
           - test_response_completed_carries_status_code_and_duration.
           - test_request_failed_on_transport_error_logs_warning.
           - test_warning_level_suppresses_http_info_events.
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (337 passed, 4 skipped).
         Scope note: this item instruments the FoundryClient `_make_request` hook
         (which PrismClient inherits and overrides for the 503 retry path). OllamaClient
         has direct `requests.post` calls outside this hook that are not yet instrumented;
         tracked as a follow-up for a later phase (low priority — Ollama runs locally so
         HTTP failure is rare; the failure path is `_make_request` on Prism). -->

- [x] <!-- Item 11.4: Retry events for the Prism 503 helper.
         Files: benchrig/core/client.py (`_post_with_503_retry`),
                tests/test_retry_logging.py (new).
         Test:
           - test_retry_attempted_logged_with_delay_and_reason (INFO, attempt=1).
           - test_retry_exhausted_logs_warning_with_reason (WARNING, attempt=4).
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (339 passed, 4 skipped). -->

- [x] <!-- Item 11.5: Capacity-exhaustion events from `BenchmarkRunner`.
         Files: benchrig/core/runner.py (`_capacity_exhausted_reason` path),
                tests/test_runner_logging.py (new).
         Test:
           - test_capacity_exhausted_logged_with_reason: a scenario with a 503 carrying
             `insufficient_resources` triggers `event=capacity.exhausted` at WARNING
             with `model` and `reason`.
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (342 passed, 4 skipped). -->
- [x] <!-- Item 11.6: Unload events for Prism.
         Files: benchrig/core/client.py (`PrismClient.unload_model`),
                tests/test_unload_logging.py (new).
         Test:
           - test_unload_ok_logs_debug_with_model (DEBUG, ok=true).
           - test_unload_failed_still_returns_true_but_logs (DEBUG, ok=false, error=…).
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (342 passed, 4 skipped). -->
- [x] <!-- Item 11.7: Docs (`docs/cli.md` new section "Logging") + CHANGELOG.md entry.
         Files: docs/cli.md ("Logging" section between "Output" and §6, with
                --log-level + BENCHRIG_LOG_LEVEL examples and JSON-shape sample),
                CHANGELOG.md (`### Added` under [Unreleased]).
         Test: tests/test_docs.py::DocsCoverageTests::test_every_cli_option_is_documented
                exercises this indirectly: --log-level is in the CLI parser and is
                asserted to appear in docs/cli.md (`2026-09-23` redaction catches this).

**Follow-up 2026-09-23:** OllamaClient.generate + OllamaClient.unload_model
also route through the instrumented hook. Shared `_logged_request` helper;
`tests/test_ollama_think.py` widened by 2 perf_counter positions to account
for the new HTTP-hook reads. No plan-item numbering: this was a deferred
extension of 11.3.
         Status: shipped 2026-09-23; sdlc_check.py exit 0 (342 passed, 4 skipped). -->
**Risks and open questions:**

- Volume on long runs: a 12-scenario reasoning suite at `--runs 3` produces ~36
  http.request_started events per model at INFO. With `--runtime all` and 5 models,
  ~180 records per invocation. Document in `docs/cli.md`; the default WARNING keeps
  the JSON stderr silent in normal runs.
- `--log-level` is additive — it must not fight with a future `--verbose`/`--quiet`.
  Consolidate on a single verbosity flag if any appear.
- `rich.console.print` to stdout stays as-is. Do not migrate to logging; that would
  rewrite the UX layer beyond Phase 11's scope.
- `run_id` is one UUID per `cli.py::main` invocation, shared across all sub-runs
  (`--runtime all`, `--runs N`). This is deliberate: a single run_id correlates
  every record from one CLI call.
- Defer: file output, log rotation, OpenTelemetry OTLP, trace_id/span_id propagation.
  Re-scope into a numbered phase when needed.

---

## Backlog (not scheduled)

- **Scorecard-wide `~` is coarse (Phase 10 review finding 13).** `compute_model_scorecard` sets `eval_tok_sec_floored`
  when *any* token-producing record is floored, so one sub-1.5 ms answer among hundreds of real ones marks the whole
  leaderboard speed `~` although the token-weighted average is dominated by real records. Decide a rule first (flag only
  when floored records carry most tokens? show a count?) — operator deferred, 2026-09-23.
- **Log write inside the measured TTFT window (Phase 11 review finding 14).** `start_wall_time` is taken before
  `_make_request`, so at `--log-level INFO`/`DEBUG` the `http.request_started` write (JSON format + stderr) lands
  inside TTFT and slightly inflates it (Constraint 5). The default WARNING is unaffected. Fix would be to log before
  taking `start_wall_time` (or take the timestamp after the log call) — operator deferred, 2026-09-23; until then, do
  not compare timings taken at INFO/DEBUG with timings taken at WARNING.

Surfaced while specifying Phase 6 against `~/03-foundy-local` `main`; none are approved for implementation.
Re-scope into a numbered phase (with its own `spec.md` section and operator approval) before starting any of these.

- **Embeddings suite.** Prism serves `/v1/embeddings` (Ollama-backed models only; ONNX Runtime GenAI does not
  produce embeddings). Benchrig has no embedding suite or `BaseRuntimeClient.embed()` method today — would need
  a new suite type end-to-end (scenario shape, scoring, reporting columns), not just a client method.
- **Tool-calling suite.** Prism's `/v1/chat/completions` accepts `tools`/`tool_choice` and returns `tool_calls`
  (including the `d143788` fix that turns a template render failure into a `400` instead of silently dropping
  tools). Benchrig has no tool-use suite; would need scenario definitions with expected tool calls and a scorer.
  Re-scoped as proposed Phase 13 (2026-10-01).
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

---

## Phase 12: Repo-quality hardening from independent review

Status: approved for items 12.1, 12.2, 12.6, 12.8, 12.9, 12.10 (operator requests 2026-10-03); items 12.1, 12.2, 12.6, 12.8, 12.9, 12.10 completed and independently reviewed (all findings triaged and resolved); sdlc_check.py exit 0; ready for operator release decision. Item 12.3 approved by operator 2026-10-03 (spec drafted in `spec.md` §12.3); implementation shipped and independently reviewed twice (fresh-context `general-purpose` subagent). First review: 8 findings (1×P1, 2×P2, 5×nit) — all addressed. Second review: 5 prior fixes verified, 2 new nits raised — non-string `evaluator` value now rejected by `isinstance` check (added test), and dead `(or Decimal)` clause dropped from `spec.md`. sdlc_check.py exit 0; ready for operator release decision. Item 12.4 approved by operator 2026-10-03; completed 2026-10-04. Restored compatibility exports, package/source-module dispatch and helper lookups, and dispatch-test patch cleanup. Review: fresh-context Codex subagent `/root/review_cli`, three passes; three P2 findings reproduced red, fixed, and independently re-reviewed with no remaining findings (23 dispatch tests independently passed). Full gate exit 0: 460 passed, 4 skipped, 4 subtests passed; lint, format, changelog PASS. Final `.venv/bin/python -m build` exit 0: wheel and sdist built; isolated dependencies required approved network escalation after sandbox DNS failure. Ready for operator release decision; no commit or push. Item 12.5 revised to PyPI-first installation documentation, authorized 2026-10-04; item 12.7 relocated and independently reviewed 2026-10-04; gate green, MkDocs build unverified (not installed).

- [x] **12.1 Sandbox docstring matches actual isolation level:** update docstrings in `benchrig/core/sandbox.py`, CLI `--help`, and docs to describe process-group isolation and disclose residual risks. Files: `benchrig/core/sandbox.py`, `benchrig/cli.py`, `docs/benchmark-suites.md`, `docs/development.md`. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **12.2 Document coding suite is Python-only:** add clear Python-only scope statements and create `benchrig/data/scenarios/coding.README.md`. Files: `docs/benchmark-suites.md`, `README.md`, `benchrig/data/scenarios/coding.README.md`. Shipped 2026-10-03, sdlc_check.py exit 0.

- [x] **12.3 Numeric-equivalence layer for reasoning ground truth.**
         Files: `benchrig/core/reasoning_parser.py` (extend `evaluate_reasoning_answer` with an `evaluator`
         kwarg accepting `"numeric"` and `"numeric_text"` per `spec.md` §12.3), `benchrig/core/runner.py`
         (`_evaluate_answer` passes `evaluator=sc.get("evaluator")` through), `benchrig/data/scenarios/reasoning.json`
         (add `"evaluator"` field to numeric scenarios), `tests/test_reasoning_evaluator.py` (new — see test
         description below), `docs/benchmark-suites.md` (add "Reasoning ground-truth comparison" subsection),
         `CHANGELOG.md` (one-line `[Unreleased]` entry).
         Test: `tests/test_reasoning_evaluator.py::NumericEquivalenceTests` — model answer `"42"` matches
         expected `"42"`, `"42.0"`, and `"  42  "` (whitespace-trimmed) when scenario has `evaluator: "numeric"`;
         `"forty-two"` matches `"42"` only when scenario has `evaluator: "numeric_text"`; `"Forty-Two"` (capitalization)
         does NOT match `"forty-two"`; non-numeric ground truth (e.g. word puzzles, regex-based answers) keeps
         the current string/regex path unchanged; an unknown `evaluator` value is treated as `None` and emits
         a structured-log event `evaluator.unknown`.
         Spec: `spec.md` §12.3 written. Status: implemented 2026-10-03; sdlc_check.py exit 0; ready for independent review.

- [x] **12.4 Split `benchrig/cli.py` into per-command modules.**
         Files: `benchrig/cli.py` (removed), `benchrig/cli/__init__.py` (new thin dispatcher: `build_parser`, `main`, and backward-compat re-exports of every symbol tests import/patch via `benchrig.cli.*`), `benchrig/cli/_common.py` (constants + cross-command helpers), `benchrig/cli/check.py` (`--check` and `--pull-recommended` paths), `benchrig/cli/compare.py` (`--compare` path), `benchrig/cli/run.py` (main benchmark run), `benchrig/cli/report.py` (`--csv`/`--chart`/markdown writers).
         Test: existing CLI tests must keep passing unchanged (`tests/test_benchmark_cli.py`, `test_cli_logging.py`, `test_cli_report_flags.py`, `test_cli_capacity_skip_notice.py`, `test_packaging.py`, `test_foundry_runtime.py`, `test_warmup_protocol.py`, `test_report_1to1.py`); add `tests/test_cli_dispatch.py` asserting that `benchrig.cli:main` routes `--check` (→ `run_system_check`), a benchmark run (→ `run_benchmarks`), and `--csv`/`--chart` (→ `save_outputs` in `report.py`) to the right subcommand module, that `benchrig --help` renders identically to today, and that every symbol tests import via `from benchrig.cli import X` keeps resolving after the restructure.
         Spec: `spec.md` §12.4 written. Status: completed 2026-10-04; gate and package build green, independent review clean.

### Item 12.5: Installation documentation — PyPI first

Status: revised scope explicitly authorized by operator 2026-10-04 ("zrób to"
after proposing PyPI-first instructions and deferring setup scripts).
Completed: PyPI-first README/quickstart and separate development instructions.
Verification: `python3 scripts/sdlc_check.py` exit 0 (460 passed, 4 skipped,
4 subtests passed; lint, format, changelog PASS); `git diff --check` clean.
No fresh package installation performed. No commit or push requested.
Trivial documentation exception applies: stages 1–4/red-first tests skipped.

- [x] **12.5 Installation instructions.** Files: `README.md`,
  `docs/tutorials/quickstart.md`, `spec.md`, `plan.md`. Put virtualenv + PyPI
  installation first, keep optional extras in that environment, and separate
  repository development installation (Python 3.11+) from product use (3.10+).
  Verification: review command consistency and run `python3 scripts/sdlc_check.py`
  including existing `tests/test_docs.py` and `tests/test_packaging.py` coverage.

Deferred: the Linux/WSL bootstrap scripts and their proposed tests. No runtime,
intent, or invariant changes. PyPI downloads and live hardware checks are not
part of this documentation-only task. No commit, push, or release requested.

- [x] **12.6 Surface composite weights in LATEST_SUMMARY.md:** render active composite scoring weights in the metadata header of the markdown report. Files: `benchrig/reporting/markdown.py`, `benchrig/core/runner.py`, `tests/test_report_markdown.py`. Verification: `tests/test_report_markdown.py::CompositeWeightsHeaderTests`. Shipped 2026-10-03, sdlc_check.py exit 0.

### Item 12.7: Move historical analyses into documentation

Status: implementation authorized by operator direction 2026-10-04 ("przeneisc");
completed: analyses relocated and labelled, archive index/nav/links updated.
Gate exit 0: 460 passed, 4 skipped, 4 subtests passed; lint, format, changelog PASS.
Independent read-only review `/root/review_analysis`: no actionable findings;
7 docs tests passed, 62 relative links checked, MkDocs YAML/navigation verified.
MkDocs build unverified (not installed). Original ignored-file byte parity was
not independently verified; relocation preserved bodies via rename and added notices.
Documentation-only change; no new implementation-mirroring tests. Ready for
operator decision; no commit, push or publication.

- [x] **12.7 Relocate and label historical analyses.** Files:
  `results/RUN_ANALYSIS.md` → `docs/analysis/2026-09-23/RUN_ANALYSIS.md`,
  `results/ANOMALY_VERIFICATION.md` → `docs/analysis/2026-09-23/ANOMALY_VERIFICATION.md`,
  `docs/analysis/index.md`, `mkdocs.yml`, `README.md`, `docs/index.md`,
  `CHANGELOG.md`, `spec.md`, `plan.md`. Verification:
  `tests/test_docs.py::DocsCoverageTests::test_every_page_is_in_the_nav_and_every_nav_entry_exists`,
  link/content review, `python3 scripts/sdlc_check.py`, fresh independent review.

Risks: source analyses are ignored local files, while source run JSON/CSV/PNG
remain unversioned. Preserve source paths as provenance and explicitly disclose
that inputs may be unavailable in a fresh checkout. Historical findings are not
revalidated or rewritten as current claims. No intent/invariant changes or shipping.

- [x] **12.8 Pin extract_python_code single-vs-multi-fence dedent behavior:** add regression tests pinning single-fence dedent, multi-fence indentation preservation, and zero-fence fallback. Files: `tests/test_sandbox.py`. Verification: `tests/test_sandbox.py::ExtractPythonCodeDedentTests`. Shipped 2026-10-03, sdlc_check.py exit 0.

- [x] **12.9 Add last green CI run indicator to README:** add indicator link to latest successful CI run. Files: `README.md`. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **12.10 Slim AGENTS.md and move operational playbook to docs:** create `docs/agent-debugging.md`, register in `mkdocs.yml`, and streamline `AGENTS.md`. Files: `AGENTS.md`, `docs/agent-debugging.md`, `mkdocs.yml`. Shipped 2026-10-03, sdlc_check.py exit 0.

Risks and open questions:

- Items 12.4 and 12.7 are the largest in scope. Defer them if the operator wants the quick wins (12.1, 12.2, 12.6, 12.8, 12.9, 12.10) shipped first.
- Item 12.7 direction chosen 2026-10-04: relocate historical analyses; automated analysis is outside scope.
- Items 12.1, 12.2, 12.9, 12.10 are docs-only and can be approved in a single batch with no code or test churn — good candidate for the next sprint's "boring but valuable" work.
- Per `plan.md` header rule, **no item may start implementation until `spec.md` is updated** for the behavior it adds and the operator approves the item. The Phase 11 pattern (spec section first, then per-item approve-and-implement) applies.

---

## Phase 13: Tool-use suite (spec/plan drafted)

Status: implementation authorized by operator 2026-10-04 ("zatwierdzam") for
13.1–13.4, including the additive I1 CSV contract change. Implementation green:
full gate exit 0 (516 passed, 4 skipped, 4 subtests passed; lint/format/changelog
PASS). Red evidence: new scenario loading, client API, runner continuation,
aggregation, legacy isolation, CSV columns and null-safe report tests observed
failing before their implementation. Existing progress behavior retained and
covered (test factory corrected to wire the callback). Next: package build and
independent review; CI3.10 fix and requested methodology feedback included. Live experiments/threshold decision 13.5 remain deferred.
Supersedes the "Tool-calling suite" backlog bullet. Motivation: `~/06-dark-factory` may evolve into a tool-using harness
(with `~/07-rag` as a `rag_search` tool); whether the local models can drive one is unmeasured.

Evidence from a throwaway probe, 2026-10-01 (Ollama `/api/chat`, `tools` set, `temperature 0`, 3 prompts per model, **n=1,
indicative only, not a benchmark**; script was not kept in the repo):

| Model | select `rag_search` | select `read_file` | "no tool needed" (2+2) | Notes |
|---|---|---|---|---|
| `llama3.1:8b` | structured call | structured call | no call, but refused instead of answering | `k` returned as string `"1"` (schema says integer) |
| `mistral:7b` | structured call | structured call | **called `read_file` wrongly** | over-calls |
| `phi4-mini` | structured call | structured call | no call, but refused instead of answering | |
| `qwen2.5-coder:7b` / `:14b` | **JSON as plain text, no `tool_calls`** | same | answered `4` | right intent, wrong channel (template/parse issue); a client must not count this as a pass without a rule |
| `gemma3:12b` | HTTP 400 "does not support tools" | same | same | not usable for tools on Ollama |

Takeaways to design against: (a) "emitted valid JSON in `content`" and "returned structured `tool_calls`" are different
outcomes and must be scored separately; (b) abstention (no call when none is needed) fails in both directions; (c) argument
types drift; (d) unsupported-tools models must be reported as `unsupported`, not as `0%`.

- [x] **13.1 Scenario format and bundled cases.** Files: `benchrig/data/scenarios/tool_use.json`,
  `benchrig/core/tool_use.py` (schema validation), `docs/scenarios.md`,
  `tests/test_tool_use_scenarios.py`. Tests: `test_bundled_categories_have_three_cases_each`,
  `test_invalid_scenario_is_rejected_before_requests`, `test_multistep_requires_fixture_result`,
  `test_abstention_requires_answer`, `test_typed_matchers_are_validated`.
- [x] **13.2a Tool chat clients.** Files: `benchrig/core/client.py`,
  `benchrig/core/onnx_client.py`, `tests/test_tool_use_clients.py`.
  Tests: `test_ollama_uses_chat_and_object_arguments`,
  `test_foundry_and_prism_decode_argument_strings`,
  `test_direct_onnx_returns_unsupported_without_generation`,
  `test_malformed_arguments_preserve_raw_evidence`,
  `test_content_json_is_not_a_structured_call`, `test_only_explicit_capability_rejection_is_unsupported`,
  `test_chat_tools_preserves_auth_loading_and_retry_contracts`,
  `test_nonstreaming_latency_is_not_ttft`. Mock HTTP response fixtures;
  no live daemon requirement or cloud client.
- [x] **13.2b Fixture conversations and measured records.** Files:
  `benchrig/core/runner.py`, `benchrig/core/tool_use.py`, `tests/test_tool_use_runner.py`.
  Tests: `test_continuation_uses_actual_call_id_and_fixture_result`,
  `test_failed_turn_stops_followups_but_preserves_planned_count`,
  `test_unsupported_stops_only_tool_requests_for_that_model`,
  `test_warmups_and_repetitions_use_fresh_conversations`,
  `test_fixture_tools_never_execute_model_selected_actions`,
  `test_effective_settings_and_placement_are_recorded`.
- [x] **13.3 Strict scoring and aggregation.** Files: `benchrig/core/tool_use.py`,
  `benchrig/core/runner.py`, `tests/test_tool_use_scoring.py`.
  Tests: `test_argument_matching_is_typed_and_rejects_extras`,
  `test_refusal_and_text_json_fail_abstention`,
  `test_rates_use_planned_denominators_and_exclude_unsupported_and_warmups`,
  `test_zero_denominators_are_unavailable`, `test_tool_evidence_does_not_change_legacy_composite`.
  Cover booleans vs integers, malformed/multiple calls, reordered keys,
  array ordering, transport failures and skipped continuation turns.
- [x] **13.4a CLI and reports.** Files: `benchrig/cli/__init__.py`,
  `benchrig/cli/_common.py`, `benchrig/cli/run.py`, `benchrig/cli/compare.py`, `benchrig/reporting/common.py`,
  `benchrig/reporting/display.py`, `benchrig/reporting/markdown.py`,
  `benchrig/reporting/csv_export.py`, `benchrig/reporting/charts.py`,
  `tests/test_tool_use_reports.py`, `tests/test_cli_dispatch.py` (update help snapshot).
  Tests: `test_tool_suite_is_opt_in_and_all_is_unchanged`,
  `test_reports_render_counts_and_unavailable_rates`,
  `test_tool_only_scorecards_have_no_fabricated_composite`,
  `test_csv_appends_tool_columns_preserving_existing_columns`,
  `test_charts_distinguish_tool_success_from_composite`,
  `test_progress_counts_scenario_repetitions_not_turns_or_warmups`.
- [x] **13.4b Documentation and changelog.** Files: `docs/benchmark-suites.md`,
  `docs/cli.md`, `docs/runtimes.md`, `README.md`, `CHANGELOG.md`.
  Verification: existing `tests/test_docs.py`, all new hermetic tests, full SDLC
  gate, package build and independent review. Describe `--suite tool_use --runs 3`,
  fixture-only calls, backend coverage, metric denominators and timing limitations.
- [ ] **13.5 Live results / harness decision (deferred).** Files: future dated
  `docs/analysis/` note and `mkdocs.yml`; scenarios/protocol above. Requires a
  separate operator request with selected model/runtime pairs and agreed thresholds.
  Verification: raw local run artifacts with versions/settings/counts/coverage;
  no implementation or live run authorized by planning this phase.

Risks and decisions proposed for approval:

- Opt-in suite, existing `--runs` default 1, documented recommendation 3; no new
  `--repeat` flag or automatic nonzero-temperature experiment.
- Initial non-streaming transports expose honest wall-time latency; unavailable
  TTFT/decode remain null. Streaming tool fragments are outside this first version.
- Foundry is included with Ollama and Prism; direct ONNX is unsupported via its
  current adapter. Capability errors must be distinguished from generic bad requests.
- Rates are per (model,runtime); templates/quantization/runtime versions matter.
  Do not generalize results across engines or certify determinism from a seed.
- Appended CSV columns require explicit approval under I1. Composite weights,
  defaults and legacy suite metrics remain unchanged; tool-only null rendering
  is required across all reporters, including chart exports.
- Multi-step calls use inert fixtures and stop after first mismatch. Abstention
  checks require a correct plain answer, so a refusal is not rewarded as no-call.
- Local Prism checkout from older notes is unavailable at its recorded path;
  hermetic fixtures establish client behavior, not deployed compatibility.
- No live benchmarks, cross-repo changes, commits, pushes or releases until
  explicitly requested. Nonzero-temperature variants and 13.5 thresholds deferred.


## Adversarial review follow-up — 2026-10-02

Status: implemented, verified green, and independently reviewed (clean verdict, no findings). Trust boundary defined in `spec.md` (I10), regression coverage in `tests/test_sandbox.py`, out-of-band JSON result file with secret token in `benchrig/core/sandbox.py`. Ready for operator decision.

- [x] **P0 — verdict integrity:** define the trust boundary in `spec.md`; revise `benchrig/core/sandbox.py` so candidate stdout cannot determine the verdict or expected test count. Add regression coverage in `tests/test_sandbox.py` for forged markers plus early successful exit, forged totals, and a marker printed before genuinely failing tests. Every case must fail against the caller's original assertions; rerun the SDLC gate and obtain independent review before closing.


## Workspace SDLC unification — 2026-10-02

Status: migration implemented under operator authorization; validation recorded below. Preserve earlier plan entries and uncommitted work.

- [x] U1. Install/update kit-owned runner, skills, hook and Cursor rule; normalize the process in `AGENTS.md` and harness adapters. Evidence: `.sdlc/test_unification.py` compares all eight projects to the kit.
- [x] U2. Configure `sdlc.toml` without dropping existing checks; preserve project-specific helpers and red-mode error handling. Evidence: migrated gate-helper tests where present, gate CLI smoke and configuration validation.
- [x] U3. Update process references, `REVIEW.md`, changelog and documentation mirrors where applicable. Run the full project gate; record pass/fail/skip evidence and obtain independent review. No checkbox closes on partial checks alone.

### Codex migration evidence — 2026-10-02

Status: shared process, kit distribution, project profiles, Codex adapters, review policy and CI implemented under the operator request. Existing product-version CI remains. Workspace distribution checks: 2 passed. Independent read-only review found two migration defects (legacy Python bootstrap and lint comparison-base forwarding); both reproduced red, fixed and re-reviewed with no new findings. Actual Python 3.8/3.9 was unavailable; bootstrap regression uses simulated old builtin typing behavior.

Verification: Gate exit 0: 376 passed, 4 skipped; lint, format, changelog PASS.

Unverified gates stay open; no sandbox bypass or skipped test replacement. Earlier adversarial remediation remains in its own plan phase.

## Phase 14: Interpretable benchmark comparisons

Status: completed (2026-10-03); all items 14.1–14.5 shipped; review complete (7 findings triaged and resolved test-first); sdlc_check.py exit 0; ready for operator release decision.

- [x] **14.1 Execution placement and CPU fallback:** capture requested/observed provider/device and expose fallback in reports. Files: `benchrig/core/client.py`, `benchrig/core/onnx_client.py`, `benchrig/core/runtimes.py`, `benchrig/core/runner.py`, `benchrig/reporting/markdown.py`, `benchrig/reporting/csv_export.py`, `docs/runtimes.md`. Verification: `tests/test_execution_placement.py` covering fallback flags, unknown placement attribution, and markdown/csv export visibility. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **14.2 Timing breakdown and provenance:** report TTFT, prefill and decode independently with provenance and unavailable/estimated states. Files: `benchrig/core/client.py`, `benchrig/core/onnx_client.py`, `benchrig/core/runner.py`, `benchrig/reporting/markdown.py`, `benchrig/reporting/csv_export.py`. Verification: `tests/test_timing_provenance.py` proving TTFT is never treated as engine prefill duration and metrics remain distinct across exports. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **14.3 Peak RSS at fixed context:** define process coverage and sampling, record workload parameters and report RSS alongside platform-specific GPU/UMA telemetry. Files: `benchrig/core/hardware.py`, `benchrig/core/runner.py`, `benchrig/reporting/markdown.py`, `benchrig/reporting/csv_export.py`, `docs/hardware-telemetry.md`. Verification: `tests/test_rss_sampling.py` verifying host RSS peak sampling, process coverage fallback (`client_only` vs `client_and_server`), and fixed-context workload recording. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **14.4 Warm-up and KV reuse protocol:** separate cold runs, warm model/runtime runs and verified prefix-cache reuse with repeated identical workloads. Files: `benchrig/cli.py`, `benchrig/core/runner.py`, `benchrig/core/client.py`, `docs/tutorials/cross-engine-benchmarking.md`. Verification: `tests/test_warmup_protocol.py` proving warm-ups are excluded from measured aggregates and unverified prefix reuse is labelled correctly. Shipped 2026-10-03, sdlc_check.py exit 0.
- [x] **14.5 Executable-test evidence:** show pass counts/denominators and distinguish task success from assertion success, alongside per-task evidence and supplementary aggregate scores. Files: `benchrig/core/runner.py`, `benchrig/reporting/markdown.py`, `benchrig/reporting/csv_export.py`, `docs/benchmark-suites.md`, `CHANGELOG.md`. Verification: `tests/test_coding_metrics.py` covering disaggregated task vs assertion pass counts and reporting. Dependency: P0 verdict integrity is closed. Shipped 2026-10-03, sdlc_check.py exit 0.

## Phase 15: Model selection for 12 GB VRAM (RTX 5070) — measured, not assumed (proposed)

Status: proposed (2026-10-04); **pending operator approval** before any item moves to `approved` and `spec.md` is updated.
Origin: external feedback recommending Qwen3 32B Q4_K_M / GLM 32B Q4 / DeepSeek-R1 Distill 32B Q4 over a 78B IQ2 model
(Kolibri-1) on an RTX 5070 12 GB for LM Studio (architecture, IoT, code, PowerShell, analysis). The feedback contained no
measurements; this phase turns its claims into hypotheses that benchrig can confirm or reject on the operator's host.

Hypotheses to test (none verified yet):

- H1: a 32B Q4_K_M (~19-20 GB) does not fit in 12 GB VRAM, so it runs with partial CPU offload; the claimed "2-4x more
  tokens/s" over 78B IQ2 (~22-26 GB, also offloaded) is more plausibly ~1.5-2x and depends on RAM bandwidth and context.
- H2: IQ2 degrades code and strict-format output (PowerShell, JSON) enough that a 32B Q4 wins on `coding`/`polish` quality.
- H3: models that fit entirely in VRAM (e.g. Qwen3 14B Q4/Q5) or MoE models with few active parameters (Qwen3-30B-A3B,
  Qwen3-Coder-30B-A3B) beat the dense 32B on tokens/s at comparable quality — the feedback did not consider them.
- H4: reasoning-distilled models (DeepSeek-R1 Distill 32B) lose on effective answer latency because of long thinking
  traces; `answer_ttft_sec` / `thinking_chars` (Phase 6) should show it.
- Unverified names: "glm-5.3 32B" and "Kolibri-1 78B" must be confirmed to exist (and in which quant) before they enter a run.

- [ ] <!-- Item 15.1: Candidate matrix and run protocol (docs only).
         Files: docs/tutorials/model-selection-12gb.md (new), mkdocs.yml (register page).
         Content: the candidate list (Qwen3 14B Q4/Q5, Qwen3-30B-A3B Q4, Qwen3-Coder-30B-A3B Q4, Qwen3 32B Q4_K_M,
         DeepSeek-R1 Distill 32B Q4, GLM 32B Q4 once its name is verified, Kolibri-1 78B IQ2 as benchmark-only), the fixed
         context (reuse 14.3's fixed-context workload), `--runs 3`, warm-up protocol from 14.4, suites
         `coding,reasoning,polish,speed`, and the recorded host facts (RAM size/type/speed, driver, context length,
         KV-cache quantization).
         Test: tests/test_docs.py presence check for the page and the mkdocs nav entry.
         Status: proposed. -->

- [ ] <!-- Item 15.2: Record how much of a model ran on the GPU (partial offload fraction).
         Files: benchrig/core/client.py / benchrig/core/runtimes.py (read offloaded-layer info where the runtime exposes
                it, e.g. Ollama `/api/ps` size vs size_vram), benchrig/reporting/markdown.py, benchrig/reporting/csv_export.py.
         Builds on 14.1 (requested/observed placement): that item records provider/device and CPU fallback, but nothing
         in the repo captures a *partial* GPU/CPU split, which is exactly the regime of H1/H3. Report as
         "unavailable" (not 0%) when the runtime does not expose it.
         Test: fake-server tests for the exposed and not-exposed cases; markdown/CSV rendering tests (hermetic).
         Open question: LM Studio is the feedback's target runtime; benchrig has no LM Studio client. Decide whether to
         benchmark the same GGUFs via Ollama (llama.cpp) as a proxy, or add an OpenAI-compatible runtime for LM Studio.
         Status: proposed; needs a spec and an operator decision on the runtime question. -->

- [ ] <!-- Item 15.3: Real run and results note (documentation, not code).
         Files: results/ (run JSON, as for other runs), docs/ (short results note).
         Run the matrix from 15.1 on the RTX 5070 12 GB host, record tok/s, TTFT, answer latency, peak VRAM/RSS and suite
         scores per model; state for each of H1-H4 whether it held, with the numbers. Feeds the operator's default-model
         choice for LM Studio; the 78B IQ2 stays a benchmark data point only.
         Status: proposed; needs real hardware and enough free VRAM (opt-in, like the other live runs). -->

Risks and open questions:

- Offloaded 32B/78B runs are slow (single-digit tok/s); a full `--suite all --runs 3` may take hours per model. Order the
  matrix so the fast, fits-in-VRAM models run first and give a result early.
- `generic-cpu`/offload results depend on RAM bandwidth; do not generalize them to other hosts.
- Quant names (Q4_K_M, IQ2) are llama.cpp/GGUF concepts; Prism/ONNX int4 is not directly comparable (see Phase 4 and the
  ONNX-int4-vs-Ollama open item in `scratch/TODO.md` §4).
- Per the `plan.md` header rule, **no item may start implementation until `spec.md` is updated** and the operator approves it.


## CI Python 3.10 dependency-install fix — 2026-10-04

Status: authorized by operator's CI failure report; dependency marker implemented,
regression red-proven, full gate exit 0 (469 passed, 4 skipped); pending final
independent review with the active change. Phase 13.1 is implemented/green (468 passed,
4 skipped, 4 subtests passed); phase 13 resumes after this fix.

- [x] **CI3.10** Gate the kit dev requirement on Python >=3.11. Files:
  `pyproject.toml`, `tests/test_ci_dependencies.py`, `CHANGELOG.md`, `spec.md`,
  `plan.md`. Test: `test_kit_dependency_excludes_python310_and_includes_python311`.
  Verification: red-first, full gate, independent review. Actual GitHub run is
  external evidence, not claimed by local marker verification.


## Methodology feedback — Rafał Warzycha, 2026-10-04

Status: operator requested recording the feedback; documented in
`docs/tutorials/cross-engine-benchmarking.md`. Separate weights/kernel warm-up
from steady-state prefix KV reuse. Record load/prefill/decode independently of
TTFT; compare identical-prefix second requests at fixed cache type/context.
A future explicit cache-controlled experiment needs its own specification and
backend capability verification; no new cache-hit claim or invariant change.


### Phase 13 final verification — 2026-10-04

Implementation and independent review complete: `/root/review_tool_core` (48 focused
tests) and `/root/review_tool_reports` (31 tests, 4 subtests) found no remaining
actionable findings after fixes. Full gate exit 0: 516 passed, 4 skipped,
4 subtests passed; lint/format/changelog PASS. Review regressions cover capacity
suppression, request timing, raw telemetry, malformed usage, nonfinite arguments
and pair-selected suite preflight. Actual Python 3.10 execution and live runtime
experiments remain unverified; marker regression is hermetic. No shipping requested.

Final package build: `.venv/bin/python -m build` exit 0; wheel and sdist
built successfully with approved build-network escalation.
