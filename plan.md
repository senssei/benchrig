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
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

- [x] <!-- Item 1.2: PNG chart of composite score per runtime/model.
         Files: benchrig/reporting/charts.py (new), benchrig/reporting/__init__.py (export),
                benchrig/cli.py (wire --chart flag), tests/test_report_charts.py (new).
         Test: tests/test_report_charts.py::test_chart_export_writes_png_with_one_bar_per_scorecard
               generates a chart from a synthetic scorecard list, asserts the output PNG is non-empty,
               and asserts the bar count equals len(scorecards).
         Status: shipped 2026-09-21; sdlc_check.py exit 0. -->

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