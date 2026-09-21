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

Status: not approved yet.

- [ ] <!-- Item 2.1: Warn when `generic-cpu` execution provider is selected on a CUDA host.
         Files: benchrig/core/runner.py (detect cuda + generic-cpu, emit warning), benchrig/core/runtimes.py
                (expose `warning_for_provider`), tests/test_warnings.py (new).
         Test: tests/test_warnings.py::test_generic_cpu_on_cuda_emits_warning
               builds a synthetic RuntimeConfig with provider='generic-cpu' on a cuda host and asserts
               the warning helper returns the documented message; the CLI path is mocked.
         Pre-condition: write the warning text and exit code into spec.md §3 before this item starts. -->

- [ ] <!-- Item 2.2: Surface foreign-GPU-process VRAM in scorecards as `vram_baseline_dirty_mb`.
         Files: benchrig/core/hardware.py (measure nvidia-smi used memory minus own process before run),
                benchrig/core/runtimes.py (forward into scorecard), tests/test_hardware.py (extend).
         Test: tests/test_hardware.py::test_vram_baseline_dirty_records_foreign_process_memory
               injects a fake nvidia-smi output with a third-party PID and asserts the scorecard
               includes vram_baseline_dirty_mb > 0 and the existing vram_baseline_mb equals
               (own-process memory + foreign_process memory).
         Notes: `vram_baseline_dirty` is already produced by the runtimes layer (TODO §2 row 2);
                this item is the hardware-side measurement that feeds it. -->

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

## Phase 4: Integration with the next `prism-local` (unreleased, target 0.2.0)

Status: not approved yet. Behaviour here depends on the 0.2.0 contract that is still being designed in `~/03-foundy-local`; the items below are drafted from what we already know from `scratch/TODO.md` §2 and must be re-read once `prism-local` 0.2.0 is tagged.

- [ ] <!-- Item 4.1: Consume `stream_options.include_usage` end-to-end.
         Files: benchrig/core/client.py (PrismClient), benchrig/core/runtimes.py (usage flag),
                tests/test_prism_runtime.py (extend).
         Pre-condition: prism-local 0.2.0 exposes `usage` in the last streamed chunk; this item
         does NOT start before that contract is published. Today the benchrig side already flags
         the value as `usage_estimated` (TODO §2 row 3); the item upgrades the flag to `usage`
         when the real number arrives.
         Test: tests/test_prism_runtime.py::test_prism_stream_yields_usage_in_last_chunk
               with a recorded Prism stream from a 0.2.0 server (or a fake that mimics it). -->

- [ ] <!-- Item 4.2: `prism list` / `/v1/models` device label = the device the model will actually run on.
         Files: benchrig/core/client.py (PrismClient), benchrig/core/runtimes.py,
                tests/test_prism_runtime.py (extend).
         Pre-condition: prism-local 0.2.0 distinguishes `device` (resolved at serve time) from
         `exported_for` (the label the user passed). Today the benchrig side stores the field
         already (TODO §2 row 4); this item decides which one becomes the runtime label in the
         Markdown report.
         Test: tests/test_prism_runtime.py::test_prism_scorecard_uses_resolved_device_label
               feeds two synthetic /v1/models responses (one with mismatched exported_for) and
               asserts the scorecard's `runtime_label` comes from `device`, not `exported_for`. -->

- [ ] <!-- Item 4.3: `PRISM_PREFILL_CHUNK` opt-in path (already in scratch/TODO.md §2 row 8, kept opt-in).
         Files: benchrig/core/client.py (read env var, pass through), tests/test_prism_runtime.py
                (extend).
         Decision (operator): stay opt-in, do not flip on by default. Document the trade-off
         (peak −44/−54/−5 %, TTFT +4/+110/+33 %) in `docs/runtimes.md` and in `benchrig --help`.
         Test: tests/test_prism_runtime.py::test_prism_client_reads_PRISM_PREFILL_CHUNK_env
               asserts the client sends the chunked variant only when the env var is set. -->

Risks and open questions:

- `prism-local` 0.2.0 is unreleased; the contract may still shift. Items in this phase must be re-read
  at the start of each item against the current `~/03-foundy-local` `main` and against the eventual
  0.2.0 tag. If the contract drifts, stop and re-spec — do not silently change scope (sdlc-implement).
- Item 4.1 and 4.2 overlap with `benchrig.core.runtimes` constants (`ESTIMATED_USAGE_NOTE`,
  `MODEL_MEMORY_NOTE`). Adding a new note is fine; renaming or removing one is a spec change (I1).
- Tests for Prism currently mock the daemon. Recording a real 0.2.0 fixture would be cheaper in
  the long run; out of scope for this phase, recorded under `scratch/TODO.md` §4.