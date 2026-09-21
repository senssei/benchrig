# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer](https://semver.org/) (pre-1.0: minor
versions may include breaking changes).

## [Unreleased]

### Added
- CSV export of scorecards (`benchrig.reporting.csv_export.write_scorecards_csv`) with a fixed column order
  (`SCORECARD_CSV_COLUMNS`). Wired to the upcoming `--csv <path>` CLI flag (Phase 1, item 1.1).
- PNG chart export of scorecards (`benchrig.reporting.charts.write_scorecards_chart`) with one bar per
  scorecard (bar height = `composite_score`, label = `<model> (<runtime>)`). Wired to the upcoming
  `--chart <path>` CLI flag (Phase 1, item 1.2). `matplotlib` is imported lazily inside the function
  so the CLI still loads when the `[charts]` extra is not installed (spec.md I3).
- Markdown report (`benchrig.reporting.markdown.generate_markdown_report`) embeds the chart as
  `![Composite scores](chart_path)` near the top and links the CSV as `**Attachments:** [CSV](csv_path)`
  near the bottom, but only when those files were produced this run (Phase 1, item 1.3; spec.md I2).
- Warning when a benchmark target uses the `generic-cpu` execution provider on a CUDA host
  (`benchrig.core.runtimes.warning_for_provider`, wired into `benchrig.cli._warn_slow_provider`).
  Printed once per run before any model is benchmarked; the run continues and exits 0 (Phase 2,
  item 2.1; spec.md I4).
- New scorecard field `vram_baseline_dirty_mb` records the foreign-process GPU memory at the baseline
  (MiB, rounded; `None` on macOS / generic CPU; `0.0` when no foreign process holds the GPU). Surfaced
  via `BaseHardwareProvider.read_gpu_foreign_memory_mb(own_pids)` (NVIDIA: nvidia-smi
  `--query-compute-apps=pid,used_memory`) and `BenchmarkRunner.measure_vram_baseline` (Phase 2,
  item 2.2).
- Regression test pinning that `PrismClient.generate` reads the device from `telemetry.device` and
  ignores `exported_for` (Phase 4, item 4.2; spec.md I5).
- Documentation: new "Tuning the Prism server with environment variables" section in `docs/runtimes.md`
  lists `PRISM_PREFILL_CHUNK` (default `1024`), `PRISM_THREADS` (default unset), and `PRISM_DEVICE`
  (default `auto`) with their trade-offs (Phase 4, item 4.3). No CLI / code change: the env vars are
  read by prism-local itself.
- `PrismClient` retries 503 with the server-provided `Retry-After` (capped at 60 s, max 3 retries)
  via the new `benchrig.core.client._post_with_503_retry` helper. After the budget runs out the
  result is a clear failure carrying the JSON reason (e.g. `server_busy`,
  `insufficient_resources`); other 5xx are NOT retried. Wired through `FoundryClient._make_request`
  with a `PrismClient._make_request` override. New `PrismBusyError` exception is raised by the
  helper itself (Phase 4, item 4.5; spec.md I6).

## [0.1.0] - 2026-09-20

First PyPI release.

### Added
- Installable package `benchrig` (import `benchrig`) with a `benchrig` console script (also `python -m benchrig`).
- `benchrig --version`, `--config` and `--scenarios-dir`. The default `config.yaml` and all scenario suites are bundled;
  `./config.yaml`, `./scenarios` or `$BENCHRIG_CONFIG` override them.
- Direct ONNX models are looked up in `$BENCHRIG_MODEL_DIRS`, then `./models`, then `~/.benchrig/models`.
- Benchmarking across Ollama, Microsoft Foundry Local and direct ONNX Runtime GenAI (CUDA), with hardware telemetry,
  sandboxed coding tests, reasoning/speed/context/Polish suites and 1:1 cross-engine comparison reports.
- CI (Python 3.10-3.13, wheel smoke test) and a trusted-publishing workflow for TestPyPI and PyPI.

- **Prism runtime** (`--runtime prism`, `prism:` prefix, `prism:` config section): benchmark a [Prism](https://github.com/senssei/prism-local)
  server (`prism serve`) next to Ollama, Foundry Local and direct ONNX. Supports `PRISM_API_KEY`, `--check`, `--pull-recommended`
  (via `prism pull`), 1:1 pairs with a `prism:` model, and records the device Prism reports. `--models installed` skips Prism's
  `ollama:` proxies. Reports label the runtime as "Prism" and the cross-engine table covers every non-Ollama runtime.
- Documentation site (MkDocs Material, deployed to GitHub Pages) with a new [Runtimes](https://github.com/senssei/benchrig/blob/main/docs/runtimes.md) page; docs files were
  renamed to lowercase names (`docs/cli.md`, `docs/configuration.md`, ...).

### Changed (measurement methodology; results from earlier versions are not comparable)
- **Context suite:** on Ollama each step is preceded by an untimed request with the same `num_ctx`, so the timed request no longer
  includes model reload (it was about 4 to 7 s per step and dominated TTFT). Prompts are sized from the window (`fill_ratio`,
  default 0.75) instead of a fixed multiplier, the real prompt size is recorded (`prompt_tokens_actual`), and a 1024 step is added.
- **Prefill:** reports use one definition for every runtime (`prefill_eff_tok_per_sec`, prompt tokens / time to first token). The
  engine-reported value stays as `prompt_tok_per_sec`, and `prefill_source` says where it came from.
- **VRAM:** GPU memory is read before each model is loaded; scorecards add `vram_baseline_mb` and `vram_model_mb` (peak minus
  baseline). `peak_vram_mb` remains the whole-GPU peak.
- **Thinking models:** `benchmark.thinking_models` / `thinking_token_multiplier` raise `num_predict` for models such as
  `deepseek-r1` on the coding, reasoning and polish suites. Results and reports flag responses that stopped at the token limit
  (`truncated`, `truncated_runs`, `finish_reason`).

- **Repetitions (`--runs N`):** repetitions after the first send their prompts with a short marker, because Ollama caches the prompt
  prefix and a repeated prompt measured 0.006 s of prefill instead of 0.12 s. Each repetition uses its own seed
  (`execution_alignment_1to1.seed` + N), and `execution_alignment_1to1` (`temperature`, `top_p`, `seed`, `context_tokens`) is now
  actually applied as default sampling on every runtime (it was documented but unused); its unused keys were removed. Scorecards
  report the min-max of the headline metrics over repetitions (`runs`, `spread`).
- **Reasoning suite:** six new scenarios (two trains, knights and knaves, distinct-digit numbers, a recurrence, chained percentages,
  letter counting), nine in total, with a new `final_answer` check type (the last `Final answer:` line must match in full). Their
  expected answers are recomputed by independent solvers in `tests/test_scenarios.py`.
- **Ollama `think`:** the `thinking` stream is read. `ttft_sec` is now the first token of any kind (latency), with `answer_ttft_sec`,
  `think_time_sec` and `thinking_chars` alongside; a response that never left the thinking phase no longer reports 0.0 s. `benchmark.think`
  sets Ollama's `think` per suite (speed and context off by default) and is sent only to models Ollama lists as able to think.
  `thinking_token_multiplier` default 6 -> 12; it also applies with `think: false` (deepseek-r1:14b without thinking still reached the base budget).
- **Context suite:** each step hides a fact in the text and asks for it, so it scores whether the model uses the context
  (`retrieved`, `context_retrieval_pct`); wrong answers show as FAIL/"missed the fact" instead of ERROR, and failed requests keep their `error`.
- After the context suite an untimed request puts the model back at the default context size. It used to leave Ollama at `num_ctx` 8192,
  so with `--runs N` the first request of the next repetition paid a model reload (TTFT about 0.8 s instead of 0.06 s).
- **Prism token counts:** results say when the counts are estimates (`usage_estimated`, with a note in the reports) instead of silently guessing;
  Prism now sends exact `usage` and telemetry in streamed responses (`stream_options.include_usage`).
- **Model memory of a runtime that cannot unload:** when a model is still loaded at baseline time (Prism), the model-only figure is `-`
  (`vram_baseline_dirty`) instead of a value that understates the model.
- The 1:1 report also lists the min-max spread of repeated runs, and three easier reasoning scenarios (bat and ball, multiples of three or five,
  letters in one word) bring the suite to twelve.
- **Decode speed** is weighted by tokens (total tokens / total generation time); the plain mean is kept as `avg_eval_tok_sec_mean`.
- **New figures:** `cold_start_sec` (first request), `gpu_fit_pct` (Ollama `/api/ps`, share of the model in GPU memory) and
  `tokens_per_joule` (GPU power), shown in a "Start-up, GPU fit & efficiency" table.

### Fixed
- `exact_or_contains` matches numbers as whole numbers: `11/60`, `21/60` and `1/60` were accepted for `1/6`, and `1114.60` for `114.6`.
- Reasoning: `regex` answers are matched against the normalized end of the answer (markdown, bullets and line breaks collapsed),
  so a multi-line or bulleted `Box 1: ... Box 2: ...` conclusion is accepted (it was rejected because `.` did not match newlines),
  while the bundled patterns use bounded gaps and cannot be satisfied by words scattered through the reasoning (an unbounded
  multi-line match accepted wrong answers). LaTeX answers such as `\boxed{\dfrac{1}{6}}` now match `1/6`. A response cut off
  inside `<think>` is no longer scored on its reasoning trace (it produced false passes). Results record `answer_excerpt` and
  `truncated_thinking`.
- 1:1 comparison report: it compares the Ollama scorecard that matches the other model (by name and size tag) instead of the first
  one in the list, and it only uses that model's own records (it used to include every Ollama model's results, so another
  model's errors showed up in the scenario breakdown). Speed, context, reasoning and polish scenarios show OK/PASS/FAIL from
  their own fields instead of "FAIL (0/0)", cut-off answers are marked, and a Prism run is labelled "Prism", not "MS Foundry",
  in the title and in every "who is faster" cell.
- Coding: code inside `<think>` traces is ignored, and a response cut off before its code is complete is reported as truncated
  instead of `name '...' is not defined`.
- `FoundryClient` reports the engine that actually serves each model (from `owned_by`, or the `ollama:` prefix) instead of always
  "ONNX Runtime GenAI"; matters for multi-engine servers such as Prism.
- With an explicit endpoint (`auto_detect_port: false`) `load_model`/`unload_model` no longer call the `foundry` CLI (which
  talks to a different daemon); `load_model` reports whether the server lists the model.

### Changed
- The agent skills and MCP servers moved to [local-coders](https://github.com/senssei/local-coders).
- `core/` and `reporting/` moved to `benchrig.core` and `benchrig.reporting`; `benchmark.py` is now `benchrig.cli`
  (run `benchrig ...` instead of `python3 benchmark.py ...`).
- The CUDA library bootstrap looks in the active environment's `site-packages` instead of a `.venv` next to the script.
- `requirements*.txt` replaced by `pip install -e ".[dev]"`.
