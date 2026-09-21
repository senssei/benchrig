# Intent: BenchRig

> **Status: draft.** The operator approves any change to this file (see `AGENTS.md`, operator gates).

## 1. Problem

Picking a local LLM runtime (Ollama, Microsoft Foundry Local, ONNX Runtime GenAI) is a guess, not a measurement. Today:

- **Cross-runtime quality gaps are undocumented in numbers.** On reasoning scenarios, ONNX int4 scored 58.3% vs Ollama's 77.8% on the same prompt template — but the cause is unknown (RTN quantization vs runtime numerics); we have no fp16 ONNX or matched-quantization Ollama variant to isolate it (`scratch/TODO.md` §4).
- **`generic-cpu` execution provider on CUDA is unusable for some models.** qwen variants reach 2–22 tok/s, Phi-3.5-mini fails to finish 4500 tokens in 5 minutes; today the user has to discover this by hand.
- **Results are only Markdown and terminal.** There is no chart or CSV export from `results/runs/*.json`, so spread, time-of-day, and hardware comparisons live in chat history instead of files (`scratch/TODO.md` §4).
- **VRAM baselines are noisy.** When the system holds other GPU consumers (e.g. two `prism.cli mcp` instances from an IDE ≈ 3 GB), the dirty-baseline subtraction works but the residual still varies between runs.

For whom: developers running local LLMs on macOS Apple Silicon (M1/M2/M3/M4) or Linux/WSL2 (NVIDIA RTX) who need reproducible, comparable numbers before picking a runtime.

## 2. Outcome

BenchRig is a single CLI (`benchrig`) that, on a real workload (not perplexity), produces reproducible scorecards across Ollama, Foundry Local, ONNX Runtime GenAI and Prism. Each scorecard carries composite score, coding pass rate, reasoning accuracy, TTFT, prompt/eval tok·s, peak VRAM and a JSON run record. Reports are Markdown **and** charts/CSV. The same model on two runtimes, run twice with `--runs 3`, has a publishable spread.

## 3. Constraints

1. Local-only; no cloud LLM traffic. The MIT-licensed `benchrig` PyPI package, Python 3.10+.
2. Supported hardware: macOS Apple Silicon (Metal & Unified Memory) and Linux / WSL2 (NVIDIA CUDA, optional TensorRT). `AGENTS.md` §3 lists the runtime diagnostics required per platform.
3. Verification gate before any release: `ruff check . && ruff format --check . && python3 -m pytest && python3 -m build` (`AGENTS.md` §5). The SDLC gate (`python3 scripts/sdlc_check.py`) enforces the same checks on every commit.
4. Suites are deterministic: sandboxed code execution, isolated reasoning scenarios, fixed seeds where the model permits. No flaky-by-design benchmarks.
5. Numbers reported are honest: estimated tokens are flagged (`usage_estimated`), dirty baselines are flagged (`vram_baseline_dirty`), truncated runs are counted, never hidden.

## 4. Non-goals

1. Cloud LLM benchmarking (OpenAI, Anthropic, etc.) — local-only by license and runtime.
2. Production deployment of LLMs — BenchRig measures; it does not serve.
3. Model training, fine-tuning, or dataset curation — out of scope.
4. A web dashboard — reports are file artifacts (Markdown, PNG, CSV) under `results/`.
5. Replacing `ollama` / `foundry` / `prism` CLIs — BenchRig calls them, it does not reimplement them.

## 5. Success criteria

| Criterion | Evidence |
|---|---|
| One CLI produces reproducible scorecards across all four runtimes on the same hardware | `benchrig --runtime all --runs 3 …` exits 0 and emits `results/runs/<ts>.json` plus `results/LATEST_SUMMARY.md`; two consecutive runs on the same model show a publishable spread (TODO §2 row 1) |
| Reasoning-quality gap (ONNX int4 vs Ollama) is root-caused or bounded | A comparison artifact in `results/` with matched quantization, or an upstream issue filed against `onnxruntime-genai` with measurements (TODO §4 row 1 / row 3) |
| `generic-cpu` on CUDA either succeeds in bounded time or warns/skips before running | Exit code or warning printed by `benchrig`; covered by `tests/test_warnings.py` (TODO §4 row 2) |
| Result runs are exportable as charts and CSV | `benchrig --chart path.png --csv path.csv` (or equivalents) produce files; covered by `tests/test_report_charts.py` and `tests/test_report_csv.py` (TODO §4 row 6) |
| VRAM baseline is reported as `dirty` whenever foreign GPU processes are present | Field present in scorecards; covered by `tests/test_hardware.py` (TODO §2 row 2 / §4 row 5) |
| Every change passes the SDLC gate before commit | `python3 scripts/sdlc_check.py` exits 0 in the session that wrote the change (sdlc.toml, REVIEW.md §A) |