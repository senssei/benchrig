# 🖥 CLI Reference & Usage Manual

This manual documents the command-line interface for **`benchrig`**, including runtime selection, model specification, test suite filters, and automation recipes.

---

## 1. Synopsis

```bash
benchrig [OPTIONS]
```

---

## 2. Command-Line Options

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`--runtime`** | `ollama` \| `foundry` \| `onnx-gpu` \| `prism` \| `all` | Config default (`ollama`) | Selects the active inference runtime engine. `onnx-gpu` runs direct ONNX GenAI CUDA kernels. `prism` benchmarks a running Prism server. `all` runs cross-engine comparison. |
| **`--models`** | `string` | `installed` | Comma-separated list of models to evaluate. Supports runtime prefixes (e.g. `foundry:phi-4-mini` or `ollama:phi3:mini`). Defaults to all installed models. |
| **`--suite`** | `all` \| `speed` \| `coding` \| `reasoning` \| `context` \| `polish` | `all` | Filters the benchmark to a specific evaluation domain. |
| **`--runs`** | `integer` | `1` | Number of repetitions of every scenario. Each repetition uses its own seed and (after the first) a prompt marker that defeats Ollama's prompt cache; scorecards then report the min-max of the headline metrics over the repetitions. Use `--runs 3` before quoting differences of a few percent. |
| **`--check`** | `flag` | `false` | Runs non-destructive environment diagnostics, accelerator detection, and runtime connectivity verification. |
| **`--pull-recommended`** | `flag` | `false` | Automatically downloads and pulls recommended benchmark models configured in `config.yaml`. |
| **`--baseline`** | `path` | `None` | Path to a cached benchmark JSON run providing Ollama metrics so Ollama is never re-run. |
| **`--compare`** | `path` | `None` | Path to historical benchmark JSON run to load, analyze, and render 1:1 report without running any LLMs. |
| **`--pair`** | `string` | `None` | 1:1 model comparison pair ID from `config.yaml` (e.g. `phi_mini`, `phi4_mini`, `qwen_coder_7b`). |
| **`--config`** | `path` | `None` | Config file. Lookup order: `--config`, `$BENCHRIG_CONFIG`, `./config.yaml`, then the bundled default. |
| **`--scenarios-dir`** | `path` | `None` | Scenario JSON directory. Lookup order: `--scenarios-dir`, `./scenarios`, then the bundled scenarios. |
| **`--version`** | `flag` | `false` | Prints the installed version and exits. |
| **`--output-dir`** | `string` | `results` | Path to directory where run summaries (`LATEST_SUMMARY.md`) and raw JSON run data are persisted. |
| **`--csv`** | `path` | `None` | Writes a CSV of the run's scorecards to this path and links it from `LATEST_SUMMARY.md`. |
| **`--chart`** | `path` | `None` | Writes a PNG chart (one bar per scorecard) to this path and embeds it in `LATEST_SUMMARY.md`. Requires the `[charts]` extra (`pip install -e ".[charts]"`). |
| **`--log-level`** | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` | `WARNING` (env: `BENCHRIG_LOG_LEVEL`) | Minimum structured-log level on stderr. Default `WARNING` keeps stderr silent during normal runs; `DEBUG` or `INFO` emit one JSON object per diagnostic event (HTTP lifecycle, retries, capacity waits, run lifecycle). Env var `BENCHRIG_LOG_LEVEL` overrides the default; `--log-level` overrides the env. See [§5 Logging](#5-logging). |

---

## 3. Practical Recipes

### 1. Environment Readiness & Diagnostics
Inspect GPU memory, platform drivers, and runtime availability:
```bash
benchrig --check
```

### 2. Fast Speed & Latency Benchmark
Evaluate raw token throughput and Time-to-First-Token (TTFT) on Ollama models:
```bash
benchrig --runtime ollama --suite speed --runs 1
```

### 3. Head-to-Head 1:1 Comparison Using Cached Baseline (Zero Redundancy)
Evaluate Microsoft Foundry Local (or Direct ONNX GenAI) against a cached Ollama run without re-running Ollama:
```bash
benchrig \
  --runtime foundry \
  --models foundry:phi-4-mini \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

### 4. Direct ONNX Runtime GenAI GPU Evaluation
Evaluate HuggingFace ONNX INT4 AWQ models directly with native CUDA acceleration:
```bash
benchrig \
  --runtime onnx-gpu \
  --models Phi-4-mini-instruct-cuda-gpu \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

### 5. Benchmarking Through a Prism Server
Start [Prism](https://github.com/senssei/prism-local) (`pip install prism-local && prism serve`), then:
```bash
benchrig --runtime prism --models prism:phi-4-mini --suite coding
benchrig --runtime prism --models installed        # every ONNX model Prism serves (its `ollama:` proxies are skipped)
benchrig --runtime prism --pair phi4_mini --baseline results/runs/benchmark_20260918_212608.json
```
Results record which engine served each model and, when Prism reports it, the device that ran the request.

### 6. Re-Rendering Reports & Scores Without Inference
Re-display terminal leaderboards and re-generate `1TO1_COMPARISON_REPORT.md` instantly:
```bash
benchrig --compare results/latest.json
```

### 7. Sandboxed Coding Precision Suite
Run algorithmic coding challenges with automated unit test assertions in isolated subshells:
```bash
benchrig --runtime ollama \
  --models "qwen2.5-coder:7b" \
  --suite coding
```

### 8. Automated Setup of Recommended Models
Pull standard benchmark models configured in `config.yaml`:
```bash
benchrig --runtime ollama --pull-recommended
```

---

## 4. Output Artifacts

Upon completing an evaluation, `benchrig` generates:
1. **Interactive Terminal Leaderboard**: Formatted using Rich with medals (🥇, 🥈, 🥉), composite rankings, VRAM status alerts, and cloud cost savings estimates.
2. **`results/LATEST_SUMMARY.md`**: Markdown report detailing individual scenario assertion passes, TTFT, speed, and cross-engine comparison deltas.
3. **`results/runs/benchmark_<timestamp>.json`**: Raw machine-readable telemetry and scores for CI/CD or historical trend analysis.
4. **`--csv`/`--chart` (opt-in)**: a scorecards CSV and/or a PNG bar chart, written where requested and linked/embedded from `LATEST_SUMMARY.md`.

---

## 5. Logging

Structured diagnostic logs are emitted to **stderr** as one JSON object per line. The colored
terminal UX on **stdout** is unchanged.

Enable a more verbose log via either:

```bash
benchrig --log-level INFO …                                   # CLI flag (wins over env)
BENCHRIG_LOG_LEVEL=DEBUG benchrig …                            # env var only
```

**Stable record shape** (every record carries these; per-event extras add more):

```json
{"ts": "2026-09-23T18:42:33.123Z", "level": "INFO",
 "event": "http.request_started", "run_id": "…uuid…",
 "message": "…", "model": "phi-4-mini", "runtime": "prism", "engine": "ONNX Runtime GenAI",
 "url": "http://127.0.0.1:5272/v1/chat/completions", "attempt": 1}
```

| Event | Level | When |
|---|---|---|
| `run.started` | INFO | Top of a `benchrig` invocation (`run_benchmarks`) |
| `run.completed` | INFO | End of a run, including on mid-run crashes (`finally:`) |
| `http.request_started` / `http.response_completed` / `http.request_failed` | INFO / INFO / WARNING | Every HTTP call from `benchrig.core.client`; `http.response_completed` only for 2xx, `http.request_failed` otherwise (DEBUG for GET health/version probes) |
| `retry.attempted` / `retry.exhausted` | INFO / WARNING | Prism 503 retry helper |
| `capacity.exhausted` | WARNING | Persistent `insufficient_resources` short-circuit |
| `unload.completed` | DEBUG | `PrismClient.unload_model` returning |

**Notes**

- Stdout (`rich.console`) is unchanged; nothing duplicates to stderr by default at WARNING.
- `run_id` is a single UUID per `benchrig` invocation; use it to filter a run out of a noisy shared log target.
- Trace context, file output, and OTLP export are **out of scope** for this phase; pipe stderr to `jq` for ad-hoc filtering:
  ```bash
  benchrig --log-level INFO 2> >(jq -c 'select(.event=="http.request_failed")') …
  ```
