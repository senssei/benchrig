# 🖥 CLI Reference & Usage Manual

This manual documents the command-line interface for **`benchmark.py`**, including runtime selection, model specification, test suite filters, and automation recipes.

---

## 1. Synopsis

```bash
python3 benchmark.py [OPTIONS]
```

---

## 2. Command-Line Options

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **`--runtime`** | `ollama` \| `foundry` \| `onnx-gpu` \| `all` | Config default (`ollama`) | Selects the active inference runtime engine. `onnx-gpu` runs direct ONNX GenAI CUDA kernels. `all` runs cross-engine comparison. |
| **`--models`** | `string` | `installed` | Comma-separated list of models to evaluate. Supports runtime prefixes (e.g. `foundry:phi-4-mini` or `ollama:phi3:mini`). Defaults to all installed models. |
| **`--suite`** | `all` \| `speed` \| `coding` \| `reasoning` \| `context` \| `polish` | `all` | Filters the benchmark to a specific evaluation domain. |
| **`--runs`** | `integer` | `1` | Number of test repetitions per model scenario for variance calculation. |
| **`--check`** | `flag` | `false` | Runs non-destructive environment diagnostics, accelerator detection, and runtime connectivity verification. |
| **`--pull-recommended`** | `flag` | `false` | Automatically downloads and pulls recommended benchmark models configured in `config.yaml`. |
| **`--baseline`** | `path` | `None` | Path to a cached benchmark JSON run providing Ollama metrics so Ollama is never re-run. |
| **`--compare`** | `path` | `None` | Path to historical benchmark JSON run to load, analyze, and render 1:1 report without running any LLMs. |
| **`--pair`** | `string` | `None` | 1:1 model comparison pair ID from `config.yaml` (e.g. `phi_mini`, `phi4_mini`, `qwen_coder_7b`). |
| **`--output-dir`** | `string` | `results` | Path to directory where run summaries (`LATEST_SUMMARY.md`) and raw JSON run data are persisted. |

---

## 3. Practical Recipes

### 1. Environment Readiness & Diagnostics
Inspect GPU memory, platform drivers, and runtime availability:
```bash
python3 benchmark.py --check
```

### 2. Fast Speed & Latency Benchmark
Evaluate raw token throughput and Time-to-First-Token (TTFT) on Ollama models:
```bash
python3 benchmark.py --runtime ollama --suite speed --runs 1
```

### 3. Head-to-Head 1:1 Comparison Using Cached Baseline (Zero Redundancy)
Evaluate Microsoft Foundry Local (or Direct ONNX GenAI) against a cached Ollama run without re-running Ollama:
```bash
python3 benchmark.py \
  --runtime foundry \
  --models foundry:phi-4-mini \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

### 4. Direct ONNX Runtime GenAI GPU Evaluation
Evaluate HuggingFace ONNX INT4 AWQ models directly with native CUDA acceleration:
```bash
python3 benchmark.py \
  --runtime onnx-gpu \
  --models Phi-4-mini-instruct-cuda-gpu \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

### 5. Re-Rendering Reports & Scores Without Inference
Re-display terminal leaderboards and re-generate `1TO1_COMPARISON_REPORT.md` instantly:
```bash
python3 benchmark.py --compare results/latest.json
```

### 6. Sandboxed Coding Precision Suite
Run algorithmic coding challenges with automated unit test assertions in isolated subshells:
```bash
python3 benchmark.py --runtime ollama \
  --models "qwen2.5-coder:7b" \
  --suite coding
```

### 7. Automated Setup of Recommended Models
Pull standard benchmark models configured in `config.yaml`:
```bash
python3 benchmark.py --runtime ollama --pull-recommended
```

---

## 4. Output Artifacts

Upon completing an evaluation, `benchmark.py` generates:
1. **Interactive Terminal Leaderboard**: Formatted using Rich with medals (🥇, 🥈, 🥉), composite rankings, VRAM status alerts, and cloud cost savings estimates.
2. **`results/LATEST_SUMMARY.md`**: Markdown report detailing individual scenario assertion passes, TTFT, speed, and cross-engine comparison deltas.
3. **`results/runs/benchmark_<timestamp>.json`**: Raw machine-readable telemetry and scores for CI/CD or historical trend analysis.
