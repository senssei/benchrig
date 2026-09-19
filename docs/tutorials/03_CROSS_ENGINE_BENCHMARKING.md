# ⚖️ Tutorial 3: Fair 1:1 Cross-Engine Benchmarking (Ollama vs MS Foundry)

This tutorial explains how to conduct fair, scientifically rigorous **1:1 head-to-head benchmarks** comparing **Ollama** (`llama.cpp`) against **Microsoft Foundry Local** (`ONNX Runtime GenAI`) on identical hardware.

---

## 🎯 The Importance of 1:1 Parameter Alignment

Comparing local LLM inference engines fairly requires eliminating non-deterministic variances. If one engine runs with a context window of 2048 and temperature 0.7, while another runs at context 8192 and greedy sampling (temperature 0.0), differences in throughput and code quality stem from hyperparameter divergence rather than kernel performance.

BenchRig establishes an **Execution Alignment Baseline** defined in `config.yaml`:

```yaml
execution_alignment_1to1:
  context_tokens: 4096     # Identical KV cache pre-allocation
  temperature: 0.1         # Low sampling temperature for deterministic code
  top_p: 0.9               # Standard nucleus cutoff
  seed: 42                 # Fixed RNG seed across runs
  warmup_runs: 1           # Compiles kernels and warms memory caches
  timeout_sec: 180         # Strict execution boundary
  unload_after_test: true  # Reclaims VRAM between model runs
```

---

## 🏎 Standard 1:1 Model Comparison Pairs

| Pair ID | Architecture | Ollama Model (`llama.cpp`) | MS Foundry Model (`ONNX Runtime GenAI`) |
| :--- | :--- | :--- | :--- |
| **`phi_mini`** | Phi-3 / 3.5 Mini (3.8B) | `ollama:phi3:mini` | `foundry:phi-3.5-mini` |
| **`phi4_mini`** | Phi-4 Mini (3.8B) | `ollama:phi4-mini:latest` | `foundry:phi-4-mini` |
| **`qwen_coder_7b`** | Qwen 2.5 Coder 7B | `ollama:qwen2.5-coder:7b` | `foundry:qwen2.5-coder-7b` |
| **`qwen_small`** | Qwen 3 (0.5B - 0.6B) | `ollama:qwen2.5:0.5b` | `foundry:qwen3-0.6b` |

---

## ⚡ Zero-Redundancy Workflow: Using Cached Baselines

A critical feature of BenchRig is the **`--baseline` flag**. When benchmarking new Foundry configurations or direct ONNX GPU kernels, you **never need to re-run Ollama** if a verified baseline run already exists.

### Step 1: Collect or Locate Your Ollama Baseline Run
When an Ollama run completes, it is saved in `results/runs/benchmark_<timestamp>.json`. Inspect existing runs:
```bash
python3 -c "
import json, glob
for path in sorted(glob.glob('results/runs/*.json')):
    with open(path) as f:
        data = json.load(f)
    models = [sc.get('model') for sc in data.get('scorecards', [])]
    print(f'{path} -> {models}')
"
```

### Step 2: Evaluate MS Foundry Against the Baseline
Run the benchmark exclusively for Foundry, passing the baseline file:

```bash
benchrig \
  --runtime foundry \
  --models foundry:phi-4-mini \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

BenchRig will:
1. **Load only MS Foundry**: Ollama is untouched and idle.
2. **Execute the selected suite**: Runs test assertions through `FoundryClient`.
3. **Merge scorecards**: Combines the cached Ollama metrics with the fresh Foundry metrics.
4. **Render the side-by-side terminal table**.
5. **Generate `results/1TO1_COMPARISON_REPORT.md`**.

---

## 📊 Reading the 1:1 Comparative Scorecard

When evaluation concludes, BenchRig outputs a side-by-side comparative table:

```text
   ⚖️ 1:1 Cross-Engine Model Comparison: phi3:mini (Ollama) vs phi-4-mini (MS Foundry)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Evaluation Metric ┃         phi3:mini ┃        phi-4-mini ┃ Delta /          ┃
┃                   ┃           (Ollama ┃ (MS Foundry (ONNX ┃ Advantage        ┃
┃                   ┃      (llama.cpp)) ┃         Runtime)) ┃                  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Decode Speed      │          94.0 t/s │         130.2 t/s │ Foundry 1.4x     │
│ (tok/s)           │                   │                   │ faster           │
│ Prefill Speed     │        2639.8 t/s │        1006.8 t/s │ Ollama 2.6x      │
│ (tok/s)           │                   │                   │ faster           │
│ Avg TTFT          │             0.21s │             0.09s │ Foundry 2.3x     │
│ (Latency)         │                   │                   │ lower            │
│ Coding Pass Rate  │             17.6% │            100.0% │ Foundry +82.4%   │
│ Peak Memory Usage │   11135 MB (VRAM) │          10875 MB │ GPU vs CPU/RAM   │
│                   │                   │        (RAM/VRAM) │                  │
│ Composite Score   │          31.1/100 │          64.0/100 │ Foundry (+32.9)  │
└───────────────────┴───────────────────┴───────────────────┴──────────────────┘
```

### Key Metrics Explained:
1. **Decode Speed (Generation)**: Rate of token emission during text synthesis (`eval_tokens / eval_duration_sec`). Measures interactive responsiveness.
2. **Prompt Prefill Speed**: Rate of ingesting and embedding the input prompt (`prompt_tokens / prompt_eval_duration_sec`). Measures context processing speed.
3. **Time to First Token (TTFT)**: Elapsed wall time from request dispatch to the arrival of the first output token. Slashed by GPU prefill acceleration.
4. **Coding Pass Rate**: Percentage of deterministic unit tests passed in the isolated sandbox (`benchrig/core/sandbox.py`).
5. **Peak VRAM / Memory Fit**: Maximum hardware allocation recorded during execution.

---

## 🔄 Re-Analyzing Historical Runs (`--compare`)

To re-display terminal tables and re-generate `1TO1_COMPARISON_REPORT.md` without invoking any inference engines:

```bash
benchrig --compare results/latest.json
# Or specify any historical run:
benchrig --compare results/runs/benchmark_20260918_235227.json
```
