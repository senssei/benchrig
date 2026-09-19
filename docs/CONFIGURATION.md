# ⚙️ Configuration Reference Guide

This document provides a comprehensive reference for configuring **BenchRig** via [`config.yaml`](../benchrig/data/config.yaml) and environment variables.

---

## 📋 Configuration File Overview (`config.yaml`)

BenchRig uses a centralized YAML configuration file to control runtime connectivity, timeout thresholds, benchmark execution defaults, hardware telemetry sampling, and composite scoring formulas.

```yaml
# ==============================================================================
# BenchRig - Global Runtime & Benchmark Configuration
# ==============================================================================

ollama:
  base_url: "http://localhost:11434"
  timeout_sec: 180
  warmup: true             # Pre-warms weights into VRAM before timing
  unload_after_test: true  # Evicts model from VRAM between benchmark runs
  default_num_ctx: 4096    # Default context window size

foundry:
  base_url: "http://localhost:5272/v1"  # Default Microsoft Foundry Local endpoint
  auto_detect_port: true                # Reads active port from ~/.foundry/daemon.json
  cli_path: "foundry"                   # Foundry CLI binary name or absolute path
  timeout_sec: 180
  warmup: true
  unload_after_test: true

benchmark:
  default_runs: 1                       # Number of benchmark iterations per test
  default_runtime: "ollama"             # Default runtime: "ollama", "foundry", or "all"
  composite_weights:
    coding: 0.40                        # 40% weight for coding unit test pass rate
    reasoning: 0.30                     # 30% weight for reasoning ground-truth accuracy
    performance: 0.30                   # 30% weight for normalized throughput & TTFT

hardware:
  sample_interval_sec: 0.15             # Telemetry polling frequency (seconds)
  vram_warning_threshold_mb: "auto"     # "auto" defaults to 88% of detected VRAM/UMA
```

---

## 🔧 Section-by-Section Reference

### 1. `ollama` Configuration

Controls connectivity and lifecycle management for the local [Ollama](https://ollama.com) daemon (`llama.cpp` backend).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `base_url` | `string` | `"http://localhost:11434"` | Base URL where the Ollama HTTP REST API is listening. Overridden by `OLLAMA_HOST` if defined. |
| `timeout_sec` | `integer` | `180` | Maximum socket timeout in seconds for prompt generation and model pulling requests. |
| `warmup` | `boolean` | `true` | When `true`, executes a non-timed single-token prompt before benchmarking to ensure model weights and KV cache are loaded into VRAM/UMA. |
| `unload_after_test` | `boolean` | `true` | When `true`, explicitly unloads the model from memory after each test iteration by issuing a request with `keep_alive: 0`. Prevents VRAM fragmentation. |
| `default_num_ctx` | `integer` | `4096` | Default context window length in tokens allocated for inference requests unless explicitly overridden by a scenario. |

---

### 2. `foundry` Configuration

Controls connectivity, execution provider negotiation, and port discovery for **Microsoft Foundry Local** (`ONNX Runtime GenAI` backend).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `base_url` | `string` | `"http://localhost:5272/v1"` | Fallback endpoint for Foundry Local OpenAI-compatible REST server. |
| `auto_detect_port` | `boolean` | `true` | When `true`, reads the dynamic listening port and endpoint URL directly from `~/.foundry/daemon.json`. Prevents connection failures when Foundry starts on ephemeral ports (e.g. 51834, 5272). |
| `cli_path` | `string` | `"foundry"` | Path to the `foundry` CLI executable. Can be a bare command name (if located in `$PATH`) or an absolute path (e.g. `~/.local/bin/foundry`). |
| `timeout_sec` | `integer` | `180` | Request timeout in seconds for Foundry completions and model queries. |
| `warmup` | `boolean` | `true` | Warms up ONNX graph execution and execution provider memory pools prior to recorded benchmark runs. |
| `unload_after_test` | `boolean` | `true` | Releases model session from memory between runs to ensure clean hardware telemetry. |

---

### 3. `benchmark` Configuration

Controls test orchestration, iteration counts, and composite score weighting.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `default_runs` | `integer` | `1` | Default number of iterations executed per test scenario. When greater than 1, metrics (throughput, TTFT, memory) are averaged. |
| `default_runtime` | `string` | `"ollama"` | Target runtime engine when `--runtime` is omitted on the CLI. Supported values: `"ollama"`, `"foundry"`, `"all"`. |
| `composite_weights.coding` | `float` | `0.40` | Relative weighting (0.0 to 1.0) of automated unit test execution pass rate in the final composite score. |
| `composite_weights.reasoning` | `float` | `0.30` | Relative weighting (0.0 to 1.0) of multi-step logical deduction and mathematical ground truth accuracy. |
| `composite_weights.performance`| `float` | `0.30` | Relative weighting (0.0 to 1.0) of generation throughput, prefill throughput, and low Time to First Token (TTFT). |

#### Composite Score Calculation Formula
$$\text{Composite Score} = (W_{\text{code}} \times \text{PassRate}) + (W_{\text{reason}} \times \text{Accuracy}) + (W_{\text{perf}} \times \text{ThroughputFactor})$$

Where:
- $\text{PassRate} = \frac{\text{Passed Coding Tests}}{\text{Total Coding Tests}} \times 100$
- $\text{Accuracy} = \frac{\text{Passed Reasoning Tests}}{\text{Total Reasoning Tests}} \times 100$
- $\text{ThroughputFactor} = \min\left(100, \frac{\text{Eval Tokens/sec}}{80} \times 100\right)$

---

### 4. `onnx` Configuration (Direct ONNX Runtime GenAI)

Controls direct execution through Python bindings to `onnxruntime-genai-cuda` (**Option 3**).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `models_dir` | `string` | `"models"` | Directory path where HuggingFace ONNX INT4 AWQ models are located. |
| `timeout_sec` | `integer` | `300` | Execution timeout in seconds for direct ONNX generation. |
| `default_max_tokens` | `integer` | `4096` | Max tokens limit for generation. |
| `warmup` | `boolean` | `true` | Warms up ONNX graph execution and allocator pools before timing. |
| `unload_after_test` | `boolean` | `true` | Releases model session from memory between runs. |

---

### 5. `execution_alignment_1to1` Configuration

Standardizes inference parameters for fair cross-engine comparisons:

```yaml
execution_alignment_1to1:
  context_tokens: 4096     # Fixed context window size
  temperature: 0.1         # Low temperature for deterministic code
  top_p: 0.9               # Standard nucleus sampling
  seed: 42                 # Fixed RNG seed
  warmup_runs: 1           # Kernel compilation and cache warm-up
  timeout_sec: 180         # Socket timeout
  unload_after_test: true  # Reclaims memory between model runs
```

---

### 6. `model_pairs_1to1` Configuration

Defines cross-engine 1:1 comparison pairs between Ollama and Microsoft Foundry Local / ONNX GenAI:

| Pair ID | Architecture | Ollama Model | Foundry Model | Recommended Suite |
| :--- | :--- | :--- | :--- | :--- |
| `phi_mini` | Phi-3 / 3.5 Mini (3.8B) | `phi3:mini` | `phi-3.5-mini` | `coding` |
| `phi4_mini` | Phi-4 Mini (3.8B) | `phi4-mini:latest` | `phi-4-mini` | `coding` |
| `qwen_coder_7b` | Qwen 2.5 Coder 7B | `qwen2.5-coder:7b` | `qwen2.5-coder-7b` | `coding` |
| `qwen_small` | Qwen 3 (0.5B - 0.6B) | `qwen2.5:0.5b` | `qwen3-0.6b` | `coding` |

---

### 7. `hardware` Configuration

Controls the background hardware sampling thread and memory capacity warning thresholds.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sample_interval_sec` | `float` | `0.15` | Polling frequency in seconds for hardware monitoring threads (`DarwinAppleSiliconProvider` and `LinuxNvidiaProvider`). Balanced to capture transient spikes without consuming measurable CPU cycles. |
| `vram_warning_threshold_mb` | `string` \| `integer` | `"auto"` | Memory threshold at which BenchRig issues a visual alert in terminal and reports. When set to `"auto"`, the runner calculates $88\%$ of detected VRAM or Unified Memory. Can also be set to a fixed integer in megabytes (e.g. `10240` for 10 GB). |

---

## 🌐 Environment Variables

Environment variables take precedence over settings defined in `config.yaml`:

| Variable | Target Parameter | Description |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `ollama.base_url` | Full URL or host:port where Ollama is reachable (e.g. `http://192.168.1.100:11434` or `127.0.0.1:11434`). |
| `FOUNDRY_BASE_URL` | `foundry.base_url` | Overrides the base URL for the Microsoft Foundry REST endpoint. |
| `LD_LIBRARY_PATH` | System Dynamic Linker | Must include paths to CUDA, cuDNN, and TensorRT shared objects (`.so`) when running on Linux / WSL2. |
| `CUDA_VISIBLE_DEVICES` | System GPU Index | Restricts GPU enumeration to specific device indices (e.g. `CUDA_VISIBLE_DEVICES=0`). |

---

## 🔄 Dynamic Port Auto-Discovery Workflow

When Microsoft Foundry Local daemon starts, it may bind to an ephemeral port to avoid conflicts. With `auto_detect_port: true`, BenchRig locates the daemon configuration file:

```mermaid
flowchart TD
    A["BenchmarkRunner / FoundryClient"] --> B{"auto_detect_port == true?"}
    B -- No --> C["Use config.yaml base_url"]
    B -- Yes --> D["Read ~/.foundry/daemon.json"]
    D --> E{"File exists & contains port?"}
    E -- Yes --> F["Construct dynamic base_url: http://localhost:{port}/v1"]
    E -- No --> G["Fallback to config.yaml base_url"]
    F --> H["Check HTTP /v1/models endpoint"]
    G --> H
    H --> I["Connected Successfully"]
```

---

## 🧹 Memory Management & VRAM Hygiene

Running multiple large language models sequentially can cause out-of-memory (OOM) faults or memory fragmentation if previous models linger in VRAM:

1. **Pre-test Warmup (`warmup: true`)**:
   Sends a minimal generation query (`"hi"`) to trigger model loading and weight caching before timing begins, preventing model loading I/O latency from corrupting TTFT and decode throughput benchmarks.
2. **Post-test Unload (`unload_after_test: true`)**:
   - **Ollama**: Sends `{"model": "<name>", "keep_alive": 0}` to the `/api/generate` endpoint, instructing Ollama to immediately evict the model weights from GPU memory.
   - **Foundry Local**: Signals model session teardown to reset execution provider allocator pools.
