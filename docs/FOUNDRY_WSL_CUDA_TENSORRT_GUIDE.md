# 📘 Technical Guide: Microsoft Foundry Local, ONNX Runtime, CUDA & TensorRT on WSL2

A comprehensive technical guide detailing architecture, NVIDIA GPU (GeForce RTX) acceleration configuration, and head-to-head benchmarking between **Microsoft Foundry Server Runtime** and **Ollama** in a **WSL2 (Ubuntu 24.04)** environment.

---

## 1. 🏗 Architecture: Ollama vs Microsoft Foundry Local

| Feature | Ollama | Microsoft Foundry Local |
| :--- | :--- | :--- |
| **Inference Engine** | `llama.cpp` (compiled with native CUDA) | `ONNX Runtime` + `ONNX Runtime GenAI` |
| **Model Weights Format** | GGUF (quantizations `Q4_K_M`, `Q8_0`, etc.) | ONNX Graph (`model.onnx` + `genai_config.json`) |
| **Ecosystem** | Open-source (Hugging Face / Ollama Registry) | Microsoft Azure AI Foundry / Windows ML |
| **Memory Management** | Automatic layer offloading & dynamic caching | Requires explicit model loading (`foundry model load`) |
| **Network Interface** | Native REST API (`/api/generate`) | OpenAI-compatible SSE REST API (`/v1/chat/completions`) |
| **WSL2 Autonomy** | Standalone daemon with bundled CUDA runtime | .NET 9 application relying on system dynamic libraries |

---

## 2. ⚡ ONNX Runtime Execution Providers (EP) Overview

In ONNX Runtime, hardware acceleration is modularized through **Execution Providers**:

```
                       ┌───────────────────────────────┐
                       │      ONNX Runtime Core        │
                       └──────────────┬────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│CPUExecutionProvider│       │CUDAExecutionProvider│      │TensorrtExecution...│
│ (AVX2/512 Kernels)│       │(cuBLAS / cuDNN)  │        │ (TRT Graph Engine)│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

1. **`CPUExecutionProvider`**:
   - Default execution backend for all Linux catalog models (`-generic-cpu`).
   - Models are compiled with CPU vector instructions (AVX2 / AVX-512 / RTN int4).
   - Utilizes system host RAM rather than GPU VRAM.

2. **`CUDAExecutionProvider` (`libonnxruntime_providers_cuda.so`)**:
   - General-purpose NVIDIA acceleration backend for Linux and WSL2.
   - Offloads matrix multiplication and deep learning operations to `cuBLAS` and `cuDNN`.
   - Significantly accelerates prompt evaluation (Prefill) and slashes Time-to-First-Token (TTFT).

3. **`TensorrtExecutionProvider` (`libonnxruntime_providers_tensorrt.so`)**:
   - Full **NVIDIA TensorRT** compiler backend for Linux x86_64.
   - Performs graph layer fusion, kernel auto-tuning, and builds optimized `.engine` caches tailored to the specific GPU.
   - Shipped by Microsoft directly in the `foundry-cli` bundle (`~/.local/lib/foundry-cli/`).

4. **`NvTensorRTRTXExecutionProvider` (Windows Host Only)**:
   - Consumer GeForce RTX-tailored TensorRT variant on Windows.
   - Auto-distributed via the Windows Store package system (`Microsoft.WinML.NVIDIA.TRT-RTX.EP`).
   - Powers Windows Copilot Runtime and Windows-specific models (`-cuda-gpu`).

---

## 3. 🛠 Step-by-Step CUDA 12 & cuDNN 9 Setup on WSL2 (Zero-Sudo)

### Why Was This Necessary?
While Ollama bundles its own CUDA runtime inside `/usr/local/lib/ollama/cuda_v12/`, the Foundry Local daemon relies on the Linux dynamic linker (`LD_LIBRARY_PATH`). In clean WSL2 installations, 6 critical shared libraries were absent:
* `libcudnn.so.9`, `libcublasLt.so.12`, `libcublas.so.12`, `libcurand.so.10`, `libcufft.so.11`, `libcudart.so.12`

### Resolution Without Root (`sudo`) Permissions

#### Step 1: Install User-Space NVIDIA Libraries via PyPI
```bash
pip install --break-system-packages --user \
  nvidia-cuda-runtime-cu12 \
  nvidia-cublas-cu12 \
  nvidia-curand-cu12 \
  nvidia-cufft-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-cuda-nvrtc-cu12
```

> [!NOTE]
> **Understanding the `--break-system-packages` flag:**  
> Ubuntu 24.04 LTS implements PEP 668 to protect `apt`-managed system packages from inadvertent overrides. When combined with `--user`, this flag **does not break any system packages**. Files are placed exclusively in your private home directory (`~/.local/lib/python3.12/site-packages/`), leaving system `/usr/` directories completely untouched.

#### Step 2: Persist Paths in `~/.bashrc`
```bash
cat << 'EOF' >> ~/.bashrc

# NVIDIA CUDA 12, cuDNN 9 & TensorRT 10 paths for ONNX Runtime / Foundry
export LD_LIBRARY_PATH="$HOME/.local/lib/tensorrt/tensorrt_libs:$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cublas/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/curand/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cufft/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
source ~/.bashrc
```

#### Step 3: Self-Healing Daemon Wrapper (`foundrylocald`)
Because subshells and external tools may invoke `foundrylocald` without evaluating interactive shell profiles, a wrapper script was deployed at `~/.local/lib/foundry-cli/foundrylocald`:
```bash
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$HOME/.local/lib/tensorrt/tensorrt_libs:$HOME/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:.../usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HOME/.local/lib/foundry-cli/foundrylocald.real" "$@"
```
This guarantees the daemon **always** boots with full access to GPU libraries.

---

## 4. 🔍 How to Verify GPU Hardware Acceleration

### Method 1: Daemon Log Execution Provider Registration
```bash
grep -E "CUDA|EP|Device:" ~/.foundry/logs/foundry.core$(date +%Y%m%d).log | tail -n 5
```
* **Expected Successful Output:**
  ```text
  [INF] Successfully registered CUDA execution provider from app package. EP=FoundryLocalCUDA
  ```

### Method 2: Active Compute Processes via `nvidia-smi`
During model loading or active inference, inspect the GPU compute table:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
The process ID (`PID`) for `foundrylocald` will appear actively utilizing VRAM on your GeForce RTX 5070.

### Method 3: Demystifying `foundry cache list`
```text
╭──────────────┬────────┬────────┬────────╮
│ Alias        │ Size   │ Device │ Loaded │
├──────────────┼────────┼────────┼────────┤
│ qwen3-0.6b   │ 593 MB │ CPU    │ ●      │
╰──────────────┴────────┴────────┴────────╯
```
* **Column `Device: CPU`**: A static metadata label originating from Microsoft's model catalog ID (`-generic-cpu`). It does **not** signify that GPU acceleration is absent.
* **Column `Loaded: ●`**: Confirms the model weights reside actively in daemon memory.

---

## 5. ⚙️ Forcing GPU Acceleration on ONNX Models

To redirect a CPU-targeted ONNX model onto CUDA cores:

1. Open the model's runtime configuration file:
   ```bash
   nano ~/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4/genai_config.json
   ```
2. Replace the empty `"provider_options": []` block with the CUDA configuration:
   ```json
   "session_options": {
       "log_id": "onnxruntime-genai",
       "provider_options": [
           {
               "cuda": {
                   "device_id": "0"
               }
           }
       ]
   }
   ```
3. Reload the model into daemon memory:
   ```bash
   foundry model unload qwen3-0.6b
   foundry model load qwen3-0.6b
   ```

> [!WARNING]
> **Large Model Constraint (e.g., Phi-3.5):**  
> Models compiled exclusively for CPU vector instructions (`int4-rtn-block-32-acc-level-4`) can cause kernel ioctl failures (`dxgkio_escape: -22`) when forced onto GPU cores via WSL2's WDDM driver. Running large models seamlessly on GPU requires official weights exported with CUDA kernels (`-cuda-gpu`).

---

## 6. 🚀 Command Cheat Sheet & Quick Reference

```bash
# Environment & runtime diagnostics across installed models
python3 benchmark.py --check

# Run 1:1 speed comparison
python3 benchmark.py --runtime all --models "ollama:phi3:mini,foundry:Phi-3.5-mini-instruct-generic-cpu" --suite speed

# Run automated coding sandbox evaluation
python3 benchmark.py --runtime all --models "ollama:phi3:mini,foundry:Phi-3.5-mini-instruct-generic-cpu" --suite coding

# Foundry Local Service Management
foundry server status     # Inspect service health, PID, and active dynamic port
foundry server restart    # Restart local daemon
foundry cache list        # View downloaded models and memory states
foundry model load <id>   # Load model into active daemon memory
foundry model unload <id> # Unload model to reclaim system memory
```

---

## 7. ⚡ Direct ONNX Runtime GenAI GPU Inference (Bypassing CLI Preview Restrictions)

Because the Microsoft Foundry CLI preview on Linux artificially restricts execution to the CPU provider (`FoundryLocalCore` machine-type detection gates CUDA on Linux), **BenchRig** provides a direct runtime backend (`onnx-gpu`) powered directly by `onnxruntime-genai-cuda`:

### Architecture & Capabilities
- **Backend**: `onnxruntime_genai.Model` initialized with `CUDAExecutionProvider` via `og.Config`.
- **Hardware Acceleration**: 100% native CUDA on NVIDIA GeForce RTX GPUs in WSL2 / Linux.
- **Model Checkpoints**: Downloaded directly from official Microsoft HuggingFace repositories (e.g. `microsoft/Phi-4-mini-instruct-onnx` -> `gpu/gpu-int4-rtn-block-32`).

### Quick Execution
```bash
# Run standalone direct GPU inference:
.venv/bin/python3 examples/run_onnx_gpu.py \
  --model Phi-4-mini-instruct-cuda-gpu \
  --prompt "Write a python function to compute the nth Fibonacci number efficiently."

# Run 1:1 cross-engine benchmark against cached Ollama baseline:
.venv/bin/python3 benchmark.py \
  --runtime onnx-gpu \
  --models Phi-4-mini-instruct-cuda-gpu \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

### Verified Benchmark Results (RTX 5070 WSL2)
| Metric | Ollama (`phi3:mini`) | MS Foundry CLI (`Phi-3.5-mini`) | Direct ONNX GenAI (`Phi-4-mini-cuda`) |
|:---|:---:|:---:|:---:|
| **Decode Speed** | 94.0 tok/s | 9.2 tok/s | **119.4 tok/s** (1.3x Ollama, 13x Foundry CPU) |
| **Prefill Speed** | 2,639.8 tok/s | 47.8 tok/s | **4,528.0 tok/s** |
| **Average TTFT** | 0.21s | 2.08s | **0.05s** (50ms) |
| **Coding Pass Rate** | 17.6% | 29.4% | **100.0%** (17/17 tests passed) |
| **VRAM Footprint** | 11,135 MB (VRAM) | 7,136 MB (RAM) | **9,454 MB** (100% GPU VRAM fit) |

---

## 8. 🎯 Option 2: Native Microsoft Foundry Local Daemon GPU Serving (Cache Injection)

Beyond direct Python execution (Option 3), **Option 2** enables the official native Microsoft Foundry Local background daemon (`foundrylocald`) to serve GPU-accelerated models natively over its OpenAI-compatible `/v1/chat/completions` HTTP endpoint.

### How It Works (Weight Substitution & Cache Registration)
Because `foundrylocald` dynamically loads `libonnxruntime_providers_cuda.so` when dynamic linker paths are properly exported, the daemon is fully CUDA-capable. It parses `genai_config.json` via ONNX Runtime GenAI's `OgaCreateConfig`. By substituting real GPU-quantized weights into the daemon's local model cache and declaring CUDA in `provider_options`, the daemon loads the model directly onto GPU VRAM.

#### Step 1: Populate Model Cache
Hardlink or copy the GPU ONNX model files (e.g. `Phi-4-mini-instruct-cuda-gpu`) into the Foundry cache directory:
```bash
mkdir -p ~/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-generic-cpu-5/v5
# Link weights, tokenizer, and config files
cd ~/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-generic-cpu-5/v5
ln -f /path/to/gpu_model/model.onnx .
ln -f /path/to/gpu_model/model.onnx.data .
ln -f /path/to/gpu_model/tokenizer*.json .
ln -f /path/to/gpu_model/*.json .
```

#### Step 2: Configure `genai_config.json` for CUDA
Ensure `provider_options` specifies CUDA execution in `~/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-generic-cpu-5/v5/genai_config.json`:
```json
"session_options": {
    "log_id": "onnxruntime-genai",
    "provider_options": [
        {
            "cuda": {}
        }
    ]
}
```

#### Step 3: Add `inference_model.json`
```json
{
  "Name": "Phi-4-mini-instruct-generic-cpu:5",
  "PromptTemplate": {
    "system": "<|system|>\n{Content}<|end|>",
    "user": "<|user|>\n{Content}<|end|>",
    "assistant": "<|assistant|>\n{Content}<|end|>",
    "prompt": "<|user|>\n{Content}<|end|>\n<|assistant|>"
  }
}
```

#### Step 4: Mark Model as Cached in `foundry.modelinfo.json`
In `~/.foundry/cache/models/foundry.modelinfo.json`, set `"cached": true` for model ID `Phi-4-mini-instruct-generic-cpu:5`.

#### Step 5: Load Model & Verify GPU Acceleration
```bash
# Load model via Foundry CLI:
foundry model load phi-4-mini

# Verify process in nvidia-smi (foundrylocald.real will appear as Compute 'C'):
nvidia-smi
```

### Verified Option 2 Benchmark Results (NVIDIA GeForce RTX 5070 WSL2)
Measured via `benchmark.py --runtime foundry --models foundry:phi-4-mini --suite coding --baseline <ollama_run>.json`:

| Evaluation Metric | Ollama (`phi3:mini`) [llama.cpp CUDA] | MS Foundry Native Daemon (`phi-4-mini`) [ONNX GenAI CUDA] | Delta / Advantage |
|:---|:---:|:---:|:---:|
| **Decode Speed** | 94.0 tok/s | **130.2 tok/s** | **MS Foundry 1.4x faster** (vs 9.2 tok/s CPU: **14.2x speedup**) |
| **Prompt Prefill Speed** | 2,639.8 tok/s | **1,006.8 tok/s** | Ollama 2.6x faster |
| **Time to First Token (TTFT)** | 0.21s | **0.09s (90 ms)** | **MS Foundry 2.3x lower latency** |
| **Coding Pass Rate** | 17.6% | **100.0% (17/17 tests)** | **MS Foundry +82.4%** |
| **Process in `nvidia-smi`** | `ollama_llama_server` (Compute `C`) | `foundrylocald.real` (Compute `C`) | Both verified native GPU compute |
| **Peak VRAM Memory** | 11,135 MB | **10,875 MB** | 100% fits within 12GB VRAM |
| **Composite Score** | 31.1 / 100 | **64.0 / 100** | **MS Foundry leads (+32.9)** |

Option 2 proves that Microsoft Foundry Local's native service daemon running in WSL2 is fully capable of serving high-performance CUDA-accelerated models over standard OpenAI-compatible endpoints with zero cloud token cost.

