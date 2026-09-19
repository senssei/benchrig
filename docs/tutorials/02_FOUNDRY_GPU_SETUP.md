# ⚡ Tutorial 2: Unlocking NVIDIA GPU Acceleration on Microsoft Foundry Local (WSL2 / Linux)

This tutorial provides a complete walkthrough for enabling full NVIDIA GPU acceleration (CUDA & ONNX Runtime GenAI) on **Microsoft Foundry Local** in a **WSL2 (Ubuntu 24.04)** or **Linux** environment.

---

## 🎯 Background & The Challenge

By default, the Microsoft Foundry CLI (`foundry`) on Linux selects CPU variants (`-generic-cpu`) because machine-type detection defaults to CPU execution on non-Windows platforms.

However, the underlying Foundry Local core engine is powered by **ONNX Runtime** (`libonnxruntime.so`) and **ONNX Runtime GenAI** (`libonnxruntime-genai.so`), which ship with full CUDA execution provider modules:
- `~/.local/lib/foundry-cli/libonnxruntime_providers_cuda.so`
- `~/.local/lib/foundry-cli/libonnxruntime-genai-cuda.so`

To run models on an NVIDIA GPU (e.g. GeForce RTX 30/40/50 series) with zero cloud token cost, BenchRig supports two verified pathways:
1. **Option 2 (Recommended for Microservices / Agents)**: Native Daemon Weight Cache Injection. The official `foundrylocald` daemon loads GPU ONNX weights and serves them over its OpenAI-compatible `/v1` endpoint.
2. **Option 3 (Direct Script Execution)**: Direct Python ONNX Runtime GenAI C-API execution via `OnnxGenAiClient`.

---

## 🛠 Prerequisites: Dynamic Linker & CUDA Setup

Before Foundry Local can register `CUDAExecutionProvider`, the dynamic linker must locate CUDA 12 and cuDNN 9 libraries without requiring OS-level root installations.

### Step 1: Install CUDA & cuDNN in User Virtual Environment
```bash
# Activate workspace virtual environment
source .venv/bin/activate

# Install NVIDIA CUDA and cuDNN wheels:
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12
```

### Step 2: Create Compatibility Symlinks
Some libraries compiled against CUDA 12 look for specific major/minor sonames. Ensure matching links exist:
```bash
# If libcublasLt.so.12 is missing or libcufft.so.11 is required:
cd .venv/lib/python3.12/site-packages/nvidia/cufft/lib/
ln -sf libcufft.so.12 libcufft.so.11
```

### Step 3: Install the Daemon Wrapper Script
Create or update `~/.local/lib/foundry-cli/foundrylocald` to ensure dynamic library paths are exported before launching `foundrylocald.real`:

```bash
# Locate original daemon binary
DAEMON_DIR="$HOME/.local/lib/foundry-cli"
if [ ! -f "$DAEMON_DIR/foundrylocald.real" ]; then
    mv "$DAEMON_DIR/foundrylocald" "$DAEMON_DIR/foundrylocald.real"
fi

# Create wrapper script:
cat << 'EOF' > "$DAEMON_DIR/foundrylocald"
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Workspace venv libraries path:
VENV_NVIDIA="$(cd "$SCRIPT_DIR/../../.." && pwd)/02-ollama-loadtest/.venv/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH="$SCRIPT_DIR:$VENV_NVIDIA/cublas/lib:$VENV_NVIDIA/cudnn/lib:$VENV_NVIDIA/cufft/lib:$VENV_NVIDIA/cuda_runtime/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"

exec "$SCRIPT_DIR/foundrylocald.real" "$@"
EOF

chmod +x "$DAEMON_DIR/foundrylocald"
```

### Step 4: Verify CUDA Registration in Daemon Logs
Restart the daemon and inspect logs:
```bash
foundry server restart
tail -n 25 ~/.foundry/logs/foundry.core$(date +%Y%m%d).log
```
Look for the confirmation line:
```text
[INF] Successfully registered CUDA execution provider from app package. EP=FoundryLocalCUDA, Library=libonnxruntime_providers_cuda.so
```

---

## 🚀 Option 2: Native Daemon Weight Cache Injection (OpenAI `/v1` Endpoint)

Option 2 allows the official `foundrylocald` daemon to serve genuine CUDA GPU weights directly through its OpenAI-compatible `/v1/chat/completions` API.

### 1. Download Genuine GPU-Quantized ONNX Weights
Download the official Microsoft Phi-4 Mini GPU model (INT4 AWQ):
```bash
mkdir -p models/Phi-4-mini-instruct-cuda-gpu
huggingface-cli download microsoft/Phi-4-mini-instruct-onnx \
  --include "gpu/gpu-int4-rtn-block-32/*" \
  --local-dir models/Phi-4-mini-instruct-cuda-gpu
```

### 2. Populate Foundry Cache Directory
Hardlink the model files into the Foundry cache directory structure:
```bash
CACHE_DIR="$HOME/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-generic-cpu-5/v5"
mkdir -p "$CACHE_DIR"

# Hardlink weights and tokenizers:
cd "$CACHE_DIR"
for f in models/Phi-4-mini-instruct-cuda-gpu/*; do
    base=$(basename "$f")
    if [ "$base" != "genai_config.json" ]; then
        ln -f "$f" "$CACHE_DIR/$base"
    fi
done
```

### 3. Configure `genai_config.json` for CUDA
Write `genai_config.json` in `$CACHE_DIR` with `provider_options`:
```json
{
    "model": {
        "bos_token_id": 199999,
        "context_length": 131072,
        "decoder": {
            "session_options": {
                "log_id": "onnxruntime-genai",
                "provider_options": [
                    {
                        "cuda": {}
                    }
                ]
            },
            "filename": "model.onnx"
        },
        "eos_token_id": [200020, 199999],
        "pad_token_id": 199999,
        "type": "phi3",
        "vocab_size": 200064
    }
}
```

### 4. Create `inference_model.json`
Write the prompt template into `$CACHE_DIR/inference_model.json`:
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

### 5. Mark Model as Cached in `foundry.modelinfo.json`
Update `~/.foundry/cache/models/foundry.modelinfo.json` so Foundry recognizes the model without downloading:
```python
import json, os

info_path = os.path.expanduser("~/.foundry/cache/models/foundry.modelinfo.json")
with open(info_path, "r") as f:
    data = json.load(f)

for m in data.get("models", []):
    if m.get("id") == "Phi-4-mini-instruct-generic-cpu:5":
        m["cached"] = True

with open(info_path, "w") as f:
    json.dump(data, f, indent=2)
```

### 6. Load Model & Verify GPU Acceleration
```bash
# Load model into memory via Foundry CLI:
foundry model load phi-4-mini

# Verify process in nvidia-smi:
nvidia-smi
```
You will see `/foundrylocald.real` registered as a Compute process (`C`):
```text
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          104268      C   /foundrylocald.real                   N/A      |
```

### 7. Test Inference via OpenAI API
```bash
curl -X POST http://127.0.0.1:39139/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-4-mini",
    "messages": [
      {"role": "user", "content": "Write a python function to compute primes."}
    ],
    "max_tokens": 100
  }'
```
**Observed Performance**: **130.2 tokens/s**, **0.09s TTFT** (vs 9.2 tok/s on CPU).

---

## ⚡ Option 3: Direct ONNX Runtime GenAI GPU Inference

If you want direct Python-level model control bypassing the Foundry background daemon completely:

### Standalone Runner
```bash
# Run direct GPU inference:
python3 examples/run_onnx_gpu.py \
  --model Phi-4-mini-instruct-cuda-gpu \
  --prompt "Write an LRU cache implementation in Python."
```

### Programmatic Python Usage
```python
from benchrig.core.onnx_client import OnnxGenAiClient

client = OnnxGenAiClient()
print("CUDA Available:", client.is_cuda_available())

client.load_model("Phi-4-mini-instruct-cuda-gpu")
response = client.generate(
    model="Phi-4-mini-instruct-cuda-gpu", prompt="Explain quicksort in 2 sentences.", options={"max_tokens": 80}
)
print("Response:", response["response"])
print(f"Speed: {response['eval_tok_per_sec']} tok/s | TTFT: {response['ttft_sec']}s")
```

---

## 🔍 Troubleshooting Checklist

| Symptom | Root Cause | Fix |
| :--- | :--- | :--- |
| `libcublasLt.so.12: cannot open shared object file` | Dynamic linker cannot find PyPI CUDA wheels. | Ensure `foundrylocald` wrapper exports `LD_LIBRARY_PATH` pointing to `.venv/lib/.../nvidia/cublas/lib`. |
| `dxgkio_escape: -22` | CPU-compiled weights (`int4-rtn-block-32-acc-level-4`) were forced onto GPU cores. | Substitute real CUDA GPU weights (`Phi-4-mini-instruct-cuda-gpu`). |
| `Model '...' is not loaded` | Model files exist on disk but are not placed in memory. | Run `foundry model load <model>` or rely on `FoundryClient.generate()` auto-loading. |
| `libcufft.so.11: cannot open shared object file` | Linker looks for CUDA 11 soname. | Symlink `libcufft.so.12` to `libcufft.so.11` in the virtualenv. |
