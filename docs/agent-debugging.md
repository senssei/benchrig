# 🛠 Agent Debugging & Troubleshooting Playbook

This document outlines the diagnostic routine for AI coding agents and human operators when troubleshooting local daemons, runtime execution providers, linkers, hardware telemetry, or connection issues in **BenchRig**.

---

## 1. Diagnosing Daemon Status & Connectivity

### A. Ollama Daemon
```bash
# Check HTTP ping and installed models:
curl -s http://localhost:11434/api/tags

# If offline or unresponsive:
# Linux/WSL2:
ollama serve > /dev/null 2>&1 &
# Or check systemd service:
systemctl --user status ollama || sudo systemctl status ollama
```

### B. Microsoft Foundry Local Daemon
```bash
# 1. Quick diagnostic via benchrig:
benchrig --check

# 2. Native CLI status check:
foundry server status

# 3. If offline, start the server:
foundry server start

# 4. Check dynamic port auto-discovery:
cat ~/.foundry/daemon.json
# Look for "web_urls": ["http://127.0.0.1:<port>"]
# Ensure your request targets http://127.0.0.1:<port>/v1
```

---

## 2. Inspecting Log Files
When troubleshooting Foundry Local or ONNX Runtime errors:
* **Daemon Startup & IPC Logs**:
  ```bash
  tail -n 50 ~/.foundry/logs/foundrylocald-$(date +%Y-%m-%d).log
  ```
* **Core Runtime & Execution Provider Logs**:
  ```bash
  tail -n 50 ~/.foundry/logs/foundry.core$(date +%Y%m%d).log
  ```
* **Filter for EP Registration**:
  ```bash
  grep -E "CUDA|TensorRT|EP|ExecutionProvider" ~/.foundry/logs/foundry.core$(date +%Y%m%d).log | tail -n 20
  ```

---

## 3. Hardware & Dynamic Linker (`LD_LIBRARY_PATH`) Verification

### On Linux / WSL2 (NVIDIA CUDA & TensorRT):
* **Verify GPU visibility**:
  ```bash
  nvidia-smi || /usr/lib/wsl/lib/nvidia-smi
  ```
* **Verify CUDA Execution Provider shared objects**:
  ```bash
  ldd ~/.local/lib/foundry-cli/libonnxruntime_providers_cuda.so | grep "not found"
  ```
  *(Should produce no output; all symbols resolved).*
* **Verify TensorRT Execution Provider shared objects**:
  ```bash
  export LD_LIBRARY_PATH="$HOME/.local/lib/tensorrt/tensorrt_libs:$LD_LIBRARY_PATH"
  ldd ~/.local/lib/foundry-cli/libonnxruntime_providers_tensorrt.so | grep "not found"
  ```
* **Verify Daemon Linker Wrapper**:
  Check `~/.local/lib/foundry-cli/foundrylocald`. It must prepend `LD_LIBRARY_PATH` before delegating to `foundrylocald.real`.

### On macOS (Apple Silicon Metal & UMA):
* **Verify physical RAM**: `sysctl -n hw.memsize`
* **Verify active UMA memory breakdown**: `vm_stat`
* **Verify GPU compute occupancy**: `ioreg -r -d 1 -w 0 -c IOAccelerator`
* **Verify Ollama Metal allocations**: `curl -s http://localhost:11434/api/ps`

---

## 4. Debugging Model Loading Errors
In Microsoft Foundry Local, models must be loaded into memory before `/v1/chat/completions` responds:
* **Error**: `Failed to handle OpenAI completion: Model '...' is not loaded.`
* **Fix**: Run `foundry model load <model_alias>` (e.g. `foundry model load phi-3.5-mini`).
* **Note**: `benchrig` catches this error and triggers auto-loading (`FoundryClient.generate`).
