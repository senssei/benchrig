# 🚀 Tutorial 1: Quickstart Guide (Zero to Benchmark in 5 Minutes)

This tutorial walks you through setting up **BenchRig**, validating your platform hardware acceleration (macOS Apple Silicon Metal or Linux/WSL2 NVIDIA CUDA), and executing your first local LLM benchmark.

---

## 📋 Prerequisites

Before beginning, ensure you have:
1. **Python 3.10+** installed:
   ```bash
   python3 --version
   ```
2. **Ollama** installed and running:
   - macOS: Install via [ollama.com](https://ollama.com/download) or `brew install ollama`.
   - Linux / WSL2: Install via `curl -fsSL https://ollama.com/install.sh | sh`.
   - Verify daemon status:
     ```bash
     curl -s http://localhost:11434/api/tags
     ```
3. *(Optional)* **Microsoft Foundry Local** installed:
   - Verify CLI: `foundry --version`
   - Verify server daemon: `foundry server status`

---

## 🛠 Step 1: Environment Setup

Clone the repository and install dependencies in an isolated virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/senssei/benchrig.git
cd benchrig

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -e .
```

---

## 🔍 Step 2: Run Hardware & Runtime Diagnostics

Execute the non-destructive diagnostic check to verify accelerator detection and daemon reachability:

```bash
benchrig --check
```

### Expected Output:
- **macOS**: Displays Apple Silicon GPU core count (e.g. M2/M3/M4 Max), Unified Memory (UMA) size, Metal driver version, and Ollama Metal VRAM buffer allocations.
- **Linux / WSL2**: Displays NVIDIA GeForce RTX model, dedicated VRAM capacity, driver version, CUDA UMD version, and system host RAM.
- **Runtimes**: Validates whether Ollama (`http://localhost:11434`) and Microsoft Foundry Local (`http://127.0.0.1:<port>`) are responding.

---

## 📥 Step 3: Pull Recommended Models

Pull the standard benchmark models configured in `config.yaml`:

```bash
# Pull standard Ollama coding & reasoning models:
benchrig --runtime ollama --pull-recommended
```

Or manually pull your preferred model:
```bash
ollama pull qwen2.5-coder:7b
ollama pull phi3:mini
```

---

## ⚡ Step 4: Run Your First Benchmark

### 1. Fast Generation Speed & Latency Test
Evaluate raw token throughput (tokens/s) and Time to First Token (TTFT):

```bash
benchrig --runtime ollama --models qwen2.5-coder:7b --suite speed
```

### 2. Sandboxed Automated Coding Precision Test
Evaluate real-world code generation precision against deterministic unit test assertions in isolated subshells:

```bash
benchrig --runtime ollama --models qwen2.5-coder:7b --suite coding
```

During execution, BenchRig:
1. Prompts the model with algorithmic challenges (e.g., *Nested Dict Flattening*, *LRU Cache*, *Valid Parentheses*).
2. Extracts clean Python code blocks.
3. Spawns an isolated subprocess to execute the code against rigorous unit tests.
4. Reports pass/fail status, execution time, and memory consumption.

---

## 📊 Step 5: Inspect Results & Reports

Upon completion, BenchRig generates:
1. **Interactive Rich Terminal Leaderboard**: Shows rankings, composite scores, decode speeds, TTFT, and cloud API cost savings.
2. **`results/LATEST_SUMMARY.md`**: Human-readable Markdown summary with assertion breakdowns.
3. **`results/runs/benchmark_<timestamp>.json`**: Machine-readable telemetry and scores.

To display the latest run results in the terminal at any time:
```bash
benchrig --compare results/latest.json
```

---

## 🧭 Other Runtimes

The same commands work on other runtimes: `--runtime foundry`, `--runtime onnx-gpu` and `--runtime prism` (a
[Prism](https://github.com/senssei/prism-local) server started with `prism serve`). `benchrig --check` shows which are reachable;
see [Runtimes](../runtimes.md) for setup and model prefixes.

---

## 🎯 Next Steps
- Learn how to enable GPU acceleration in Microsoft Foundry Local: [Tutorial 2: Foundry GPU Setup](foundry-gpu-setup.md).
- Run side-by-side cross-engine comparisons: [Tutorial 3: Cross-Engine Benchmarking](cross-engine-benchmarking.md).
- Offload agent coding routines with zero token cost: see [local-coders](https://github.com/senssei/local-coders).
