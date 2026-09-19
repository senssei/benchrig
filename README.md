<div align="center">

# ⚡ BenchRig

**Cross-platform evaluation, hardware profiling, and benchmarking rig for local LLMs on Ollama, Microsoft Foundry Local and ONNX Runtime GenAI.**  
Tailored for **macOS Apple Silicon (M1/M2/M3/M4 Metal & Unified Memory)** and **Linux / WSL2 (NVIDIA GeForce RTX CUDA)**.

[![PyPI](https://img.shields.io/pypi/v/benchrig.svg)](https://pypi.org/project/benchrig/)
[![CI](https://github.com/senssei/benchrig/actions/workflows/ci.yml/badge.svg)](https://github.com/senssei/benchrig/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-REST%20API-black?logo=ollama)](https://ollama.com)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20NVIDIA%20CUDA-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

## 📌 Overview

**BenchRig** is an automated benchmarking and profiling suite that measures real-world code generation precision, logical reasoning, prompt prefill / generation speeds, context saturation, and hardware telemetry across local language models served via [Ollama](https://ollama.com) (`llama.cpp`) and **Microsoft Foundry Server Runtime** (**Foundry Local** / ONNX Runtime GenAI).

Unlike generic perplexity benchmarks, this suite focuses on **practical developer workloads**:
- **Executes code in isolated sandboxes** and checks deterministic unit test assertions.
- **Direct Cross-Runtime & Engine Comparison**: Benchmarks `llama.cpp` against Microsoft's ONNX Runtime GenAI side-by-side on identical hardware.
- **Analyzes reasoning models** (e.g. DeepSeek-R1) by inspecting `<think>` token patterns and extracting final answers.
- **Measures true Time to First Token (TTFT)** via high-precision streaming probes.
- **Monitors hardware saturation in real-time** (Metal buffer memory & GPU utilization on Apple Silicon; VRAM, power draw, and temperatures on NVIDIA).

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph CLI ["Benchmark CLI (benchrig/cli.py)"]
        A["--check / --runtime (ollama|foundry|all)\n--models / --suite / --runs"] --> B[BenchmarkRunner]
    end

    subgraph Hardware ["Cross-Platform Hardware Abstraction (benchrig/core/hardware.py)"]
        B -->|Initialize| HW[HardwareProvider Factory]
        HW -->|macOS Darwin| M1["DarwinAppleSiliconProvider\n- sysctl UMA RAM\n- vm_stat memory\n- ioreg GPU load\n- Ollama /api/ps Metal VRAM"]
        HW -->|Linux / WSL2| NV["LinuxNvidiaProvider\n- nvidia-smi VRAM\n- /proc/meminfo\n- GPU Power & Temp"]
    end

    subgraph Runtimes ["Unified Runtime Clients (benchrig/core/client.py)"]
        B --> BaseClient["BaseRuntimeClient (base class)"]
        BaseClient --> Ollama["OllamaClient (llama.cpp)\n- http://localhost:11434"]
        BaseClient --> Foundry["FoundryClient (ONNX Runtime GenAI)\n- http://localhost:5272/v1"]
    end

    subgraph Execution ["Test Execution Engine (benchrig/core/runner.py)"]
        Runtimes --> S1["Speed Suite\n(Decode & Prefill TTFT)"]
        Runtimes --> S2["Coding Suite\n(Isolated Sandbox Runner)"]
        Runtimes --> S3["Reasoning Suite\n(<think> Parser & Verifier)"]
        Runtimes --> S4["Polish NLP Suite\n(Declension & Grammar)"]
        Runtimes --> S5["Context Scaling\n(512 to 8192 tokens)"]
        
        S2 --> Sandbox["Sandboxed Python Subprocess\ncore/sandbox.py"]
        S3 --> ReasonParser["Reasoning Answer Extractor\ncore/reasoning_parser.py"]
    end

    subgraph Reporting ["Reporting & Output (benchrig/reporting/)"]
        B --> Leaderboard["Rich Terminal Leaderboard\n(Dedicated 'Runtime' Column)"]
        B --> MarkdownReport["Markdown Report Generator\n(Cross-Engine Delta Section)"]
        B --> JSONHistory["JSON History Dumps\nresults/runs/*.json"]
    end
```

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🚀 **Multi-Runtime Engine Support** | Evaluate models across **Ollama** (`llama.cpp`) and **Microsoft Foundry Local** (ONNX Runtime GenAI) with unified CLI and scoring. |
| ⏱ **High-Precision Timing** | Measures generation tokens/sec, prefill tokens/sec, and Time to First Token (TTFT) via nanosecond-precision streaming. |
| 💻 **Automated Sandboxed Coding** | Automatically extracts code blocks from LLM responses, wraps them with test harnesses, and executes them in isolated subprocesses against test assertions. |
| 🧠 **Reasoning & `<think>` Parser** | Detects whether models generate chain-of-thought blocks (`<think>...</think>`), calculates thinking token volume, and extracts final answers. |
| 📐 **Context Scaling (512 - 8k)** | Progressively loads larger contexts (512, 1024, 2048, 4096, 8192 tokens) to assess TTFT degradation and memory growth. |
| 📊 **Hardware Telemetry** | Samples GPU/UMA memory usage, GPU utilization %, temperatures, and power draw during execution without requiring root on macOS. |
| 🏆 **Leaderboard & Markdown Reports** | Produces Rich terminal tables with category medals (🥇, 🥈, 🥉), runtime indicators, and comparative Markdown summaries in `results/LATEST_SUMMARY.md`. |

---

## 📚 Tutorials & Documentation

Comprehensive step-by-step tutorials and engineering deep dives are available in [`docs/`](docs/README.md):

### 🚀 Step-by-Step Hands-On Tutorials:
1. [**Tutorial 1: Quickstart Guide**](docs/tutorials/01_QUICKSTART_GUIDE.md) – Zero to benchmark in 5 minutes across macOS and Linux/WSL2.
2. [**Tutorial 2: Foundry GPU Setup (WSL2 / Linux)**](docs/tutorials/02_FOUNDRY_GPU_SETUP.md) – Complete guide for NVIDIA GPU acceleration on Microsoft Foundry Local (Cache Injection & Direct ONNX GenAI).
3. [**Tutorial 3: Fair 1:1 Cross-Engine Benchmarking**](docs/tutorials/03_CROSS_ENGINE_BENCHMARKING.md) – Standardizing parameters, cached baseline evaluation (`--baseline`), and scorecards.
4. [**Tutorial 4: Authoring Custom Benchmark Scenarios**](docs/tutorials/04_CUSTOM_SCENARIO_AUTHORING.md) – Designing deterministic coding challenges, reasoning puzzles, and test harnesses.

### 📖 Technical Documentation Guides:
- [**System Architecture & Design**](docs/ARCHITECTURE.md) | [**CLI Reference & Options**](docs/CLI_USAGE.md) | [**Configuration Reference**](docs/CONFIGURATION.md)
- [**Foundry Local, CUDA & TensorRT Guide**](docs/FOUNDRY_WSL_CUDA_TENSORRT_GUIDE.md) | [**Hardware Telemetry**](docs/HARDWARE_TELEMETRY.md)
- [**Benchmark Suites**](docs/BENCHMARK_SUITES.md) | [**Developer & Contributing Guide**](docs/DEVELOPER_GUIDE.md)

---

## 🚀 Quick Start

### Install from PyPI

```bash
pipx install benchrig                         # or: pip install benchrig
pip install "benchrig[onnx-gpu]"       # + direct ONNX Runtime GenAI (CUDA) engine
pip install "benchrig[charts]"         # + matplotlib charts

benchrig --version
benchrig --check                              # diagnose runtimes and accelerators
benchrig --models qwen2.5-coder:7b --suite coding
```

Defaults (`config.yaml` and the scenario suites) are bundled in the package. Put a `./config.yaml` or `./scenarios/` in your
working directory (or pass `--config` / `--scenarios-dir`, or set `BENCHRIG_CONFIG`) to override them. Direct ONNX models are
looked up in `$BENCHRIG_MODEL_DIRS`, then `./models`, then `~/.benchrig/models`.

### From source

### Option A: macOS (Apple Silicon M1 / M2 / M3 / M4)

We provide an automated setup script that verifies your Apple Silicon chip, checks Python, creates a virtual environment, installs dependencies, and tests your Ollama connection:

```bash
git clone https://github.com/senssei/benchrig.git
cd benchrig

# Run the automated setup script:
chmod +x setup_mac.sh
./setup_mac.sh
```

### Option B: Linux / WSL2 (NVIDIA GeForce RTX)

```bash
git clone https://github.com/senssei/benchrig.git
cd benchrig

# Create virtual environment & install in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run environment diagnostic check
benchrig --check
```

---

## 💻 CLI Usage & Examples

### 1. Diagnostic Environment Check
Verifies Ollama & MS Foundry Server connectivity, lists installed models across runtimes, and displays detected hardware:
```bash
benchrig --check
```

### 2. Multi-Runtime & Cross-Engine Comparison
Compare models running under **Ollama** (`llama.cpp`) and **MS Foundry** (ONNX Runtime GenAI) head-to-head on the same hardware:
```bash
# Compare all installed models across both runtimes:
benchrig --runtime all --models installed

# Compare specific models across engines:
benchrig --models ollama:qwen2.5-coder:7b,foundry:phi-4

# Benchmark exclusively on Microsoft Foundry Server:
benchrig --runtime foundry --models phi-4,qwen2.5-coder-7b
```

### 3. Benchmark Specific Models
```bash
benchrig --models qwen2.5-coder:7b,llama3.1:8b
```

### 4. Run a Specific Test Suite
Available suites: `speed`, `coding`, `reasoning`, `polish`, `context`, `all`.
```bash
# Test only coding with automated unit tests (2 runs per model):
benchrig --models qwen2.5-coder:7b --suite coding --runs 2

# Test context scaling up to 8k tokens:
benchrig --models llama3.1:8b --suite context
```

### 5. Pull / Acquire Recommended Models
Downloads missing standard models from the respective Ollama or MS Foundry catalog:
```bash
benchrig --pull-recommended
# Or specify runtime:
benchrig --runtime foundry --pull-recommended
```

---

## ⚙️ Configuration (`config.yaml`)

Edit [config.yaml](benchrig/data/config.yaml) to customize execution parameters, endpoints, and scoring weights:

```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout_sec: 180
  warmup: true             # Pre-warms weights into memory before timing
  unload_after_test: true  # Keeps memory clean between model runs
  default_num_ctx: 4096

foundry:
  base_url: "http://localhost:5272/v1"  # Microsoft Foundry Local REST endpoint
  auto_detect_port: true                # Automatically queries active port via CLI
  cli_path: "foundry"                   # Optional Foundry Local CLI executable
  timeout_sec: 180
  warmup: true
  unload_after_test: true

benchmark:
  default_runs: 1
  default_runtime: "ollama"  # "ollama", "foundry", or "all"
  composite_weights:
    coding: 0.40           # 40% automated unit tests pass rate
    reasoning: 0.30        # 30% reasoning & logic ground truth
    performance: 0.30      # 30% normalized decode speed & memory efficiency

hardware:
  sample_interval_sec: 0.15
  # "auto" computes 88% of detected VRAM / Unified Memory
  vram_warning_threshold_mb: "auto"
```

---

## 🤖 Agent Skills & MCP Servers

The `ollama-coder` / `foundry-coder` agent skills and the stdio MCP servers that offload coding tasks to local models now live in
their own repository: [**senssei/local-coders**](https://github.com/senssei/local-coders).

---

## 🧪 Customizing Scenarios

Test cases are stored as clean JSON files inside the [scenarios/](benchrig/data/scenarios/) directory:

- [scenarios/coding.json](benchrig/data/scenarios/coding.json): Python coding tasks paired with test assertion arrays evaluated in sandbox subprocesses.
- [scenarios/reasoning.json](benchrig/data/scenarios/reasoning.json): Multi-step math and logic puzzles with expected ground truth strings and regex patterns.
- [scenarios/speed.json](benchrig/data/scenarios/speed.json): Raw generation and prefill throughput prompts.
- [scenarios/context_scaling.json](benchrig/data/scenarios/context_scaling.json): Context scaling tests up to 8k tokens.
- [scenarios/polish.json](benchrig/data/scenarios/polish.json): Multilingual tests checking Polish language morphology and syntax.

## 📚 Documentation

Detailed architecture, configuration guides, benchmark specifications, and operational manuals are available in the [`docs/`](docs/) directory:

| Document | Description |
| :--- | :--- |
| 🏛 [**System Architecture**](docs/ARCHITECTURE.md) | Deep dive into the Hardware Abstraction Layer (HAL), Runtime Abstraction Layer (RAL), sandbox isolation, and reporting pipeline. |
| 🚀 [**WSL2 CUDA & TensorRT Guide**](docs/FOUNDRY_WSL_CUDA_TENSORRT_GUIDE.md) | Complete guide to configuring Microsoft Foundry Local with NVIDIA CUDA and TensorRT acceleration on WSL2. |
| ⚙️ [**Configuration Reference**](docs/CONFIGURATION.md) | Full reference for `config.yaml`, environment variables (`OLLAMA_HOST`, `FOUNDRY_BASE_URL`), and dynamic port discovery. |
| 🧪 [**Benchmark Suites Mechanics**](docs/BENCHMARK_SUITES.md) | Evaluation methodology for Speed, Coding, Reasoning, Polish NLP, and Context Scaling suites. |
| 💻 [**CLI Usage & Recipes**](docs/CLI_USAGE.md) | Command-line parameters, scenario filtering, cross-engine flags, and automation scripts. |
| 📊 [**Hardware Telemetry & Profiling**](docs/HARDWARE_TELEMETRY.md) | Real-time GPU VRAM, compute load, Apple Silicon UMA memory, power draw, and temperature sampling. |
| 📝 [**Scenario Authoring Guide**](docs/SCENARIOS_GUIDE.md) | Schema reference and instructions for creating custom coding, reasoning, and context scaling scenarios. |
| 👩‍💻 [**Developer & Contributing Guide**](docs/DEVELOPER_GUIDE.md) | Guide for adding runtime clients (`BaseRuntimeClient`), running test suites, and adhering to sandbox security. |

---

## 📁 Repository Structure

```
benchrig/
├── benchrig/                 # The installable package (import `benchrig`, command `benchrig`)
│   ├── cli.py                # CLI entrypoint
│   ├── data/config.yaml      # Bundled default configuration (endpoints, weights, thresholds)
│   ├── data/scenarios/       # Bundled test scenario definitions (JSON)
│   ├── core/                 # Runtime clients, hardware providers, runner, sandbox, reasoning parser
│   └── reporting/            # Rich terminal UI and Markdown report generator
├── pyproject.toml            # Packaging metadata, ruff and pytest configuration
├── CHANGELOG.md  CONTRIBUTING.md  SECURITY.md
├── setup_mac.sh              # Quickstart installer for macOS Apple Silicon
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── AGENTS.md                 # Guidelines for AI agents working on this repo
├── .github/workflows/ci.yml  # CI: ruff lint/format check + pytest (Python 3.10-3.13, wheel smoke test)
├── docs/                     # Comprehensive documentation guides (guides + 4 tutorials)
│   ├── README.md             # Documentation & tutorials index
│   ├── ARCHITECTURE.md
│   ├── BENCHMARK_SUITES.md
│   ├── CLI_USAGE.md
│   ├── CONFIGURATION.md
│   ├── DEVELOPER_GUIDE.md
│   ├── FOUNDRY_WSL_CUDA_TENSORRT_GUIDE.md
│   ├── HARDWARE_TELEMETRY.md
│   ├── SCENARIOS_GUIDE.md
│   └── tutorials/            # Hands-on step-by-step tutorials
│       ├── 01_QUICKSTART_GUIDE.md
│       ├── 02_FOUNDRY_GPU_SETUP.md
│       ├── 03_CROSS_ENGINE_BENCHMARKING.md
│       └── 04_CUSTOM_SCENARIO_AUTHORING.md
├── examples/
│   ├── calculator.py         # Sample module for test generation benchmarks
│   └── run_onnx_gpu.py       # Standalone direct ONNX GenAI CUDA runner
├── results/
│   ├── 1TO1_COMPARISON_REPORT.md # Cross-engine comparative report
│   ├── LATEST_SUMMARY.md     # Latest benchmark Markdown report
│   ├── latest.json           # Latest scorecard JSON
│   └── runs/                 # Historical benchmark runs
└── tests/                    # Offline unit test suite (no network or GPU required)
    ├── test_benchmark_cli.py
    ├── test_foundry_runtime.py
    ├── test_hardware.py
    ├── test_onnx_client.py
    ├── test_packaging.py
    ├── test_runner_suites.py
    ├── test_sandbox.py
    └── test_token_savings.py
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
