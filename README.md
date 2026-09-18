<div align="center">

# ⚡ Ollama BenchRig

**Cross-platform evaluation, hardware profiling, and benchmarking rig for local LLMs running on Ollama.**  
Tailored for **macOS Apple Silicon (M1/M2/M3/M4 Metal & Unified Memory)** and **Linux / WSL2 (NVIDIA GeForce RTX CUDA)**.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-REST%20API-black?logo=ollama)](https://ollama.com)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%7C%20NVIDIA%20CUDA-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

</div>

---

## 📌 Overview

**Ollama BenchRig** is an automated benchmarking and profiling suite that measures real-world code generation precision, logical reasoning, prompt prefill / generation speeds, context saturation, and hardware telemetry across local language models served via [Ollama](https://ollama.com).

Unlike generic perplexity benchmarks, this suite focuses on **practical developer workloads**:
- **Executes code in isolated sandboxes** and checks deterministic unit test assertions.
- **Analyzes reasoning models** (e.g. DeepSeek-R1) by inspecting `<think>` token patterns and extracting final answers.
- **Measures true Time to First Token (TTFT)** via high-precision streaming probes.
- **Monitors hardware saturation in real-time** (Metal buffer memory & GPU utilization on Apple Silicon; VRAM, power draw, and temperatures on NVIDIA).
- **Includes a zero-dependency Model Context Protocol (MCP) server** for instant agentic pairing in Antigravity, Cursor, or Claude Desktop.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph CLI ["Benchmark CLI (benchmark.py)"]
        A[--check / --models / --suite / --runs] --> B[BenchmarkRunner]
    end

    subgraph Hardware ["Cross-Platform Hardware Abstraction (core/hardware.py)"]
        B -->|Initialize| HW[HardwareProvider Factory]
        HW -->|macOS Darwin| M1[DarwinAppleSiliconProvider\n- sysctl UMA RAM\n- vm_stat memory\n- ioreg GPU load\n- Ollama /api/ps Metal VRAM]
        HW -->|Linux / WSL2| NV[LinuxNvidiaProvider\n- nvidia-smi VRAM\n- /proc/meminfo\n- GPU Power & Temp]
    end

    subgraph Execution ["Test Execution Engine (core/runner.py)"]
        B --> Client[Ollama REST Client\ncore/client.py]
        Client --> S1["Speed Suite\n(Decode & Prefill TTFT)"]
        Client --> S2["Coding Suite\n(Isolated Sandbox Runner)"]
        Client --> S3["Reasoning Suite\n(<think> Parser & Verifier)"]
        Client --> S4["Polish NLP Suite\n(Declension & Grammar)"]
        Client --> S5["Context Scaling\n(512 to 8192 tokens)"]
        
        S2 --> Sandbox[Sandboxed Python Subprocess\ncore/sandbox.py]
        S3 --> ReasonParser[Reasoning Answer Extractor\ncore/reasoning_parser.py]
    end

    subgraph Reporting ["Reporting & Output (reporting/)"]
        B --> Leaderboard[Rich Terminal Leaderboard]
        B --> MarkdownReport[Markdown Report Generator\nresults/LATEST_SUMMARY.md]
        B --> JSONHistory[JSON History Dumps\nresults/runs/*.json]
    end
```

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🚀 **High-Precision Timing** | Measures generation tokens/sec, prefill tokens/sec, and Time to First Token (TTFT) via nanosecond-precision streaming. |
| 💻 **Automated Sandboxed Coding** | Automatically extracts code blocks from LLM responses, wraps them with test harnesses, and executes them in isolated subprocesses against test assertions. |
| 🧠 **Reasoning & `<think>` Parser** | Detects whether models generate chain-of-thought blocks (`<think>...</think>`), calculates thinking token volume, and extracts final answers. |
| 📐 **Context Scaling (512 - 8k)** | Progressively loads larger contexts (512, 1024, 2048, 4096, 8192 tokens) to assess TTFT degradation and memory growth. |
| 📊 **Hardware Telemetry** | Samples GPU/UMA memory usage, GPU utilization %, temperatures, and power draw during execution without requiring root on macOS. |
| 🏆 **Leaderboard & Markdown Reports** | Produces Rich terminal tables with category medals (🥇, 🥈, 🥉) and self-contained Markdown summaries in `results/LATEST_SUMMARY.md`. |
| 🔌 **Built-in MCP Server** | Exposes local Ollama models (`qwen2.5-coder:7b`, `llama3.1:8b`, etc.) as tools for AI agents with zero cloud token cost. |

---

## 🚀 Quick Start

### Option A: macOS (Apple Silicon M1 / M2 / M3 / M4)

We provide an automated setup script that verifies your Apple Silicon chip, checks Python, creates a virtual environment, installs dependencies, and tests your Ollama connection:

```bash
git clone https://github.com/senssei/ollama-benchrig.git
cd ollama-benchrig

# Run the automated setup script:
chmod +x setup_mac.sh
./setup_mac.sh
```

### Option B: Linux / WSL2 (NVIDIA GeForce RTX)

```bash
git clone https://github.com/senssei/ollama-benchrig.git
cd ollama-benchrig

# Create virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run environment diagnostic check
python3 benchmark.py --check
```

---

## 💻 CLI Usage & Examples

### 1. Diagnostic Environment Check
Verifies Ollama connectivity, lists installed models, and displays detected hardware:
```bash
python3 benchmark.py --check
```

### 2. Benchmark All Installed Models
Runs all suites (`speed`, `coding`, `reasoning`, `polish`, `context`) across every installed model:
```bash
python3 benchmark.py --models installed
```

### 3. Benchmark Specific Models
```bash
python3 benchmark.py --models qwen2.5-coder:7b,llama3.1:8b
```

### 4. Run a Specific Test Suite
Available suites: `speed`, `coding`, `reasoning`, `polish`, `context`, `all`.
```bash
# Test only coding with automated unit tests (2 runs per model):
python3 benchmark.py --models qwen2.5-coder:7b --suite coding --runs 2

# Test context scaling up to 8k tokens:
python3 benchmark.py --models llama3.1:8b --suite context
```

### 5. Pull Recommended Models
Pulls missing standard models from the Ollama registry:
```bash
python3 benchmark.py --pull-recommended
```

---

## ⚙️ Configuration (`config.yaml`)

Edit [config.yaml](config.yaml) to customize execution parameters, Ollama endpoints, and scoring weights:

```yaml
ollama:
  base_url: "http://localhost:11434"
  timeout_sec: 180
  warmup: true             # Pre-warms weights into memory before timing
  unload_after_test: true  # Keeps memory clean between model runs
  default_num_ctx: 4096

benchmark:
  default_runs: 1
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

## 🔌 Model Context Protocol (MCP) Server

The repository includes a lightweight stdio MCP server ([ollama_mcp_server.py](ollama_mcp_server.py)) that lets AI coding assistants (such as Google Antigravity, Cursor, or Claude Desktop) offload coding tasks to your local Ollama models with zero token cost.

### Available MCP Tools:
- `ask_local_coder(task, context_code, model)`: Fast code generation, algorithm implementation, or test writing.
- `local_code_review(code, focus)`: Static review and edge-case detection.
- `list_local_models()`: Query installed models and their memory footprints.

### Claude Desktop Configuration:
Add the following to your `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "ollama-local": {
      "command": "python3",
      "args": ["/path/to/ollama-benchrig/ollama_mcp_server.py"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434",
        "DEFAULT_MODEL": "qwen2.5-coder:7b"
      }
    }
  }
}
```

---

## 🤖 Antigravity Skill & Standalone Plugin (`local-coder`)

The repository includes a ready-to-use **Antigravity Customization Skill** (`local-coder`) that equips AI coding agents with subcommands, model profiles, and an automated self-healing AST loop.

### 1-Click Global Installation
Install the skill globally across all Antigravity projects on your machine:
```bash
./install_global_skill.sh
```

### CLI Subcommands
```bash
# Code generation with self-healing syntax loop
ask_local.py code --task "Implement a rate limiter" --output src/rate_limiter.py

# Automated unit test generation
ask_local.py test --file src/rate_limiter.py --framework pytest

# Security and concurrency review
ask_local.py review --file src/server.py --focus "race conditions and memory leaks"

# Refactoring with type hints and docstrings
ask_local.py refactor --file src/util.py --type-hints --docstrings
```

### Standalone Distributable Package
Looking to publish or distribute this skill independently? The complete, standalone package is available in [`packages/antigravity-local-coder/`](packages/antigravity-local-coder/), complete with its own `plugin.json`, `mcp_config.json`, `install.sh`, and MIT license ready for its own GitHub repository.

---

## 🧪 Customizing Scenarios

Test cases are stored as clean JSON files inside the [scenarios/](scenarios/) directory:

- [scenarios/coding.json](scenarios/coding.json): Python coding tasks paired with test assertion arrays evaluated in sandbox subprocesses.
- [scenarios/reasoning.json](scenarios/reasoning.json): Multi-step math and logic puzzles with expected ground truth strings and regex patterns.
- [scenarios/speed.json](scenarios/speed.json): Raw generation and prefill throughput prompts.
- [scenarios/context_scaling.json](scenarios/context_scaling.json): Context scaling tests up to 8k tokens.
- [scenarios/polish.json](scenarios/polish.json): Multilingual tests checking Polish language morphology and syntax.

---

## 📁 Repository Structure

```
ollama-benchrig/
├── benchmark.py              # Main CLI entrypoint
├── config.yaml               # Global configuration (endpoints, weights, thresholds)
├── requirements.txt          # Python dependencies (rich, PyYAML, requests)
├── setup_mac.sh              # Quickstart installer for macOS Apple Silicon
├── LICENSE                   # MIT License
├── README.md                 # Project documentation
├── AGENTS.md                 # Guidelines and skill protocols for AI agents
├── core/
│   ├── client.py             # Ollama REST client with streaming TTFT probe
│   ├── hardware.py           # Cross-platform hardware providers (Darwin UMA / Linux CUDA)
│   ├── runner.py             # Benchmark suite coordinator & scoring engine
│   ├── sandbox.py            # Sandboxed Python test harness runner
│   └── reasoning_parser.py   # <think> tag parser & answer extractor
├── reporting/
│   ├── display.py            # Rich terminal banners and leaderboard UI
│   └── markdown.py           # Comprehensive Markdown report generator
├── scenarios/                # Test scenario definitions (JSON)
├── examples/
│   └── calculator.py         # Sample module for test generation benchmarks
├── results/
│   ├── LATEST_SUMMARY.md     # Latest benchmark Markdown report
│   ├── latest.json           # Latest scorecard JSON
│   └── runs/                 # Historical benchmark runs
└── tests/
    └── test_hardware.py      # Cross-platform hardware provider unit test suite
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
