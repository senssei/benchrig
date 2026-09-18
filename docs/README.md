# 📚 Ollama BenchRig Documentation & Tutorials

Welcome to the comprehensive technical documentation and tutorial library for **Ollama BenchRig** (`ollama-benchrig`).

---

## 🧭 Navigation Index

### 🚀 Step-by-Step Tutorials (`docs/tutorials/`)
Hands-on, step-by-step walkthroughs designed to take you from initial setup to advanced agent integration:

| Tutorial | Description |
| :--- | :--- |
| [**1. Quickstart Guide**](tutorials/01_QUICKSTART_GUIDE.md) | Zero to first benchmark in 5 minutes across macOS Apple Silicon and Linux/WSL2 NVIDIA CUDA. |
| [**2. Foundry GPU Setup (WSL2 / Linux)**](tutorials/02_FOUNDRY_GPU_SETUP.md) | Complete walkthrough for unlocking full NVIDIA GPU acceleration on Microsoft Foundry Local (Option 2 & Option 3). |
| [**3. Fair 1:1 Cross-Engine Benchmarking**](tutorials/03_CROSS_ENGINE_BENCHMARKING.md) | Standardizing parameters, running against cached baselines (`--baseline`), and analyzing side-by-side scorecards. |
| [**4. Zero-Token-Cost Agent Integration (MCP & Skills)**](tutorials/04_AGENT_INTEGRATION_MCP.md) | Connecting local LLMs to coding agents (Antigravity, Claude Code, Cursor) with AST self-healing loops. |
| [**5. Authoring Custom Benchmark Scenarios**](tutorials/05_CUSTOM_SCENARIO_AUTHORING.md) | Designing deterministic coding challenges, reasoning puzzles, and domain-specific test harnesses. |

---

### 🏗 Architecture & Core Engineering
Deep dives into the system design, interfaces, and evaluation harnesses:

| Guide | Scope |
| :--- | :--- |
| [**System Architecture & Engine Design**](ARCHITECTURE.md) | System components, Runtime Abstraction Layer (`BaseRuntimeClient`), HAL, and sandboxing. |
| [**CLI Reference & Usage Manual**](CLI_USAGE.md) | Complete reference for `benchmark.py`, options, flags, and recipes. |
| [**Configuration Reference (`config.yaml`)**](CONFIGURATION.md) | Settings, timeouts, hardware thresholds, 1:1 model pairs, and weights. |
| [**Developer & Contributing Guide**](DEVELOPER_GUIDE.md) | Adding new runtime providers, expanding test coverage, and development workflow. |

---

### ⚡ Accelerator & Hardware Telemetry
Guides tailored to specific hardware architectures and platforms:

| Guide | Scope |
| :--- | :--- |
| [**Foundry Local, CUDA & TensorRT on WSL2**](FOUNDRY_WSL_CUDA_TENSORRT_GUIDE.md) | Comprehensive technical deep dive on ONNX Runtime GenAI, Execution Providers, and WSL2 driver bridging. |
| [**Hardware Profiling & Telemetry**](HARDWARE_TELEMETRY.md) | macOS Unified Memory (UMA), Metal buffer tracking, and NVIDIA VRAM telemetry. |

---

### 🧪 Benchmark Scenarios & Agent Skills
Specifications for evaluation suites and local coding agent skills:

| Guide | Scope |
| :--- | :--- |
| [**Benchmark Suites Specification**](BENCHMARK_SUITES.md) | Detailed specifications for Speed, Coding, Reasoning, Polish NLP, and Context scaling suites. |
| [**Scenario Authoring & Schema Reference**](SCENARIOS_GUIDE.md) | JSON schemas, sandbox test execution, and `<think>` chain-of-thought parsing. |
| [**MCP Server Integration**](MCP_SERVER.md) | Model Context Protocol servers (`ollama-local` and `foundry-local`). |
| [**Ollama Coder Skill (`ollama-coder`)**](OLLAMA_CODER_SKILL.md) | Ollama-powered coding, unit test generation, security reviews, and refactoring. |
| [**Foundry Coder Skill (`foundry-coder`)**](FOUNDRY_CODER_SKILL.md) | Microsoft Foundry Local coding skill for agents. |
