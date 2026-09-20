# BenchRig

**Cross-platform benchmarking and hardware profiling for local LLMs** on Ollama, Microsoft Foundry Local, direct ONNX Runtime
GenAI and [Prism](runtimes.md#prism). Tailored for macOS Apple Silicon (Metal, Unified Memory) and Linux / WSL2 (NVIDIA CUDA).

!!! warning "Alpha software (v0.1.0)"
    It works and is tested, but flags, file formats and scoring may still change. See the [changelog](changelog.md).

## Try it

```bash
pipx install benchrig                 # or: pip install benchrig
benchrig --check                      # runtimes, accelerator, installed models
benchrig --models qwen2.5-coder:7b --suite coding
```

Reports land in `results/` (`LATEST_SUMMARY.md`, `latest.json`, and one JSON file per run in `results/runs/`).

## What you get

<div class="grid cards" markdown>

- **Practical workloads, not perplexity.** Coding tasks are executed in a sandbox against unit-test assertions; reasoning answers
  are checked against ground truth (including `<think>` output). See [Benchmark suites](benchmark-suites.md).
- **Several runtimes, one scorecard.** Compare `llama.cpp` and ONNX Runtime GenAI on identical hardware, with a fair 1:1
  parameter set. See [Runtimes](runtimes.md) and [Cross-engine benchmarking](tutorials/cross-engine-benchmarking.md).
- **True time to first token.** Measured from the stream, next to decode and prefill throughput.
- **Hardware telemetry.** VRAM or unified memory, GPU utilisation, power and temperature while a model runs.
  See [Hardware telemetry](hardware-telemetry.md).
- **Your scenarios.** Bundled suites are plain JSON and can be overridden with `./scenarios` or `--scenarios-dir`.
  See [Custom scenarios](tutorials/custom-scenarios.md).

</div>

## Where to go next

| I want to... | Read |
| :--- | :--- |
| Run my first benchmark | [Quickstart](tutorials/quickstart.md) |
| Choose or configure a runtime | [Runtimes](runtimes.md), [Configuration](configuration.md) |
| Look up a flag or a recipe | [CLI reference](cli.md) |
| Use an NVIDIA GPU with Foundry Local on WSL2 | [Foundry GPU setup](tutorials/foundry-gpu-setup.md), [CUDA & TensorRT guide](foundry-wsl-cuda-tensorrt.md) |
| Write my own scenarios | [Scenario reference](scenarios.md), [Custom scenarios](tutorials/custom-scenarios.md) |
| Understand or extend the code | [Architecture](architecture.md), [Development](development.md) |

## Related projects

- [prism-local](https://github.com/senssei/prism-local): local server and CLI for ONNX Runtime GenAI and Ollama; benchmarked here as the `prism` runtime.
- [local-coders](https://github.com/senssei/local-coders): agent skills and MCP servers that offload coding work to local models.
