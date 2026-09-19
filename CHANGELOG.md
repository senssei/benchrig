# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [SemVer](https://semver.org/) (pre-1.0: minor
versions may include breaking changes).

## [0.1.0] - Unreleased

First PyPI release.

### Added
- Installable package `benchrig` (import `benchrig`) with a `benchrig` console script (also `python -m benchrig`).
- `benchrig --version`, `--config` and `--scenarios-dir`. The default `config.yaml` and all scenario suites are bundled;
  `./config.yaml`, `./scenarios` or `$BENCHRIG_CONFIG` override them.
- Direct ONNX models are looked up in `$BENCHRIG_MODEL_DIRS`, then `./models`, then `~/.benchrig/models`.
- Benchmarking across Ollama, Microsoft Foundry Local and direct ONNX Runtime GenAI (CUDA), with hardware telemetry,
  sandboxed coding tests, reasoning/speed/context/Polish suites and 1:1 cross-engine comparison reports.
- CI (Python 3.10-3.13, wheel smoke test) and a trusted-publishing workflow for TestPyPI and PyPI.

### Fixed
- `FoundryClient` reports the engine that actually serves each model (from `owned_by`, or the `ollama:` prefix) instead of always
  "ONNX Runtime GenAI"; matters for multi-engine servers such as Prism.
- With an explicit endpoint (`auto_detect_port: false`) `load_model`/`unload_model` no longer call the `foundry` CLI (which
  talks to a different daemon); `load_model` reports whether the server lists the model.

### Changed
- The agent skills and MCP servers moved to [local-coders](https://github.com/senssei/local-coders).
- `core/` and `reporting/` moved to `benchrig.core` and `benchrig.reporting`; `benchmark.py` is now `benchrig.cli`
  (run `benchrig ...` instead of `python3 benchmark.py ...`).
- The CUDA library bootstrap looks in the active environment's `site-packages` instead of a `.venv` next to the script.
- `requirements*.txt` replaced by `pip install -e ".[dev]"`.
