# Contributing

Thanks for helping out. BenchRig is a small alpha project; issues and pull requests are welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

```bash
ruff check . && ruff format --check . && python -m pytest
python -m build && twine check --strict dist/*     # packaging
```

The test suite needs no GPU, Ollama, network or model files; keep new tests hermetic (mock clients, temp dirs).

## Layout

Everything shipped lives in `benchrig/`. The default `config.yaml` and the scenario suites are package data under
`benchrig/data/`. Agent skills and MCP servers live in [local-coders](https://github.com/senssei/local-coders).

## Pull requests

Describe what changed and why, note how you tested it, and update `CHANGELOG.md`. CI must pass.
