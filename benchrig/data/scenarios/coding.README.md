# 💻 Coding Suite Scenarios (`coding.json`)

This directory contains bundled coding benchmark scenarios evaluated by BenchRig's coding suite (`benchrig --suite coding`).

## Language Scope

- **Supported Language**: **Python 3 only**.
- **Not Bundled**: JavaScript/TypeScript, Go, Rust, or C++ scenario packs are currently not bundled in default distributions.

## How Scenarios Work

Each scenario in `coding.json` defines:
1. `id`: Unique identifier (e.g. `coding_flatten_dict`).
2. `name`: Human-readable problem title.
3. `prompt`: Self-contained problem description instructing the model to generate a Python function or class.
4. `test_assertions`: An array of independent Python `assert` statements testing edge cases, algorithmic efficiency, and return types.

## Execution and Isolation

Scenarios in `coding.json` are evaluated by [`benchrig/core/sandbox.py`](../../core/sandbox.py):
- Code blocks are extracted from the LLM completion (ignoring any `<think>...</think>` reasoning chains).
- A standalone test harness combines the extracted solution with each assertion.
- Execution runs in an isolated subprocess with its own process group and a strict 10-second timeout.
- Test verdicts and assertion pass counts are reported out-of-band using a temporary result file authenticated by a cryptographic secret token; candidate stdout/stderr cannot manipulate verdicts.
