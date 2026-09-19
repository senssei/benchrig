# 👩‍💻 Developer & Contributing Guide

This guide provides technical specifications for contributing to **Ollama BenchRig**, authoring new runtime providers, expanding test coverage, and maintaining architecture standards.

---

## 🏗 Architectural Conventions

Ollama BenchRig is architected around two core abstraction layers:
1. **Hardware Abstraction Layer (HAL)**: Defined in [`core/hardware.py`](../core/hardware.py) via `BaseHardwareProvider`.
2. **Runtime Abstraction Layer (RAL)**: Defined in [`core/client.py`](../core/client.py) via `BaseRuntimeClient`.

All business logic, test runners, and telemetry collectors interface strictly through these abstractions.

---

## 🔌 Adding a New Runtime Provider

To support a new local inference server (e.g. **vLLM**, **SGLang**, **TensorRT-LLM**, or **LM Studio**), implement the `BaseRuntimeClient` interface in `core/client.py`:

```python
class BaseRuntimeClient:
    """Interface for local LLM inference engines (see core/client.py)."""

    name: str  # 'ollama' | 'foundry' | 'onnx-gpu'
    display_name: str  # shown in terminal output
    engine_name: str  # 'llama.cpp' | 'ONNX Runtime GenAI'

    # Required: the base class raises NotImplementedError.
    def is_reachable(self) -> bool: ...
    def get_version(self) -> str: ...
    def list_installed_models(self) -> List[Dict[str, Any]]: ...
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        measure_ttft: bool = True,
    ) -> Dict[str, Any]: ...

    # Optional: sensible defaults are provided (no-op / empty / False).
    def get_running_models(self) -> List[Dict[str, Any]]: ...
    def load_model(self, model_name: str) -> bool: ...
    def unload_model(self, model_name: str) -> bool: ...
    def pull_model(self, model_name: str, stream_callback=None) -> bool: ...
```

`generate()` returns a dict containing:
- `"success"` (bool) and `"response"` (str)
- `"eval_count"` (generated tokens) and `"eval_tok_per_sec"`
- `"prompt_eval_count"` (prefill tokens) and `"prompt_tok_per_sec"`
- `"ttft_sec"` (Time to First Token) and `"total_time_sec"`

On failure, return `self._failure_result(model, error, start_time)` so the record shape stays uniform.

### Steps to Register a New Engine:
1. Create your client class inheriting from `BaseRuntimeClient` in `core/` (e.g. `VLLMClient`).
2. Register it in the runtime factory `create_runtime_client(runtime_name, config)` inside `core/client.py`.
3. Add configuration defaults to `config.yaml`.
4. Add unit test coverage in `tests/`.

---

## 🧪 Running Tests & Linting

Tests run offline using mocks and simulated hardware responses: no network, GPU, or running daemon is required.

```bash
pip install -r requirements-dev.txt   # pytest + ruff

pytest                                # run the full suite (configured in pyproject.toml)
ruff check .                          # lint (E, W, F, I, B, UP)
ruff format .                         # auto-format (use --check in CI)
```

The same three checks run in CI ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) on Python 3.10 and 3.12. The tests are plain `unittest` classes, so `python3 -m unittest discover -s tests` also works without pytest.

### Test Suite Structure:
- [`tests/test_hardware.py`](../tests/test_hardware.py): Tests platform detection, `vm_stat` parsing, `ioreg` parsing, and NVIDIA SMI telemetry parsing.
- [`tests/test_foundry_runtime.py`](../tests/test_foundry_runtime.py): Tests Microsoft Foundry REST client, OpenAI schema mapping, streaming TTFT probes, and port auto-discovery.
- [`tests/test_ask_local.py`](../tests/test_ask_local.py): Tests AST syntax validation, self-healing retry logic, and markdown code fence extraction.
- [`tests/test_token_savings.py`](../tests/test_token_savings.py): Validates token savings math and pricing estimation algorithms.
- [`tests/test_runner_suites.py`](../tests/test_runner_suites.py): Verifies the result-record schema of every suite and scorecard maths using a fake runtime client.
- [`tests/test_sandbox.py`](../tests/test_sandbox.py): Covers partial passes, syntax errors, timeouts, and temp-file cleanup in the code sandbox.
- [`tests/test_benchmark_cli.py`](../tests/test_benchmark_cli.py): Covers `--models` / `--pair` target resolution and suite registry consistency.
- [`tests/test_packaging_sync.py`](../tests/test_packaging_sync.py): Fails if the standalone `packages/antigravity-local-coder/` code copies drift from the repo sources.

---

## 🛡 Sandbox Security Principles

The code execution engine ([`core/sandbox.py`](../core/sandbox.py)) enforces strict isolation rules:
- **Subprocess Isolation**: Generated code runs in an isolated `subprocess.Popen` in its own session (dedicated process group).
- **Process Group Termination**: If a model generates an infinite loop or blocks indefinitely, the entire process group is terminated using `os.killpg` after the timeout expires, so spawned grandchildren cannot outlive the test.
- **No Global Namespace Pollution**: Code execution does not import or manipulate BenchRig's host process memory.
- **Ephemeral Filesystem Cleanliness**: All harness temporary files are deleted immediately in `finally:` blocks.
- **Packaged copies**: `packages/antigravity-local-coder/` mirrors `ollama_mcp_server.py` and `ask_local.py`; edit the repo source, then copy it over (enforced by `tests/test_packaging_sync.py`).

---

## 📐 Code Style & Guidelines

- **Python Version**: Compatible with Python 3.10+.
- **Formatting & Linting**: Enforced by [ruff](https://docs.astral.sh/ruff/) (`ruff format`, `ruff check`); configuration lives in `pyproject.toml`.
- **Type Annotations**: All public classes, functions, and return signatures must include typing annotations (`typing`).
- **Zero-Sudo Rule**: Tools and installation routines must operate in user-space without requiring root escalation.
- **Documentation**: All new features, configuration options, and command-line flags must be documented in English in the `docs/` directory.
