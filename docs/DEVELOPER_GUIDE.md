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
class BaseRuntimeClient(ABC):
    """Interface for local LLM inference engines."""

    name: str              # 'ollama' | 'foundry' | 'onnx-gpu'
    engine_name: str       # 'llama.cpp' | 'ONNX Runtime GenAI'
    base_url: str

    @abstractmethod
    def is_reachable(self) -> bool:
        """Return True if the runtime daemon is reachable."""
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Return runtime software version."""
        ...

    @abstractmethod
    def list_installed_models(self) -> List[Dict[str, Any]]:
        """Return list of installed models with size, family, and quantization details."""
        ...

    @abstractmethod
    def load_model(self, model: str) -> bool:
        """Explicitly load model into daemon/accelerator memory."""
        ...

    @abstractmethod
    def unload_model(self, model: str) -> bool:
        """Evict model from VRAM/memory pool."""
        ...

    @abstractmethod
    def pull_model(self, model: str) -> bool:
        """Download model into local cache."""
        ...

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        measure_ttft: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute completion request.
        Returns a dict containing:
        - "success": bool
        - "response": str
        - "eval_count": int (number of generated tokens)
        - "eval_tok_per_sec": float
        - "prompt_eval_count": int (prefill tokens)
        - "prompt_tok_per_sec": float
        - "ttft_sec": float (Time to First Token)
        - "total_time_sec": float
        """
        ...
```

### Steps to Register a New Engine:
1. Create your client class inheriting from `BaseRuntimeClient` in `core/` (e.g. `VLLMClient`).
2. Register it in the runtime factory `create_runtime_client(runtime_name, config)` inside `core/client.py`.
3. Add configuration defaults to `config.yaml`.
4. Add unit test coverage in `tests/`.

---

## 🧪 Running Unit Tests

Ollama BenchRig maintains 100% pass rates across its unit test suite with zero cloud dependencies. Tests run offline using mocks and simulated hardware responses.

### Run All Unit Tests:
```bash
python3 -m unittest discover -s tests
```

### Test Suite Structure:
- [`tests/test_hardware.py`](../tests/test_hardware.py): Tests platform detection, `vm_stat` parsing, `ioreg` parsing, and NVIDIA SMI telemetry parsing.
- [`tests/test_foundry_runtime.py`](../tests/test_foundry_runtime.py): Tests Microsoft Foundry REST client, OpenAI schema mapping, streaming TTFT probes, and port auto-discovery.
- [`tests/test_ask_local.py`](../tests/test_ask_local.py): Tests AST syntax validation, self-healing retry logic, and markdown code fence extraction.
- [`tests/test_token_savings.py`](../tests/test_token_savings.py): Validates token savings math and pricing estimation algorithms.

---

## 🛡 Sandbox Security Principles

The code execution engine ([`core/sandbox.py`](../core/sandbox.py)) enforces strict isolation rules:
- **Subprocess Isolation**: Generated code runs in an isolated `subprocess.Popen` with a dedicated PID.
- **Process Group Termination**: If a model generates an infinite loop or blocks indefinitely, the entire process group is terminated using `os.killpg` after the timeout expires.
- **No Global Namespace Pollution**: Code execution does not import or manipulate BenchRig's host process memory.
- **Ephemeral Filesystem Cleanliness**: All harness temporary files are deleted immediately in `finally:` blocks.

---

## 📐 Code Style & Guidelines

- **Python Version**: Compatible with Python 3.10+.
- **Formatting**: Adheres to PEP 8 standards.
- **Type Annotations**: All public classes, functions, and return signatures must include typing annotations (`typing`).
- **Zero-Sudo Rule**: Tools and installation routines must operate in user-space without requiring root escalation.
- **Documentation**: All new features, configuration options, and command-line flags must be documented in English in the `docs/` directory.
