"""Canonical runtime names and how they are shown to the user."""

RUNTIME_LABELS: dict[str, str] = {
    "ollama": "Ollama",
    "foundry": "MS Foundry",
    "onnx-gpu": "ONNX GenAI",
    "prism": "Prism",
}


def runtime_label(runtime: object, default: str = "Ollama") -> str:
    """Human-readable runtime name for a scorecard/result `runtime` value (matches by substring, like the reports do)."""
    text = str(runtime or "").lower()
    if "prism" in text:
        return RUNTIME_LABELS["prism"]
    if "onnx" in text:
        return RUNTIME_LABELS["onnx-gpu"]
    if "foundry" in text:
        return RUNTIME_LABELS["foundry"]
    if "ollama" in text:
        return RUNTIME_LABELS["ollama"]
    return default


def scorecard_prefill(scorecard: dict) -> float:
    """Prefill speed of a scorecard: the runtime-independent effective value, else the older engine-reported one."""
    return scorecard.get("avg_prefill_eff_tok_sec", scorecard.get("avg_prompt_tok_sec", 0.0))


def record_prefill(record: dict) -> float:
    """Prefill speed of one result record (effective value when present)."""
    return record.get("prefill_eff_tok_per_sec", record.get("prompt_tok_per_sec", 0.0))


PREFILL_NOTE = (
    "Prefill (eff.) = prompt tokens / time to first token. It includes request overhead and is defined the same way for "
    "every runtime; engine-reported prefill speeds (Ollama's `prompt_eval_duration`) are kept in the JSON results."
)
MODEL_MEMORY_NOTE = (
    "Peak memory is the whole GPU (`nvidia-smi memory.used`, including other processes). Model memory is the peak minus the "
    "GPU memory in use before the model was loaded."
)

DIRTY_BASELINE_NOTE = (
    "Model memory is shown as - for a model whose baseline was taken while another model was still loaded (Prism keeps one "
    "model loaded and cannot unload it): peak minus that baseline would understate the model. Restart the server before "
    "benchmarking a model to get the figure."
)

ESTIMATED_USAGE_NOTE = (
    "Token counts marked as estimated come from a server that reported no `usage` (prompt tokens are guessed from the word count, "
    "generated tokens are the streamed chunks), so prefill and decode speeds for it are approximate. Prism sends `usage` when "
    "started from a version that supports `stream_options.include_usage`."
)


# Warning text is part of the spec (spec.md §3). Tests in tests/test_warnings.py guard key substrings.
GENERIC_CPU_ON_CUDA_WARNING = (
    "generic-cpu execution provider on a CUDA host is very slow (2–22 tok/s for qwen; "
    "Phi-3.5-mini may not finish 4500 tokens in 5 min). Use the cuda provider or benchmark on CPU-only hardware."
)


def warning_for_provider(provider: object, host_gpu_type: object) -> str | None:
    """Return a warning string when ``provider`` is ``generic-cpu`` on a CUDA host, else ``None``.

    On Apple Silicon, ``generic-cpu`` is the normal CPU execution provider (not the slow path) and produces
    no warning. Empty / missing providers are also a no-op.
    """
    p = str(provider or "").strip().lower()
    host = str(host_gpu_type or "").strip().lower()
    if p == "generic-cpu" and host == "nvidia":
        return GENERIC_CPU_ON_CUDA_WARNING
    return None
