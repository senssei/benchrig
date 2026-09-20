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
