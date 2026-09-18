"""Direct ONNX Runtime GenAI Client for High-Performance Local GPU Inference.

Bypasses CLI/OS restrictions to execute ONNX models natively via onnxruntime_genai
with NVIDIA CUDA / TensorRT acceleration on Linux/WSL2 and Windows.
"""

import gc
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from core.client import BaseRuntimeClient

try:
    import onnxruntime_genai as og
    OG_AVAILABLE = True
except ImportError:
    og = None
    OG_AVAILABLE = False


class OnnxGenAiClient(BaseRuntimeClient):
    """
    Direct client executing ONNX models via onnxruntime_genai on NVIDIA CUDA / TensorRT.
    Provides identical telemetry interface to OllamaClient and FoundryClient.
    """

    name: str = "onnx-gpu"
    display_name: str = "ONNX GenAI (Direct CUDA)"
    engine_name: str = "ONNX Runtime GenAI"

    def __init__(
        self,
        models_dir: Optional[str] = None,
        timeout_sec: int = 300,
        default_max_tokens: int = 4096,
        include_foundry_cache: bool = False,
    ):
        super().__init__(base_url="local://onnxruntime-genai", timeout_sec=timeout_sec)
        self.models_dir = os.path.abspath(models_dir) if models_dir else os.path.abspath("models")
        self.default_max_tokens = default_max_tokens
        self.include_foundry_cache = include_foundry_cache
        self._loaded_model = None
        self._loaded_tokenizer = None
        self._loaded_model_name: Optional[str] = None

    def is_reachable(self) -> bool:
        """Check if onnxruntime_genai library is available and operational."""
        return OG_AVAILABLE

    def is_cuda_available(self) -> bool:
        """Check if CUDA execution provider is available in onnxruntime_genai."""
        if not OG_AVAILABLE or og is None:
            return False
        try:
            return bool(og.is_cuda_available())
        except Exception:
            return False

    def get_version(self) -> str:
        """Return onnxruntime_genai version and acceleration capability."""
        if not OG_AVAILABLE or og is None:
            return "not_installed"
        cuda_status = "CUDA" if self.is_cuda_available() else "CPU"
        return f"{getattr(og, '__version__', 'unknown')} ({cuda_status})"

    def _resolve_model_path(self, model_name: str) -> Optional[str]:
        """Resolve model name or directory path to the folder containing genai_config.json."""
        candidates = [
            model_name,
            os.path.join(self.models_dir, model_name),
            os.path.expanduser(f"~/.foundry/cache/models/Microsoft/{model_name}"),
            os.path.expanduser(f"~/.cache/models/{model_name}"),
        ]

        # Also search for subdirectories or matching folders
        if os.path.isdir(self.models_dir):
            for entry in os.listdir(self.models_dir):
                full = os.path.join(self.models_dir, entry)
                if os.path.isdir(full) and (model_name.lower() in entry.lower() or entry.lower() in model_name.lower()):
                    candidates.append(full)

        if self.include_foundry_cache:
            foundry_dir = os.path.expanduser("~/.foundry/cache/models/Microsoft")
            if os.path.isdir(foundry_dir):
                for entry in os.listdir(foundry_dir):
                    full = os.path.join(foundry_dir, entry)
                    if os.path.isdir(full) and (model_name.lower() in entry.lower() or entry.lower() in model_name.lower()):
                        candidates.append(full)
                        # Check v1, v2, v5 subdirectories
                        for sub in os.listdir(full):
                            sub_full = os.path.join(full, sub)
                            if os.path.isdir(sub_full):
                                candidates.append(sub_full)

        for path in candidates:
            if os.path.isdir(path):
                cfg_path = os.path.join(path, "genai_config.json")
                if os.path.isfile(cfg_path):
                    return os.path.abspath(path)

        return None

    def list_installed_models(self) -> List[Dict[str, Any]]:
        """List local ONNX models containing genai_config.json."""
        installed = []
        search_dirs = [self.models_dir]
        if self.include_foundry_cache:
            search_dirs.append(os.path.expanduser("~/.foundry/cache/models/Microsoft"))

        seen_paths = set()
        for root_dir in search_dirs:
            if not os.path.isdir(root_dir):
                continue
            for root, _, files in os.walk(root_dir):
                if "genai_config.json" in files:
                    abs_root = os.path.abspath(root)
                    if abs_root in seen_paths:
                        continue
                    seen_paths.add(abs_root)

                    model_name = os.path.basename(abs_root)
                    if model_name in ("v1", "v2", "v3", "v4", "v5"):
                        parent = os.path.basename(os.path.dirname(abs_root))
                        model_name = f"{parent}-{model_name}"

                    # Calculate size
                    total_bytes = 0
                    for f in os.listdir(abs_root):
                        fp = os.path.join(abs_root, f)
                        if os.path.isfile(fp):
                            total_bytes += os.path.getsize(fp)

                    # Read genai_config for model info
                    family = "onnx"
                    try:
                        with open(os.path.join(abs_root, "genai_config.json"), "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            family = cfg.get("model", {}).get("type", "onnx")
                    except Exception:
                        pass

                    quant = "INT4" if "int4" in model_name.lower() else "ONNX"
                    cuda_tag = "CUDA" if ("cuda" in model_name.lower() or "gpu" in model_name.lower()) else "CPU/GPU"

                    installed.append({
                        "name": model_name,
                        "model": model_name,
                        "path": abs_root,
                        "size": total_bytes,
                        "runtime": self.name,
                        "details": {
                            "family": family,
                            "parameter_size": "N/A",
                            "quantization_level": f"{quant} ({cuda_tag})",
                        },
                    })

        return installed

    def load_model(self, model_name: str) -> bool:
        """Load ONNX GenAI model and tokenizer into GPU memory."""
        if not OG_AVAILABLE or og is None:
            return False

        if self._loaded_model is not None and self._loaded_model_name == model_name:
            return True

        self.unload_model(self._loaded_model_name or "")

        model_path = self._resolve_model_path(model_name)
        if not model_path:
            return False

        try:
            loaded_model = None
            if hasattr(og, "Config"):
                try:
                    config = og.Config(model_path)
                    if self.is_cuda_available() and hasattr(config, "append_provider"):
                        config.clear_providers()
                        config.append_provider("cuda")
                    loaded_model = og.Model(config)
                except Exception:
                    loaded_model = None

            if loaded_model is None:
                loaded_model = og.Model(model_path)

            self._loaded_model = loaded_model
            self._loaded_tokenizer = og.Tokenizer(self._loaded_model)
            self._loaded_model_name = model_name
            return True
        except Exception:
            self._loaded_model = None
            self._loaded_tokenizer = None
            self._loaded_model_name = None
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload ONNX GenAI model from memory and release accelerator resources."""
        self._loaded_model = None
        self._loaded_tokenizer = None
        self._loaded_model_name = None
        gc.collect()
        return True

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Return currently loaded model info."""
        if self._loaded_model is not None and self._loaded_model_name:
            return [{"name": self._loaded_model_name, "model": self._loaded_model_name}]
        return []

    def _format_prompt(self, prompt: str, system: Optional[str] = None) -> str:
        """Format prompt using standard instruction markers."""
        if system:
            return f"<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        measure_ttft: bool = True,
    ) -> Dict[str, Any]:
        """Execute text generation using onnxruntime_genai with precision telemetry."""
        if not OG_AVAILABLE or og is None:
            return {
                "success": False,
                "error": "onnxruntime_genai is not installed",
                "model": model,
                "runtime": self.name,
                "engine": self.engine_name,
                "response": "",
                "eval_tok_per_sec": 0.0,
                "prompt_tok_per_sec": 0.0,
                "ttft_sec": 0.0,
                "total_time_sec": 0.0,
            }

        start_wall_time = time.perf_counter()

        # Ensure model is loaded
        if self._loaded_model is None or self._loaded_model_name != model:
            loaded = self.load_model(model)
            if not loaded:
                return {
                    "success": False,
                    "error": f"Failed to load ONNX model: {model}",
                    "model": model,
                    "runtime": self.name,
                    "engine": self.engine_name,
                    "response": "",
                    "eval_tok_per_sec": 0.0,
                    "prompt_tok_per_sec": 0.0,
                    "ttft_sec": 0.0,
                    "total_time_sec": round(time.perf_counter() - start_wall_time, 3),
                }

        opts = options or {}
        max_new_tokens = opts.get("num_predict") or opts.get("max_tokens") or 2048
        temperature = float(opts.get("temperature", 0.0))
        top_p = float(opts.get("top_p", 1.0))

        full_prompt = self._format_prompt(prompt, system=system)
        tokenizer = self._loaded_tokenizer

        try:
            tokens = tokenizer.encode(full_prompt)
            prompt_eval_count = len(tokens)

            params = og.GeneratorParams(self._loaded_model)
            max_length = prompt_eval_count + max_new_tokens

            search_kwargs = {"max_length": max_length}
            if temperature > 0.0:
                search_kwargs["temperature"] = temperature
                search_kwargs["top_p"] = top_p
                search_kwargs["do_sample"] = True
            else:
                search_kwargs["do_sample"] = False

            params.set_search_options(**search_kwargs)

            generator = og.Generator(self._loaded_model, params)
            generator.append_tokens(tokens)

            first_token_time: Optional[float] = None
            out_tokens: List[int] = []

            gen_start_time = time.perf_counter()

            while not generator.is_done():
                generator.generate_next_token()
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                next_toks = generator.get_next_tokens()
                if next_toks:
                    out_tokens.append(next_toks[0])

            end_wall_time = time.perf_counter()
            response_text = tokenizer.decode(out_tokens)

            eval_count = len(out_tokens)
            ttft_sec = (first_token_time - gen_start_time) if first_token_time else (end_wall_time - gen_start_time)
            decode_dur_sec = (end_wall_time - first_token_time) if (first_token_time and eval_count > 1) else (end_wall_time - gen_start_time)
            total_time_sec = end_wall_time - start_wall_time

            eval_tok_sec = (eval_count - 1) / decode_dur_sec if (decode_dur_sec > 0 and eval_count > 1) else (eval_count / total_time_sec if total_time_sec > 0 else 0.0)
            prompt_tok_sec = prompt_eval_count / ttft_sec if ttft_sec > 0 else 0.0

            return {
                "success": True,
                "model": model,
                "runtime": self.name,
                "engine": self.engine_name,
                "response": response_text,
                "eval_count": eval_count,
                "eval_tok_per_sec": round(eval_tok_sec, 2),
                "prompt_eval_count": prompt_eval_count,
                "prompt_tok_per_sec": round(prompt_tok_sec, 2),
                "ttft_sec": round(ttft_sec, 3),
                "load_time_sec": 0.0,
                "total_time_sec": round(total_time_sec, 3),
                "raw_metrics": {
                    "total_time_sec": total_time_sec,
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "runtime": self.name,
                "engine": self.engine_name,
                "response": "",
                "eval_tok_per_sec": 0.0,
                "prompt_tok_per_sec": 0.0,
                "ttft_sec": 0.0,
                "total_time_sec": round(time.perf_counter() - start_wall_time, 3),
            }
