"""Ollama API Client with high-precision timing and hardware offload metrics."""

import json
import time
from typing import Any, Dict, Generator, List, Optional
import requests


class OllamaClient:
    """Client for Ollama REST API with native performance metrics."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout_sec: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def is_reachable(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_version(self) -> str:
        """Get Ollama server version."""
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=3)
            if r.status_code == 200:
                return r.json().get("version", "unknown")
        except Exception:
            pass
        return "unknown"

    def list_installed_models(self) -> List[Dict[str, Any]]:
        """Return list of installed models with metadata."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception:
            pass
        return []

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Return currently loaded models in memory."""
        try:
            r = requests.get(f"{self.base_url}/api/ps", timeout=5)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception:
            pass
        return []

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from VRAM/RAM immediately by setting keep_alive to 0."""
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def pull_model(self, model_name: str, stream_callback=None) -> bool:
        """Pull a model from Ollama library."""
        try:
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=600,
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if stream_callback:
                        stream_callback(data)
                    if data.get("status") == "success":
                        return True
            return True
        except Exception as e:
            if stream_callback:
                stream_callback({"status": "error", "error": str(e)})
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        measure_ttft: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a generation request and return full response with Ollama native metrics.
        
        Calculates:
        - eval_tok_per_sec (Generation speed)
        - prompt_tok_per_sec (Prompt prefill speed)
        - ttft_sec (Time to first token)
        - load_time_sec (Model load duration)
        - total_time_sec
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": measure_ttft,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        start_wall_time = time.perf_counter()
        first_token_time: Optional[float] = None
        collected_response: List[str] = []
        final_metrics: Dict[str, Any] = {}

        try:
            if measure_ttft:
                # Streaming mode to accurately capture TTFT
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=self.timeout_sec,
                )
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    text = chunk.get("response", "")
                    if text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        collected_response.append(text)

                    if chunk.get("done", False):
                        final_metrics = chunk
                        break
            else:
                # Non-streaming mode
                payload["stream"] = False
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                r.raise_for_status()
                final_metrics = r.json()
                collected_response.append(final_metrics.get("response", ""))

            end_wall_time = time.perf_counter()

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "response": "",
                "eval_tok_per_sec": 0.0,
                "prompt_tok_per_sec": 0.0,
                "ttft_sec": 0.0,
                "total_time_sec": round(time.perf_counter() - start_wall_time, 3),
            }

        response_text = "".join(collected_response)

        # Ollama provides timings in nanoseconds
        prompt_eval_count = final_metrics.get("prompt_eval_count", 0)
        prompt_eval_dur_ns = final_metrics.get("prompt_eval_duration", 0)
        eval_count = final_metrics.get("eval_count", 0)
        eval_dur_ns = final_metrics.get("eval_duration", 0)
        load_dur_ns = final_metrics.get("load_duration", 0)
        total_dur_ns = final_metrics.get("total_duration", 0)

        # Calculate exact speeds
        prompt_tok_sec = (
            (prompt_eval_count / (prompt_eval_dur_ns / 1e9))
            if prompt_eval_dur_ns > 0
            else 0.0
        )
        eval_tok_sec = (
            (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0
        )

        ttft_sec = (
            round(first_token_time - start_wall_time, 3)
            if first_token_time
            else round(prompt_eval_dur_ns / 1e9, 3)
        )

        return {
            "success": True,
            "model": model,
            "response": response_text,
            "eval_count": eval_count,
            "eval_tok_per_sec": round(eval_tok_sec, 2),
            "prompt_eval_count": prompt_eval_count,
            "prompt_tok_per_sec": round(prompt_tok_sec, 2),
            "ttft_sec": ttft_sec,
            "load_time_sec": round(load_dur_ns / 1e9, 3),
            "total_time_sec": round(
                total_dur_ns / 1e9 if total_dur_ns > 0 else (end_wall_time - start_wall_time),
                3,
            ),
            "raw_metrics": {
                "total_duration": total_dur_ns,
                "load_duration": load_dur_ns,
                "prompt_eval_duration": prompt_eval_dur_ns,
                "eval_duration": eval_dur_ns,
            },
        }
