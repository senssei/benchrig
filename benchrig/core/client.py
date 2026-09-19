"""Unified Local LLM API Clients supporting Ollama and Microsoft Foundry Server Runtime."""

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

import requests


class BaseRuntimeClient:
    """Abstract base class establishing a uniform contract across local inference engines."""

    name: str = "base"
    display_name: str = "Base Runtime"
    engine_name: str = "Unknown Engine"

    def __init__(self, base_url: str = "", timeout_sec: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def is_reachable(self) -> bool:
        """Check if server runtime is reachable and responsive."""
        raise NotImplementedError

    def get_version(self) -> str:
        """Return server runtime version string."""
        raise NotImplementedError

    def list_installed_models(self) -> list[dict[str, Any]]:
        """Return list of locally installed / cached models with metadata."""
        raise NotImplementedError

    def get_running_models(self) -> list[dict[str, Any]]:
        """Return models currently loaded in memory/accelerator."""
        return []

    def load_model(self, model_name: str) -> bool:
        """Load model into memory/accelerator before executing inference."""
        return True

    def unload_model(self, model_name: str) -> bool:
        """Unload model from accelerator/RAM to ensure clean baseline for next test."""
        return True

    def pull_model(self, model_name: str, stream_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Pull / download a model to local cache."""
        return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        measure_ttft: bool = True,
    ) -> dict[str, Any]:
        """Execute text generation with precision timing and token throughput telemetry."""
        raise NotImplementedError

    def _failure_result(self, model: str, error: Exception, start_time: float) -> dict[str, Any]:
        """Uniform `generate()` result for a failed request (start_time from time.perf_counter())."""
        return {
            "success": False,
            "error": str(error),
            "model": model,
            "runtime": self.name,
            "engine": self.engine_name,
            "response": "",
            "eval_tok_per_sec": 0.0,
            "prompt_tok_per_sec": 0.0,
            "ttft_sec": 0.0,
            "total_time_sec": round(time.perf_counter() - start_time, 3),
        }


class OllamaClient(BaseRuntimeClient):
    """Client for Ollama REST API with native performance metrics and llama.cpp backend."""

    name: str = "ollama"
    display_name: str = "Ollama"
    engine_name: str = "llama.cpp"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_sec: int = 180,
        default_num_ctx: int = 4096,
    ):
        super().__init__(base_url=base_url, timeout_sec=timeout_sec)
        self.default_num_ctx = default_num_ctx

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

    def list_installed_models(self) -> list[dict[str, Any]]:
        """Return list of installed models with metadata."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = r.json().get("models", [])
                for m in models:
                    m["runtime"] = self.name
                return models
        except Exception:
            pass
        return []

    def get_running_models(self) -> list[dict[str, Any]]:
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

    def pull_model(self, model_name: str, stream_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
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
        system: str | None = None,
        options: dict[str, Any] | None = None,
        measure_ttft: bool = True,
    ) -> dict[str, Any]:
        """
        Send a generation request and return full response with Ollama native metrics.

        Calculates:
        - eval_tok_per_sec (Generation speed)
        - prompt_tok_per_sec (Prompt prefill speed)
        - ttft_sec (Time to first token)
        - load_time_sec (Model load duration)
        - total_time_sec
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": measure_ttft,
        }
        if system:
            payload["system"] = system

        merged_options = {"num_ctx": self.default_num_ctx}
        if options:
            merged_options.update(options)
        payload["options"] = merged_options

        start_wall_time = time.perf_counter()
        first_token_time: float | None = None
        collected_response: list[str] = []
        final_metrics: dict[str, Any] = {}

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
            return self._failure_result(model, e, start_wall_time)

        response_text = "".join(collected_response)

        # Ollama provides timings in nanoseconds
        prompt_eval_count = final_metrics.get("prompt_eval_count", 0)
        prompt_eval_dur_ns = final_metrics.get("prompt_eval_duration", 0)
        eval_count = final_metrics.get("eval_count", 0)
        eval_dur_ns = final_metrics.get("eval_duration", 0)
        load_dur_ns = final_metrics.get("load_duration", 0)
        total_dur_ns = final_metrics.get("total_duration", 0)

        # Calculate exact speeds
        prompt_tok_sec = (prompt_eval_count / (prompt_eval_dur_ns / 1e9)) if prompt_eval_dur_ns > 0 else 0.0
        eval_tok_sec = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

        ttft_sec = (
            round(first_token_time - start_wall_time, 3) if first_token_time else round(prompt_eval_dur_ns / 1e9, 3)
        )

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


class FoundryClient(BaseRuntimeClient):
    """
    Client for Microsoft Foundry Local / Foundry Server Runtime.
    Powered by ONNX Runtime GenAI with an OpenAI-compatible REST API.
    """

    name: str = "foundry"
    display_name: str = "MS Foundry"
    engine_name: str = "ONNX Runtime GenAI"

    def __init__(
        self,
        base_url: str = "http://localhost:5272/v1",
        timeout_sec: int = 180,
        auto_detect_port: bool = True,
        cli_path: str = "foundry",
        default_max_tokens: int = 4096,
    ):
        super().__init__(base_url=base_url, timeout_sec=timeout_sec)
        self.auto_detect_port = auto_detect_port
        self.cli_path = cli_path
        self.default_max_tokens = default_max_tokens
        self._endpoint_resolved = False
        self._engines: dict[str, str] = {}  # model id -> engine reported by the server (`owned_by`)
        if self.auto_detect_port:
            self._discover_endpoint()

    def _discover_endpoint(self) -> str | None:
        """Attempt to discover active Foundry Local server port via daemon.json or CLI."""
        # 1. Direct discovery file inspection (~/.foundry/daemon.json)
        daemon_json_path = os.path.expanduser("~/.foundry/daemon.json")
        if os.path.isfile(daemon_json_path):
            try:
                with open(daemon_json_path, encoding="utf-8") as f:
                    data = json.load(f)
                web_urls = data.get("web_urls", [])
                if web_urls:
                    resolved = f"{web_urls[0].rstrip('/')}/v1"
                    self.base_url = resolved
                    self._endpoint_resolved = True
                    return resolved
            except Exception:
                pass

        # 2. Fallback to CLI inspection
        foundry_bin = shutil.which(self.cli_path)
        if not foundry_bin:
            return None

        try:
            res = subprocess.run(
                [foundry_bin, "server", "status"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            output = f"{res.stdout} {res.stderr}"
            # Search for URL patterns like http://127.0.0.1:5272 or http://localhost:8080
            match = re.search(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)", output)
            if match:
                port = match.group(1)
                resolved = f"http://127.0.0.1:{port}/v1"
                self.base_url = resolved
                self._endpoint_resolved = True
                return resolved
        except Exception:
            pass
        return None

    def _get_api_endpoint(self, path: str) -> str:
        """Format REST API endpoint URL with correct /v1 prefix."""
        path = path.lstrip("/")
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/{path}"
        return f"{base}/v1/{path}"

    def is_reachable(self) -> bool:
        """Check if MS Foundry server REST API is responding."""
        urls_to_test = [self._get_api_endpoint("models")]
        if not self.base_url.endswith("/v1"):
            urls_to_test.append(f"{self.base_url.rstrip('/')}/models")

        for url in urls_to_test:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code in (200, 401, 403):
                    return True
            except Exception:
                pass

        # Try discovering port if not already resolved
        if self.auto_detect_port and not self._endpoint_resolved:
            new_url = self._discover_endpoint()
            if new_url:
                try:
                    r = requests.get(self._get_api_endpoint("models"), timeout=3)
                    if r.status_code in (200, 401, 403):
                        return True
                except Exception:
                    pass
        return False

    def get_version(self) -> str:
        """Query Foundry CLI version or server runtime."""
        foundry_bin = shutil.which(self.cli_path)
        if foundry_bin:
            try:
                res = subprocess.run(
                    [foundry_bin, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip()
            except Exception:
                pass
        return "Foundry Local (ONNX Runtime)"

    def list_installed_models(self) -> list[dict[str, Any]]:
        """List models available in Foundry Local catalog or active server."""
        models: list[dict[str, Any]] = []

        # 1. Query REST API /models endpoint
        try:
            r = requests.get(self._get_api_endpoint("models"), timeout=5)
            if r.status_code == 200:
                data = r.json()
                raw_models = data.get("data", []) if isinstance(data, dict) else data
                for m in raw_models:
                    model_id = m.get("id") or m.get("name", "")
                    if model_id:
                        owner = m.get("owned_by")
                        if isinstance(owner, str) and owner.strip():
                            self._engines[model_id] = owner.strip()
                        models.append(
                            {
                                "name": model_id,
                                "id": model_id,
                                "runtime": self.name,
                                "details": {
                                    "parameter_size": m.get("parameter_size", "ONNX"),
                                    "quantization_level": m.get("quantization", "ONNX"),
                                    "engine": self._engine_for(model_id),
                                },
                            }
                        )
                if models:
                    return models
        except Exception:
            pass

        # 2. Query CLI foundry cache list or foundry model list
        foundry_bin = shutil.which(self.cli_path)
        if foundry_bin:
            for subcmd in [["cache", "list"], ["model", "list"]]:
                try:
                    res = subprocess.run(
                        [foundry_bin] + subcmd,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if res.returncode == 0:
                        for line in res.stdout.splitlines():
                            line = line.strip()
                            if not line or line.startswith("NAME") or line.startswith("----") or "ALIAS" in line:
                                continue
                            parts = line.split()
                            if parts:
                                alias = parts[0]
                                if not any(m["name"] == alias for m in models):
                                    models.append(
                                        {
                                            "name": alias,
                                            "id": alias,
                                            "runtime": self.name,
                                            "details": {
                                                "parameter_size": "ONNX",
                                                "quantization_level": "ONNX",
                                                "engine": self.engine_name,
                                            },
                                        }
                                    )
                except Exception:
                    pass

        return models

    def _engine_for(self, model: str) -> str:
        """Engine that actually serves `model`: what the server reported, else inferred from an `ollama:` prefix.

        Servers such as Prism front several engines behind this one endpoint, so the class-level label is only a default.
        """
        reported = self._engines.get(model)
        if reported:
            return reported
        if model.lower().startswith("ollama:"):
            return OllamaClient.engine_name
        return self.engine_name

    def _clean_model_alias(self, model_name: str) -> str:
        """Strip hardware variant suffixes to resolve parent model alias."""
        alias = re.sub(
            r"-(?:generic-cpu|generic-gpu|cuda|directml|qnn|metal)(?::\d+)?$",
            "",
            model_name,
            flags=re.IGNORECASE,
        )
        alias = re.sub(r"-instruct$", "", alias, flags=re.IGNORECASE)
        return alias.lower()

    def _run_model_command(self, action: str, model_name: str, timeout: int) -> bool:
        """Run `foundry model <action> <alias>`, retrying with the cleaned parent alias."""
        foundry_bin = shutil.which(self.cli_path)
        if not foundry_bin:
            return False

        candidates = list(dict.fromkeys([model_name, self._clean_model_alias(model_name)]))
        for candidate in candidates:
            try:
                res = subprocess.run(
                    [foundry_bin, "model", action, candidate],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if res.returncode == 0:
                return True
        return False

    def _managed_by_foundry_cli(self) -> bool:
        """True only when the endpoint came from a discovered Foundry daemon.

        With an explicit `base_url` (auto_detect_port off) the server is something else, e.g. Prism, which loads models on
        first request; the `foundry` CLI would act on a different daemon, so it is not used.
        """
        return self._endpoint_resolved

    def load_model(self, model_name: str) -> bool:
        """Load a model into memory in the Foundry Local daemon.

        For explicit endpoints (lazy-loading servers) there is nothing to load: report whether the server lists the model.
        """
        if self._managed_by_foundry_cli():
            return self._run_model_command("load", model_name, timeout=180)
        return any(m["name"] == model_name for m in self.list_installed_models())

    def unload_model(self, model_name: str) -> bool:
        """Unload model via Foundry CLI to free accelerator memory (best effort; never fails a run)."""
        if self._managed_by_foundry_cli():
            self._run_model_command("unload", model_name, timeout=10)
        return True

    def pull_model(self, model_name: str, stream_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Download model via Foundry CLI (foundry model download <alias>)."""
        foundry_bin = shutil.which(self.cli_path)
        if not foundry_bin:
            if stream_callback:
                stream_callback({"status": "error", "error": "foundry CLI not found in PATH"})
            return False

        try:
            if stream_callback:
                stream_callback({"status": f"Downloading {model_name} via Foundry CLI..."})
            res = subprocess.run(
                [foundry_bin, "model", "download", model_name],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if res.returncode == 0:
                if stream_callback:
                    stream_callback({"status": "success"})
                return True
            else:
                if stream_callback:
                    stream_callback({"status": "error", "error": res.stderr.strip()})
                return False
        except Exception as e:
            if stream_callback:
                stream_callback({"status": "error", "error": str(e)})
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        measure_ttft: bool = True,
    ) -> dict[str, Any]:
        """
        Execute chat completion using MS Foundry Server OpenAI-compatible endpoint.
        Captures Time to First Token (TTFT) via Server-Sent Events streaming.
        Automatically loads model if not yet placed into memory by the daemon.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": measure_ttft,
        }
        if measure_ttft:
            payload["stream_options"] = {"include_usage": True}

        if options:
            if "temperature" in options:
                payload["temperature"] = float(options["temperature"])
            if "num_predict" in options:
                payload["max_tokens"] = int(options["num_predict"])
            elif "max_tokens" in options:
                payload["max_tokens"] = int(options["max_tokens"])
            if "top_p" in options:
                payload["top_p"] = float(options["top_p"])
            if "seed" in options:
                payload["seed"] = int(options["seed"])

        if "max_tokens" not in payload and self.default_max_tokens:
            payload["max_tokens"] = self.default_max_tokens

        start_wall_time = time.perf_counter()
        first_token_time: float | None = None
        collected_response: list[str] = []
        reported_usage: dict[str, Any] | None = None
        total_stream_chunks = 0

        endpoint = self._get_api_endpoint("chat/completions")

        def _send_request():
            if measure_ttft:
                return requests.post(
                    endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout_sec,
                )
            else:
                p = dict(payload)
                p["stream"] = False
                p.pop("stream_options", None)
                return requests.post(
                    endpoint,
                    json=p,
                    timeout=self.timeout_sec,
                )

        try:
            r = _send_request()
            if r.status_code == 400:
                err_text = r.text
                if "is not loaded" in err_text or "load the model" in err_text:
                    # Model not loaded in daemon memory - attempt auto-load and retry once
                    if self.load_model(model):
                        r = _send_request()

            r.raise_for_status()

            if measure_ttft:
                for line in r.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data:"):
                        continue
                    data_str = line_str[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    total_stream_chunks += 1

                    # Check usage in stream if returned
                    if "usage" in chunk and chunk["usage"]:
                        reported_usage = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_piece = delta.get("content", "")
                        if content_piece:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            collected_response.append(content_piece)
            else:
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    content_piece = choices[0].get("message", {}).get("content", "")
                    collected_response.append(content_piece)
                reported_usage = data.get("usage")

            end_wall_time = time.perf_counter()

        except Exception as e:
            return self._failure_result(model, e, start_wall_time)

        response_text = "".join(collected_response)

        # Token telemetry resolution
        if reported_usage:
            prompt_eval_count = reported_usage.get("prompt_tokens", 0)
            eval_count = reported_usage.get("completion_tokens", 0)
        else:
            # Heuristic calculation if server does not emit usage statistics
            words_prompt = len(prompt.split())
            prompt_eval_count = max(1, int(words_prompt * 1.3))
            if total_stream_chunks > 0:
                eval_count = total_stream_chunks
            else:
                words_eval = len(response_text.split())
                eval_count = max(1, int(words_eval * 1.3))

        # Precision durations
        total_time_sec = max(0.001, end_wall_time - start_wall_time)
        if first_token_time:
            ttft_sec = max(0.001, first_token_time - start_wall_time)
            eval_duration_sec = max(0.001, end_wall_time - first_token_time)
        else:
            ttft_sec = total_time_sec
            eval_duration_sec = total_time_sec

        eval_tok_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0.0
        prompt_tok_sec = prompt_eval_count / ttft_sec if ttft_sec > 0 else 0.0

        return {
            "success": True,
            "model": model,
            "runtime": self.name,
            "engine": self._engine_for(model),
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
                "stream_chunks": total_stream_chunks,
                "reported_usage": reported_usage,
            },
        }


def create_runtime_client(runtime_name: str, config: dict[str, Any]) -> BaseRuntimeClient:
    """Factory helper returning appropriate client instance for specified runtime."""
    runtime_name = runtime_name.lower().strip()
    if runtime_name in ("onnx-gpu", "onnx_gpu", "onnx-genai", "direct-onnx"):
        from benchrig.core.onnx_client import OnnxGenAiClient

        o_conf = config.get("onnx", {})
        models_dir = o_conf.get("models_dir")
        timeout = o_conf.get("timeout_sec", 300)
        max_tokens = o_conf.get("default_max_tokens", 4096)
        return OnnxGenAiClient(models_dir=models_dir, timeout_sec=timeout, default_max_tokens=max_tokens)

    if runtime_name in ("foundry", "ms-foundry", "ms_foundry", "onnx"):
        f_conf = config.get("foundry", {})
        base_url = f_conf.get("base_url", "http://localhost:5272/v1")
        timeout = f_conf.get("timeout_sec", 180)
        auto_detect = f_conf.get("auto_detect_port", True)
        cli = f_conf.get("cli_path", "foundry")
        max_tokens = f_conf.get("default_max_tokens", 4096)
        return FoundryClient(
            base_url=base_url,
            timeout_sec=timeout,
            auto_detect_port=auto_detect,
            cli_path=cli,
            default_max_tokens=max_tokens,
        )

    # Default to OllamaClient
    o_conf = config.get("ollama", {})
    base_url = o_conf.get("base_url", "http://localhost:11434")
    timeout = o_conf.get("timeout_sec", 180)
    num_ctx = o_conf.get("default_num_ctx", 4096)
    return OllamaClient(base_url=base_url, timeout_sec=timeout, default_num_ctx=num_ctx)
