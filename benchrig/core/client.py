"""Unified Local LLM API Clients supporting Ollama and Microsoft Foundry Server Runtime."""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

import requests

_log = logging.getLogger(__name__)

# Hard ceiling on Retry-After from prism-local: the server says "30", but we cap to keep a misconfigured
# server from blocking a benchmark run for an unbounded time. Override via env var if you really mean it.
_PRISM_RETRY_AFTER_CAP_SEC = float(os.environ.get("BENCHRIG_PRISM_RETRY_AFTER_CAP", "60"))
# Default max retries on 503 (initial attempt + retries). Prism server can stay busy for a while; 3 is a
# reasonable budget for a benchmark run that needs to finish in seconds-to-minutes.
_PRISM_503_MAX_RETRIES = 3

# Phase 10: when the eval wall-clock window is sub-millisecond (short answer, fast
# model), ``eval_duration_sec`` is floored to 0.001 to avoid divide-by-zero. The
# resulting ``eval_count / 0.001s`` ceiling is meaningful for ops but misleading as
# a "real" decode speed. Any raw window strictly less than this threshold is flagged
# so the Markdown leaderboard can render the value with a tilde prefix (~3000.0 t/s)
# and the CSV export can carry a boolean column.
EVAL_DURATION_FLOOR_THRESHOLD_SEC = 0.0015


class PrismBusyError(RuntimeError):
    """Raised after `PrismClient` exhausts its retry budget on 503 from prism-local.

    The error message includes the `error.code` and `error.message` from the JSON body, so the user
    sees whether the cause was `server_busy` (queue full) or `insufficient_resources` (load lock
    contention).
    """

    def __init__(self, reason: str, attempts: int, last_body: Any = None):
        self.reason = reason
        self.attempts = attempts
        self.last_body = last_body
        super().__init__(f"prism-local kept responding 503 ({attempts} attempts): {reason}")


def _logged_request(
    runtime: str,
    engine: str,
    url: str,
    *,
    method: str = "POST",
    **kwargs: Any,
) -> requests.Response:
    """Plain ``requests.post``/``requests.get`` with the JSON HTTP lifecycle instrumentation (Phase 11).

    Emits:
      - ``http.request_started``  (INFO,  carries method/model/runtime/engine/url/attempt)
      - ``http.response_completed`` (INFO, 2xx only, +status_code +duration_sec)
      - ``http.request_failed``  (WARNING, non-2xx or transport error, +error; +status_code for non-2xx)

    ``method`` is included in the ``http.request_started`` record so the JSON timeline can
    distinguish GETs from POSTs without parsing the URL. GET requests are health/version/list probes
    (``is_reachable`` and friends), whose failure is an expected answer rather than a fault, so their
    ``http.request_failed`` is logged at DEBUG: a down daemon does not print JSON at the default level.
    """
    log = logging.getLogger("benchrig")
    attempt = kwargs.pop("attempt", 1)
    model = kwargs.pop("model", None)
    log_failure = log.debug if method == "GET" else log.warning
    static_fields: dict[str, Any] = {
        "method": method,
        "model": model,
        "runtime": runtime,
        "engine": engine,
        "url": url,
        "attempt": attempt,
    }
    log.info(
        "http.request_started",
        extra={"event": "http.request_started", **static_fields},
    )
    started_at = time.perf_counter()
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        else:
            response = requests.post(url, **kwargs)
    except Exception as exc:
        log_failure(
            "http.request_failed",
            extra={"event": "http.request_failed", **static_fields, "error": str(exc)},
        )
        raise
    duration_sec = round(time.perf_counter() - started_at, 3)
    status = response.status_code
    if not isinstance(status, int) or 200 <= status < 300:  # real responses always carry an int status
        log.info(
            "http.response_completed",
            extra={
                "event": "http.response_completed",
                **static_fields,
                "status_code": response.status_code,
                "duration_sec": duration_sec,
            },
        )
    else:
        log_failure(
            "http.request_failed",
            extra={
                "event": "http.request_failed",
                **static_fields,
                "status_code": status,
                "duration_sec": duration_sec,
                "error": f"HTTP {status}",
            },
        )
    return response


def _logged_get(runtime: str, engine: str, url: str, **kwargs: Any) -> requests.Response:
    """Thin wrapper around ``_logged_request(..., method='GET')`` for readability at call sites."""
    return _logged_request(runtime, engine, url, method="GET", **kwargs)


def _post_with_503_retry(
    url: str,
    *,
    max_retries: int = _PRISM_503_MAX_RETRIES,
    retry_after_cap_sec: float = _PRISM_RETRY_AFTER_CAP_SEC,
    runtime: str = "prism",
    engine: str = "ONNX Runtime GenAI",
    **kwargs: Any,
) -> requests.Response:
    """``requests.post`` (through ``_logged_request``, so each attempt is logged) that retries on 503 with the server-provided ``Retry-After``.

    Other 4xx / 5xx responses are NOT retried: the caller gets them as-is and ``raise_for_status()``
    converts them to ``requests.HTTPError`` (or the caller handles the response).

    Phase 11: emits ``retry.attempted`` (INFO) per retry with attempt/delay_sec/reason, and
    ``retry.exhausted`` (WARNING) when the budget runs out with the JSON reason carried over.
    """
    attempts = 0
    while True:
        r = _logged_request(runtime, engine, url, attempt=attempts + 1, **kwargs)
        attempts += 1
        if r.status_code != 503:
            return r
        # Parse Retry-After. Header may be a delay in seconds (integer / float) or an HTTP-date.
        delay = 0.0
        raw = r.headers.get("Retry-After") if hasattr(r, "headers") else None
        if raw:
            try:
                delay = float(raw)
            except ValueError:
                # HTTP-date; ignore (we don't have a clock-comparison helper here, and the server
                # uses seconds in practice).
                delay = 0.0
        delay = min(delay, retry_after_cap_sec)

        # Decode the JSON reason from the response body so the log and the eventual
        # PrismBusyError surface the same field. If the body is missing or malformed, fall
        # back to "unknown".
        try:
            body = r.json()
        except Exception:
            body = None
        reason = "unknown"
        if isinstance(body, dict):
            err = body.get("error") if isinstance(body.get("error"), dict) else None
            if err:
                reason = f"{err.get('code', 'unknown')}: {err.get('message', '')}".strip(": ")

        if attempts > max_retries:
            # Last attempt was 503; surface the JSON reason and stop.
            _log.warning(
                "retry.exhausted",
                extra={
                    "event": "retry.exhausted",
                    "attempt": attempts,
                    "reason": reason,
                },
            )
            raise PrismBusyError(reason=reason, attempts=attempts, last_body=body)
        _log.info(
            "retry.attempted",
            extra={
                "event": "retry.attempted",
                "attempt": attempts,
                "delay_sec": delay,
                "reason": reason,
                "retry_after_header": raw,
            },
        )
        if delay > 0:
            time.sleep(delay)


class BaseRuntimeClient:
    """Abstract base class establishing a uniform contract across local inference engines."""

    name: str = "base"
    display_name: str = "Base Runtime"
    engine_name: str = "Unknown Engine"
    # Where `prompt_tok_per_sec` comes from: "server" (the engine reports its own prompt evaluation time) or
    # "client_ttft" (prompt tokens divided by the client-side time to first token, which includes request overhead).
    prefill_source: str = "client_ttft"
    # True if a request whose `num_ctx` differs from the loaded one makes the server reload the model (Ollama).
    reloads_on_context_change: bool = False

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

    def supports_thinking(self, model: str) -> bool:
        """Whether Ollama lists the `thinking` capability for `model` (cached; `think` is rejected for other models)."""
        if model not in self._thinking_support:
            try:
                r = self._make_request(
                    f"{self.base_url}/api/show",
                    json={"model": model},
                    timeout=5,
                    model=model,
                )
                capabilities = r.json().get("capabilities", []) if r.status_code == 200 else []
            except Exception:
                capabilities = []
            self._thinking_support[model] = "thinking" in capabilities
        return self._thinking_support[model]

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
    prefill_source = "server"
    reloads_on_context_change = True

    def _make_request(self, url: str, **kwargs: Any) -> requests.Response:
        """Plain ``requests.post`` with the JSON HTTP lifecycle instrumentation. Ollama has
        its own load queue on the server, so the 503-retry helper used by ``PrismClient`` is
        not needed here (Phase 11 follow-up to item 11.3 — same hook as ``FoundryClient``)."""
        return _logged_request(self.name, self.engine_name, url, **kwargs)

    def _make_get(self, url: str, **kwargs: Any) -> requests.Response:
        """Plain ``requests.get`` with the JSON HTTP lifecycle instrumentation (Phase 11 follow-up)."""
        return _logged_get(self.name, self.engine_name, url, **kwargs)

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_sec: int = 180,
        default_num_ctx: int = 4096,
    ):
        super().__init__(base_url=base_url, timeout_sec=timeout_sec)
        self.default_num_ctx = default_num_ctx
        self._thinking_support: dict[str, bool] = {}

    def is_reachable(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            r = self._make_get(f"{self.base_url}/api/version", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_version(self) -> str:
        """Get Ollama server version."""
        try:
            r = self._make_get(f"{self.base_url}/api/version", timeout=3)
            if r.status_code == 200:
                return r.json().get("version", "unknown")
        except Exception:
            pass
        return "unknown"

    def list_installed_models(self) -> list[dict[str, Any]]:
        """Return list of installed models with metadata."""
        try:
            r = self._make_get(f"{self.base_url}/api/tags", timeout=5)
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
            r = self._make_get(f"{self.base_url}/api/ps", timeout=5)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception:
            pass
        return []

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from VRAM/RAM immediately by setting keep_alive to 0."""
        try:
            r = self._make_request(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
                model=model_name,
            )
            return r.status_code == 200
        except Exception:
            return False

    def pull_model(self, model_name: str, stream_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Pull a model from Ollama library."""
        try:
            r = self._make_request(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=600,
                model=model_name,
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

        options = dict(options or {})
        think = options.pop("think", None)  # a top-level request field, not a model option
        merged_options = {"num_ctx": self.default_num_ctx, **options}
        payload["options"] = merged_options
        if think is not None and self.supports_thinking(model):
            payload["think"] = bool(think)

        start_wall_time = time.perf_counter()
        first_token_time: float | None = None  # first token of any kind: thinking or answer
        first_answer_time: float | None = None
        collected_response: list[str] = []
        thinking_chars = 0
        final_metrics: dict[str, Any] = {}

        try:
            if measure_ttft:
                # Streaming mode to accurately capture TTFT
                r = self._make_request(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=self.timeout_sec,
                    model=model,
                )
                r.raise_for_status()

                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    thinking = chunk.get("thinking", "")
                    text = chunk.get("response", "")
                    if thinking:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        thinking_chars += len(thinking)
                    if text:
                        now = time.perf_counter()
                        first_token_time = first_token_time or now
                        first_answer_time = first_answer_time or now
                        collected_response.append(text)

                    if chunk.get("done", False):
                        final_metrics = chunk
                        break
            else:
                # Non-streaming mode
                payload["stream"] = False
                r = self._make_request(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_sec,
                    model=model,
                )
                r.raise_for_status()
                final_metrics = r.json()
                collected_response.append(final_metrics.get("response", ""))
                thinking_chars = len(final_metrics.get("thinking") or "")

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
        eval_dur_sec = eval_dur_ns / 1e9 if eval_dur_ns > 0 else 0.0
        eval_tok_sec = (eval_count / eval_dur_sec) if eval_dur_sec > 0 else 0.0
        # Phase 10: Ollama reports its own eval_duration and applies no measurement floor, so a short window
        # is a measurement, never the floor artifact: the flag is always False here.
        eval_tok_sec_floored = False

        # Time to the first token of any kind, so a thinking model's latency is not its thinking time; the first *answer*
        # token is reported separately. With no token at all (non-streaming) the engine's prompt evaluation time stands in.
        ttft_sec = (
            round(first_token_time - start_wall_time, 3) if first_token_time else round(prompt_eval_dur_ns / 1e9, 3)
        )
        answer_ttft_sec = round(first_answer_time - start_wall_time, 3) if first_answer_time else None

        return {
            "success": True,
            "model": model,
            "runtime": self.name,
            "engine": self.engine_name,
            "response": response_text,
            "eval_count": eval_count,
            "eval_tok_per_sec": round(eval_tok_sec, 2),
            "eval_tok_sec_floored": eval_tok_sec_floored,  # Phase 10: tilde-prefix trigger
            "prompt_eval_count": prompt_eval_count,
            "prompt_tok_per_sec": round(prompt_tok_sec, 2),
            "finish_reason": final_metrics.get("done_reason"),
            "ttft_sec": ttft_sec,
            "answer_ttft_sec": answer_ttft_sec,
            "thinking_chars": thinking_chars,
            "think": payload.get("think"),
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
        api_key: str | None = None,
    ):
        super().__init__(base_url=base_url, timeout_sec=timeout_sec)
        self.api_key = api_key
        self.auto_detect_port = auto_detect_port
        self.cli_path = cli_path
        self.default_max_tokens = default_max_tokens
        self._endpoint_resolved = False
        self._engines: dict[str, str] = {}  # model id -> engine reported by the server (`owned_by`)
        if self.auto_detect_port:
            self._discover_endpoint()

    def _request_kwargs(self) -> dict[str, Any]:
        """Extra `requests` arguments: a bearer token when the server requires an API key."""
        return {"headers": {"Authorization": f"Bearer {self.api_key}"}} if self.api_key else {}

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
                r = requests.get(url, timeout=3, **self._request_kwargs())
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
            r = requests.get(self._get_api_endpoint("models"), timeout=5, **self._request_kwargs())
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

    def _make_request(self, url: str, **kwargs: Any) -> requests.Response:
        """Single POST hook. Default = instrumented plain ``requests.post`` (JSON lifecycle events).

        ``PrismClient`` overrides this to add ``_post_with_503_retry`` semantics on top of the
        same logging — see plan.md Phase 11 item 11.3 for the contract.
        """
        return _logged_request(self.name, self.engine_name, url, **kwargs)

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
        start_wall_time = time.perf_counter()
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

        try:
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
                if "top_k" in options:
                    payload["top_k"] = int(options["top_k"])
                if "repetition_penalty" in options:
                    payload["repetition_penalty"] = float(options["repetition_penalty"])
                if "stop" in options:
                    payload["stop"] = options["stop"]
        except (TypeError, ValueError) as e:
            # A malformed sampling option (e.g. options={"top_k": "many"}) must fail this one scenario, not
            # crash the whole --runs N benchmark with an uncaught exception from int()/float().
            return self._failure_result(model, e, start_wall_time)

        if "max_tokens" not in payload and self.default_max_tokens:
            payload["max_tokens"] = self.default_max_tokens

        first_token_time: float | None = None  # first token of any kind: reasoning or content
        first_answer_time: float | None = None  # first *content* (answer) token
        collected_response: list[str] = []
        collected_reasoning: list[str] = []
        reported_usage: dict[str, Any] | None = None
        reported_telemetry: dict[str, Any] | None = None  # e.g. Prism reports the device that ran the request
        finish_reason: str | None = None  # "stop" or "length" (token budget exhausted)
        total_stream_chunks = 0

        endpoint = self._get_api_endpoint("chat/completions")

        def _send_request():
            if measure_ttft:
                return self._make_request(
                    endpoint,
                    json=payload,
                    stream=True,
                    timeout=self.timeout_sec,
                    model=model,
                    **self._request_kwargs(),
                )
            else:
                p = dict(payload)
                p["stream"] = False
                p.pop("stream_options", None)
                return self._make_request(
                    endpoint,
                    json=p,
                    timeout=self.timeout_sec,
                    model=model,
                    **self._request_kwargs(),
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
                    if isinstance(chunk.get("telemetry"), dict):
                        reported_telemetry = chunk["telemetry"]

                    choices = chunk.get("choices", [])
                    if choices:
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                        delta = choices[0].get("delta", {})
                        reasoning_piece = delta.get("reasoning_content", "")
                        content_piece = delta.get("content", "")
                        if reasoning_piece:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            collected_reasoning.append(reasoning_piece)
                        if content_piece:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            first_answer_time = first_answer_time or now
                            collected_response.append(content_piece)
            else:
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    finish_reason = choices[0].get("finish_reason") or finish_reason
                    message = choices[0].get("message", {})
                    collected_response.append(message.get("content", ""))
                    reasoning_piece = message.get("reasoning_content", "")
                    if reasoning_piece:
                        collected_reasoning.append(reasoning_piece)
                reported_usage = data.get("usage")
                if isinstance(data.get("telemetry"), dict):
                    reported_telemetry = data["telemetry"]

            end_wall_time = time.perf_counter()

        except Exception as e:
            return self._failure_result(model, e, start_wall_time)

        response_text = "".join(collected_response)
        reasoning_text = "".join(collected_reasoning)

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
        total_time_sec_raw = max(0.0, end_wall_time - start_wall_time)
        if first_token_time:
            ttft_sec_raw = max(0.0, first_token_time - start_wall_time)
            eval_duration_sec_raw = max(0.0, end_wall_time - first_token_time)
        else:
            ttft_sec_raw = total_time_sec_raw
            eval_duration_sec_raw = total_time_sec_raw

        # Apply the 0.001 s floor only at the reporting boundary; ``*_raw`` is what the
        # floor-engagement detection compares against the threshold (Phase 10).
        total_time_sec = max(0.001, total_time_sec_raw)
        ttft_sec = max(0.001, ttft_sec_raw)
        eval_duration_sec = max(0.001, eval_duration_sec_raw)

        eval_tok_sec = eval_count / eval_duration_sec if eval_duration_sec > 0 else 0.0
        # Phase 10: a raw eval window under the threshold (including exactly 0.0, where ``max(0.001, ...)``
        # engages) with any token produced makes ``eval_tok_sec`` a floor artifact, not a measurement.
        eval_tok_sec_floored = eval_count > 0 and eval_duration_sec_raw < EVAL_DURATION_FLOOR_THRESHOLD_SEC
        prompt_tok_sec = prompt_eval_count / ttft_sec if ttft_sec > 0 else 0.0
        # Same floor as ttft_sec: a mocked/instant response can round both to 0.000, but answer_ttft_sec must
        # never be measurably earlier than ttft_sec (the answer token cannot arrive before the first token).
        answer_ttft_sec = round(max(ttft_sec, first_answer_time - start_wall_time), 3) if first_answer_time else None

        result = {
            "success": True,
            "model": model,
            "runtime": self.name,
            "engine": self._engine_for(model),
            "response": response_text,
            "finish_reason": finish_reason,
            "usage_estimated": not reported_usage,  # token counts below are guesses when the server sent no `usage`
            "eval_count": eval_count,
            "eval_tok_per_sec": round(eval_tok_sec, 2),
            "eval_tok_sec_floored": eval_tok_sec_floored,  # Phase 10: tilde-prefix trigger
            "prompt_eval_count": prompt_eval_count,
            "prompt_tok_per_sec": round(prompt_tok_sec, 2),
            "ttft_sec": round(ttft_sec, 3),
            "load_time_sec": 0.0,
            "total_time_sec": round(total_time_sec, 3),
            "raw_metrics": {
                "total_time_sec": total_time_sec,
                "stream_chunks": total_stream_chunks,
                "reported_usage": reported_usage,
                "telemetry": reported_telemetry,
            },
        }
        if reasoning_text:
            result["thinking_chars"] = len(reasoning_text)
            if answer_ttft_sec is not None:
                result["answer_ttft_sec"] = answer_ttft_sec
        return result


class PrismClient(FoundryClient):
    """
    Client for a Prism (`prism-local`) server: one OpenAI-compatible endpoint in front of ONNX Runtime GenAI (CUDA or CPU)
    and Ollama models. The endpoint is always explicit (default `http://127.0.0.1:5272/v1`); nothing is auto-discovered and
    the `foundry` CLI is never used.
    """

    name: str = "prism"
    display_name: str = "Prism"
    engine_name: str = "ONNX Runtime GenAI"
    DEFAULT_BASE_URL = "http://127.0.0.1:5272/v1"

    def _make_request(self, url: str, **kwargs: Any) -> requests.Response:
        # Prism returns 503 + Retry-After when its load lock is held or the queue is full
        # (prism-local HEAD, prism/server.py:669-671). Retry with the server-provided delay
        # (capped at 60 s), then give up with PrismBusyError. Other 4xx / 5xx are NOT retried.
        return _post_with_503_retry(url, runtime=self.name, engine=self.engine_name, **kwargs)

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: int = 180,
        default_max_tokens: int = 4096,
        api_key: str | None = None,
        cli_path: str = "prism",
    ):
        super().__init__(
            base_url=base_url,
            timeout_sec=timeout_sec,
            auto_detect_port=False,
            cli_path=cli_path,
            default_max_tokens=default_max_tokens,
            api_key=api_key or os.environ.get("PRISM_API_KEY"),
        )

    def _health(self) -> dict[str, Any]:
        """Prism's public `/health` document ({} if the server is unreachable or is not Prism)."""
        try:
            r = requests.get(self._get_api_endpoint("health"), timeout=3, **self._request_kwargs())
            if r.status_code == 200 and isinstance(r.json(), dict):
                return r.json()
        except Exception:
            pass
        return {}

    def get_running_models(self) -> list[dict[str, Any]]:
        """The model Prism currently holds (its `/health` `active_model`); Prism keeps one model loaded at a time."""
        active = self._health().get("active_model")
        return [{"name": active}] if active else []

    def unload_model(self, model_name: str) -> bool:
        """`POST /v1/unload` to free the resident ONNX model's VRAM/RAM (prism-local HEAD `6d6467a`, post-`v0.2.0`).

        Best-effort, like `FoundryClient.unload_model`: any transport error or non-2xx status (including a `404`
        from a prism-local build that predates this endpoint) is logged and swallowed, never raised, so it never
        fails a benchmark run.

        The server holds the engine lock for the call, so if a generation is still in flight this blocks until it
        finishes; the request can take seconds, not just round-trip latency.

        Phase 11: emits `unload.completed` at DEBUG with `ok` (bool) and `error` (str or absent).
        """
        error: str | None = None
        try:
            r = requests.post(self._get_api_endpoint("unload"), timeout=self.timeout_sec, **self._request_kwargs())
            r.raise_for_status()
        except Exception as exc:
            error = str(exc)
            _log.debug(
                "unload.completed",
                extra={
                    "event": "unload.completed",
                    "model": model_name,
                    "ok": False,
                    "error": error,
                },
            )
            return True
        _log.debug(
            "unload.completed",
            extra={"event": "unload.completed", "model": model_name, "ok": True},
        )
        return True

    def get_version(self) -> str:
        """Prism has no version endpoint; report what `/health` says about the accelerator."""
        health = self._health()
        if not health:
            return "Prism"
        device = health.get("active_device")
        return f"Prism (active device: {device})" if device else "Prism (idle)"

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        measure_ttft: bool = True,
    ) -> dict[str, Any]:
        """Generate, then record the device that actually ran the request (never assume GPU vs CPU).

        The device comes from the response's telemetry. For ONNX models served without telemetry (streaming) it falls back to
        `/health`, which reports the ONNX engine's provider; that is not used for Ollama-served models, where it could be
        stale state from an earlier ONNX model.
        """
        result = super().generate(model, prompt, system=system, options=options, measure_ttft=measure_ttft)
        if result.get("success"):
            telemetry = (result.get("raw_metrics") or {}).get("telemetry") or {}
            device = telemetry.get("device")
            engine = self._engine_for(model).lower()
            if not device and "llama" not in engine and "ollama" not in engine:
                device = self._health().get("active_device")
            if device:
                result["device"] = device
        return result

    def pull_model(self, model_name: str, stream_callback: Callable[[dict[str, Any]], None] | None = None) -> bool:
        """Download a model with `prism pull <model>` (needs the `prism` CLI on PATH)."""
        prism_bin = shutil.which(self.cli_path)
        if not prism_bin:
            if stream_callback:
                stream_callback({"status": "error", "error": "prism CLI not found in PATH (pip install prism-local)"})
            return False
        try:
            if stream_callback:
                stream_callback({"status": f"Downloading {model_name} via prism pull..."})
            res = subprocess.run([prism_bin, "pull", model_name], capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as e:
            if stream_callback:
                stream_callback({"status": "error", "error": str(e)})
            return False
        if stream_callback:
            stream_callback(
                {"status": "success"} if res.returncode == 0 else {"status": "error", "error": res.stderr.strip()}
            )
        return res.returncode == 0


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

    if runtime_name in ("prism", "prism-local", "prism_local"):
        p_conf = config.get("prism", {})
        return PrismClient(
            base_url=p_conf.get("base_url", PrismClient.DEFAULT_BASE_URL),
            timeout_sec=p_conf.get("timeout_sec", 180),
            default_max_tokens=p_conf.get("default_max_tokens", 4096),
            api_key=p_conf.get("api_key"),
            cli_path=p_conf.get("cli_path", "prism"),
        )

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
