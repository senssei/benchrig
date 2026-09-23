"""Tests for HTTP lifecycle log events on `_make_request` (plan.md Phase 11, item 11.3).

Covers the http.request_started / http.response_completed / http.request_failed events emitted
around every ``requests.post`` invoked via ``BaseRuntimeClient._make_request``.
"""

import json
import logging
import unittest
from unittest.mock import MagicMock, patch

import requests


def _capture_records(log):
    """Return (records, restore) pair capturing every record the JSON handler emits."""
    records: list[logging.LogRecord] = []
    handler = next(h for h in log.handlers if getattr(h, "name", "") == "benchrig.json")
    original_emit = handler.emit

    def cap(record):
        records.append(record)

    handler.emit = cap
    return records, lambda: setattr(handler, "emit", original_emit)


class MakeRequestLoggingTests(unittest.TestCase):
    """`FoundryClient._make_request` is the transport hook PrismClient inherits; instrumenting it
    covers every Prism/Foundry HTTP call. OllamaClient has direct `requests.post` (not via this
    hook) and is out of scope for item 11.3 (covered by item 11.6 / follow-up).
    """

    def _client(self) -> MagicMock:
        """A stub FoundryClient exposing the four fields the log events require."""
        from benchrig.core.client import FoundryClient

        c = MagicMock(spec=FoundryClient)
        c.name = "foundry"
        c.engine_name = "ONNX Runtime GenAI"
        return c

    def test_request_started_logged_on_post_with_attempt(self):
        """A successful 200 reply surfaces `http.request_started` at INFO with the required fields."""
        from benchrig.core.client import FoundryClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="http-test")
        captured = []
        client = self._client()
        client.name = "foundry"
        client.engine_name = "ONNX Runtime GenAI"

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200

        with patch.object(log, "info", side_effect=lambda msg, *a, **k: captured.append(k)):
            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                response = FoundryClient._make_request(  # noqa: SLF001
                    client,
                    "http://127.0.0.1:5272/v1/chat/completions",
                    json={"model": "phi-4"},
                    attempt=1,
                )

        self.assertIs(response, mock_response)
        started = [k for k in captured if k.get("extra", {}).get("event") == "http.request_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["extra"]["model"], None)
        self.assertEqual(started[0]["extra"]["runtime"], "foundry")
        self.assertEqual(started[0]["extra"]["engine"], "ONNX Runtime GenAI")
        self.assertEqual(started[0]["extra"]["url"], "http://127.0.0.1:5272/v1/chat/completions")
        self.assertEqual(started[0]["extra"]["attempt"], 1)

    def test_response_completed_carries_status_code_and_duration(self):
        """A successful POST logs http.response_completed with status_code and duration_sec."""
        from benchrig.core.client import FoundryClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="http-test-2")
        captured = []
        client = self._client()

        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200

        with patch.object(log, "info", side_effect=lambda msg, *a, **k: captured.append(k)):
            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                FoundryClient._make_request(client, "http://x", json={})  # noqa: SLF001

        completed = [k for k in captured if k.get("extra", {}).get("event") == "http.response_completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["extra"]["status_code"], 200)
        self.assertGreaterEqual(completed[0]["extra"]["duration_sec"], 0.0)

    def test_request_failed_on_transport_error_logs_warning(self):
        """A connection error logs http.request_failed at WARNING with the error string."""
        from benchrig.core.client import FoundryClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="http-test-3")
        captured = []
        client = self._client()

        with (
            patch.object(log, "info", side_effect=lambda msg, *a, **k: captured.append(k)),
            patch.object(log, "warning", side_effect=lambda msg, *a, **k: captured.append(k)),
            patch(
                "benchrig.core.client.requests.post",
                side_effect=requests.exceptions.ConnectionError("boom"),
            ),
        ):
            with self.assertRaises(requests.exceptions.ConnectionError):
                FoundryClient._make_request(client, "http://x", json={})  # noqa: SLF001

        failed = [k for k in captured if k.get("extra", {}).get("event") == "http.request_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("boom", failed[0]["extra"]["error"])

    def test_warning_level_suppresses_http_info_events(self):
        """At default WARNING, http.lifecycle events (INFO) do not reach the JSON handler."""
        from benchrig.core.client import FoundryClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("WARNING", run_id="http-test-4")
        captured_records: list[logging.LogRecord] = []
        handler = next(h for h in log.handlers if getattr(h, "name", "") == "benchrig.json")
        original_emit = handler.emit

        def capture_emit(record):
            captured_records.append(record)

        handler.emit = capture_emit
        try:
            client = self._client()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                FoundryClient._make_request(client, "http://x", json={})  # noqa: SLF001
        finally:
            handler.emit = original_emit

        http_records = [r for r in captured_records if getattr(r, "event", "").startswith("http.")]
        self.assertEqual(
            http_records,
            [],
            f"no http.* events should reach the handler at WARNING; got: {[r.event for r in captured_records]}",
        )


class OllamaClientLoggingTests(unittest.TestCase):
    """`OllamaClient` does not override `_make_request`/`_make_get`, so plumbing the
    instrumented hook for both also covers the local Ollama path (Phase 11 follow-up).
    `_make_ollama()` builds a real `OllamaClient` instance (via ``object.__new__``, no
    constructor) and binds every method the tests under it exercise to the *real*
    implementation on the class, so the production code paths run.
    """

    def _stream_response(self, payload_chunks):
        """Build a mock response that yields one chunk per iteration over lines."""
        from unittest.mock import MagicMock

        mock = MagicMock(spec=requests.Response)
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        mock.iter_lines.return_value = [json.dumps(c).encode() for c in payload_chunks]
        mock.json.return_value = payload_chunks[-1]
        return mock

    def _make_ollama(self):
        """Build a real ``OllamaClient`` (no ``__init__``) and bind the methods under test."""
        from benchrig.core.client import OllamaClient

        client = object.__new__(OllamaClient)
        client.base_url = "http://127.0.0.1:11434"
        client.default_num_ctx = 4096
        client.timeout_sec = 30
        client._thinking_support = {}
        # Bind the instrumented hooks (now in the class) and the methods used in tests.
        client._make_request = OllamaClient._make_request.__get__(client, OllamaClient)
        client._make_get = OllamaClient._make_get.__get__(client, OllamaClient)
        client.unload_model = OllamaClient.unload_model.__get__(client, OllamaClient)
        client.supports_thinking = OllamaClient.supports_thinking.__get__(client, OllamaClient)
        client.is_reachable = OllamaClient.is_reachable.__get__(client, OllamaClient)
        client.get_version = OllamaClient.get_version.__get__(client, OllamaClient)
        client.list_installed_models = OllamaClient.list_installed_models.__get__(client, OllamaClient)
        client.get_running_models = OllamaClient.get_running_models.__get__(client, OllamaClient)
        client.pull_model = OllamaClient.pull_model.__get__(client, OllamaClient)
        return client

    def test_ollama_generate_emits_http_request_started_and_completed(self):
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="ollama-http-test")
        records, restore = _capture_records(log)
        try:
            client = self._make_ollama()
            stream_mock = self._stream_response([{"done": True, "response": "ok"}])
            with patch("benchrig.core.client.requests.post", return_value=stream_mock):
                import itertools

                with patch(
                    "benchrig.core.client.time.perf_counter",
                    side_effect=itertools.count(0.0, 0.01),
                ):
                    client.generate("phi4", "Hi.")

            http_records = [r for r in records if getattr(r, "event", "").startswith("http.")]
            events = [r.event for r in http_records]
            self.assertIn("http.request_started", events)
            self.assertIn("http.response_completed", events)
            started = next(r for r in http_records if r.event == "http.request_started")
            self.assertEqual(started.model, "phi4")
            self.assertEqual(started.runtime, "ollama")
            self.assertEqual(started.engine, "llama.cpp")
            self.assertEqual(started.url, "http://127.0.0.1:11434/api/generate")
        finally:
            restore()

    def test_ollama_supports_thinking_emits_http_request_started_via_api_show(self):
        """`supports_thinking()` POSTs /api/show — must emit http.request_started with the model field."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="ollama-think-test")
        records, restore = _capture_records(log)
        try:
            client = self._make_ollama()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {"capabilities": ["completion"]}

            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                client.supports_thinking("phi4-mini:latest")

            http_records = [r for r in records if getattr(r, "event", "").startswith("http.")]
            started = next(r for r in http_records if r.event == "http.request_started")
            self.assertEqual(started.model, "phi4-mini:latest")
            self.assertEqual(started.url, "http://127.0.0.1:11434/api/show")
        finally:
            restore()

    def test_ollama_pull_model_emits_http_request_started_via_api_pull(self):
        """`pull_model()` POSTs /api/pull (streaming) — must emit http.request_started."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="ollama-pull-test")
        records, restore = _capture_records(log)
        try:
            client = self._make_ollama()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [b'{"status": "success"}']

            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                client.pull_model("phi4-mini:latest")

            http_records = [r for r in records if getattr(r, "event", "").startswith("http.")]
            started = next(r for r in http_records if r.event == "http.request_started")
            self.assertEqual(started.model, "phi4-mini:latest")
            self.assertEqual(started.url, "http://127.0.0.1:11434/api/pull")
        finally:
            restore()

    def test_ollama_is_reachable_emits_http_request_started_via_api_version(self):
        """`is_reachable()` GETs /api/version — must emit http.request_started (GET path)."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="ollama-reach-test")
        records, restore = _capture_records(log)
        try:
            client = self._make_ollama()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200

            with patch("benchrig.core.client.requests.get", return_value=mock_response):
                client.is_reachable()

            http_records = [r for r in records if getattr(r, "event", "").startswith("http.")]
            started = next(r for r in http_records if r.event == "http.request_started")
            self.assertEqual(started.url, "http://127.0.0.1:11434/api/version")
        finally:
            restore()

    def test_ollama_unload_emits_http_request_started(self):
        """`OllamaClient.unload_model` POSTs /api/generate with keep_alive=0 — must emit http.request_started."""
        from benchrig.core.client import OllamaClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="ollama-unload-test")
        records, restore = _capture_records(log)
        try:
            client = MagicMock(spec=OllamaClient)
            client.name = "ollama"
            client.engine_name = "llama.cpp"
            client.base_url = "http://127.0.0.1:11434"
            client.timeout_sec = 30

            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {}

            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                client._make_request = OllamaClient._make_request.__get__(client, OllamaClient)
                client.unload_model = OllamaClient.unload_model.__get__(client, OllamaClient)
                client.unload_model("phi4")

            http_records = [r for r in records if getattr(r, "event", "").startswith("http.")]
            self.assertTrue(any(r.event == "http.request_started" for r in http_records))
            started = next(r for r in http_records if r.event == "http.request_started")
            self.assertEqual(started.model, "phi4")
            self.assertEqual(started.url, "http://127.0.0.1:11434/api/generate")
        finally:
            restore()


class ReviewFixTests(unittest.TestCase):
    """Phase 11 review fixes (2026-09-23): Prism coverage, non-2xx shape, model field, probe noise."""

    def _prism(self) -> MagicMock:
        from benchrig.core.client import PrismClient

        c = MagicMock(spec=PrismClient)
        c.name = "prism"
        c.engine_name = "ONNX Runtime GenAI"
        return c

    def _response(self, status: int) -> MagicMock:
        r = MagicMock(spec=requests.Response)
        r.status_code = status
        r.headers = {}
        r.json.return_value = {"error": {"code": "server_busy", "message": "queue full"}}
        return r

    def test_prism_make_request_emits_http_events(self):
        """PrismClient overrides _make_request; its HTTP calls must be logged like every other runtime's."""
        from benchrig.core.client import PrismClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="prism-http")
        records, restore = _capture_records(log)
        try:
            with patch("benchrig.core.client.requests.post", return_value=self._response(200)):
                PrismClient._make_request(self._prism(), "http://127.0.0.1:5272/v1/chat/completions")
        finally:
            restore()
        events = [(r.event, r.runtime) for r in records if getattr(r, "event", "").startswith("http.")]
        self.assertEqual(events, [("http.request_started", "prism"), ("http.response_completed", "prism")])

    def test_non_2xx_is_request_failed_not_response_completed(self):
        """spec.md Phase 11: response_completed is 2xx only; a 500 is http.request_failed at WARNING."""
        from benchrig.core.client import FoundryClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="non-2xx")
        records, restore = _capture_records(log)
        try:
            client = MagicMock(spec=FoundryClient)
            client.name, client.engine_name = "foundry", "ONNX Runtime GenAI"
            with patch("benchrig.core.client.requests.post", return_value=self._response(500)):
                FoundryClient._make_request(client, "http://127.0.0.1:5272/v1/chat/completions")
        finally:
            restore()
        events = [r.event for r in records if getattr(r, "event", "").startswith("http.")]
        self.assertEqual(events, ["http.request_started", "http.request_failed"])
        failed = records[-1]
        self.assertEqual((failed.levelno, failed.status_code), (logging.WARNING, 500))

    def test_prism_retry_attempts_are_numbered_and_exhaustion_is_request_failed(self):
        """Every 503 attempt is http.request_failed with its attempt number; the budget running out still raises."""
        from benchrig.core.client import PrismBusyError, PrismClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="prism-503")
        records, restore = _capture_records(log)
        try:
            with patch("benchrig.core.client.requests.post", return_value=self._response(503)):
                with patch("benchrig.core.client.time.sleep"):
                    with self.assertRaises(PrismBusyError):
                        PrismClient._make_request(self._prism(), "http://127.0.0.1:5272/v1/chat/completions")
        finally:
            restore()
        failed = [r for r in records if getattr(r, "event", "") == "http.request_failed"]
        self.assertEqual([r.attempt for r in failed], [1, 2, 3, 4])
        self.assertEqual({r.status_code for r in failed}, {503})

    def test_foundry_generate_logs_the_model(self):
        """Foundry/Prism generate() passes the model so http.* records do not say model=null."""
        from benchrig.core.client import PrismClient
        from benchrig.core.logging import setup_logging

        client = object.__new__(PrismClient)
        client.base_url = "http://127.0.0.1:5272/v1"
        client.name, client.engine_name = "prism", "ONNX Runtime GenAI"
        client.timeout_sec, client.default_num_ctx, client.default_max_tokens = 30, 4096, None
        client.api_key, client._engines = None, {}
        client.unload_model = lambda *_a, **_k: True
        stream = MagicMock()
        stream.status_code = 200
        stream.iter_lines.return_value = [b"data: [DONE]\n\n"]

        log = setup_logging("INFO", run_id="model-field")
        records, restore = _capture_records(log)
        try:
            with patch("benchrig.core.client.requests.post", return_value=stream):
                client.generate("phi4", "hi")
        finally:
            restore()
        started = [r for r in records if getattr(r, "event", "") == "http.request_started"]
        self.assertTrue(started)
        self.assertEqual({r.model for r in started}, {"phi4"})

    def test_failed_probe_is_silent_at_the_default_warning_level(self):
        """`benchrig --check` with the daemon down must not print JSON on stderr at WARNING (GET probes)."""
        from benchrig.core.client import OllamaClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("WARNING", run_id="probe-quiet")
        records, restore = _capture_records(log)
        try:
            with patch("benchrig.core.client.requests.get", side_effect=requests.ConnectionError("refused")):
                self.assertFalse(OllamaClient(base_url="http://127.0.0.1:1").is_reachable())
        finally:
            restore()
        self.assertEqual(records, [])

    def test_failed_probe_is_visible_at_debug(self):
        from benchrig.core.client import OllamaClient
        from benchrig.core.logging import setup_logging

        log = setup_logging("DEBUG", run_id="probe-debug")
        records, restore = _capture_records(log)
        try:
            with patch("benchrig.core.client.requests.get", side_effect=requests.ConnectionError("refused")):
                OllamaClient(base_url="http://127.0.0.1:1").is_reachable()
        finally:
            restore()
        failed = [r for r in records if getattr(r, "event", "") == "http.request_failed"]
        self.assertEqual([r.levelno for r in failed], [logging.DEBUG])


if __name__ == "__main__":
    unittest.main()
