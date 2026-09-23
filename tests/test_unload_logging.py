"""Tests for unload.completed log event (plan.md Phase 11, item 11.6)."""

import logging
import unittest
from unittest.mock import MagicMock, patch

import requests


class UnloadCompletedLoggingTests(unittest.TestCase):
    def _capture_records(self, log):
        records: list[logging.LogRecord] = []
        handler = next(h for h in log.handlers if getattr(h, "name", "") == "benchrig.json")
        original_emit = handler.emit

        def cap(record):
            records.append(record)

        handler.emit = cap
        return records, lambda: setattr(handler, "emit", original_emit)

    def _client(self) -> MagicMock:
        """PrismClient stub bypassing __init__ (no real server required).

        `MagicMock(spec=PrismClient)` provides the attribute surface, but its methods are
        no-ops returning MagicMocks. We bind the *real* ``unload_model`` method to the mock
        instance so the call goes through the production code path under test.
        """
        from benchrig.core.client import PrismClient

        c = MagicMock(spec=PrismClient)
        c.name = "prism"
        c.engine_name = "ONNX Runtime GenAI"
        c.timeout_sec = 10
        c._get_api_endpoint.side_effect = lambda path: f"http://127.0.0.1:5272/v1/{path}"
        c._request_kwargs.return_value = {}
        # Bind the real method so the call exercises the actual logging code.
        c.unload_model = PrismClient.unload_model.__get__(c, PrismClient)
        return c

    def test_unload_ok_logs_debug_with_model(self):
        """A successful unload emits unload.completed at DEBUG with ok=true and the model name."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("DEBUG", run_id="unload-test")
        records, restore = self._capture_records(log)
        try:
            client = self._client()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None

            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                result = client.unload_model("phi-4")
        finally:
            restore()

        self.assertTrue(result)
        unload_records = [r for r in records if getattr(r, "event", "") == "unload.completed"]
        self.assertEqual(len(unload_records), 1)
        self.assertTrue(unload_records[0].ok)
        self.assertEqual(unload_records[0].model, "phi-4")
        self.assertFalse(hasattr(unload_records[0], "error"))

    def test_unload_failed_still_returns_true_but_logs_false_with_error(self):
        """A 404 from an older prism-local still returns True but logs unload.completed with ok=false, error=…."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("DEBUG", run_id="unload-test-2")
        records, restore = self._capture_records(log)
        try:
            client = self._client()
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 404
            # raise_for_status() raises HTTPError on non-2xx.
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")

            with patch("benchrig.core.client.requests.post", return_value=mock_response):
                result = client.unload_model("phi-4")
        finally:
            restore()

        self.assertTrue(result, "unload is best-effort; must return True on HTTP failure")
        unload_records = [r for r in records if getattr(r, "event", "") == "unload.completed"]
        self.assertEqual(len(unload_records), 1)
        self.assertFalse(unload_records[0].ok)
        self.assertEqual(unload_records[0].model, "phi-4")
        self.assertIn("404", unload_records[0].error)

    def test_unload_failure_is_logged_at_debug(self):
        """spec.md Phase 11: unload.completed is a DEBUG event, on failure too."""
        from benchrig.core.logging import setup_logging

        log = setup_logging("DEBUG", run_id="unload-level")
        records, restore = self._capture_records(log)
        try:
            client = self._client()
            with patch("benchrig.core.client.requests.post", side_effect=OSError("boom")):
                client.unload_model("phi-4")
        finally:
            restore()
        (rec,) = [r for r in records if getattr(r, "event", "") == "unload.completed"]
        self.assertEqual(rec.levelno, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
