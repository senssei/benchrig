"""Tests for retry lifecycle log events on `_post_with_503_retry` (plan.md Phase 11, item 11.4).

Covers `retry.attempted` (per retry) and `retry.exhausted` (when the budget runs out).
"""

import logging
import unittest
from unittest.mock import MagicMock, patch

import requests


class RetryEventsTests(unittest.TestCase):
    def _capture_records(self, log):
        """Return (records, restore) pair that captures every record the JSON handler emits."""
        records: list[logging.LogRecord] = []
        handler = next(h for h in log.handlers if getattr(h, "name", "") == "benchrig.json")
        original_emit = handler.emit

        def cap(record):
            records.append(record)

        handler.emit = cap
        return records, lambda: setattr(handler, "emit", original_emit)

    def test_retry_attempted_logged_with_delay_and_reason(self):
        """A 503 + Retry-After triggers `retry.attempted` at INFO with attempt, delay_sec, reason (from JSON body)."""
        from benchrig.core.client import _post_with_503_retry
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="retry-test")
        records, restore = self._capture_records(log)
        try:
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 503
            mock_response.headers = {"Retry-After": "1"}
            mock_response.json.return_value = {"error": {"code": "server_busy", "message": "queue full"}}

            ok_response = MagicMock(spec=requests.Response)
            ok_response.status_code = 200
            ok_response.headers = {}

            with patch("time.sleep"):
                with patch(
                    "benchrig.core.client.requests.post",
                    side_effect=[mock_response, ok_response],
                ):
                    _post_with_503_retry("http://x", json={}, max_retries=3)
        finally:
            restore()

        retries = [r for r in records if getattr(r, "event", "") == "retry.attempted"]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0].attempt, 1)
        self.assertEqual(retries[0].delay_sec, 1.0)
        self.assertEqual(retries[0].reason, "server_busy: queue full")

    def test_retry_exhausted_logs_warning_with_reason(self):
        """4 consecutive 503s raise PrismBusyError and emit `retry.exhausted` at WARNING."""
        import logging

        from benchrig.core.client import PrismBusyError, _post_with_503_retry
        from benchrig.core.logging import setup_logging

        log = setup_logging("INFO", run_id="retry-test-2")
        records, restore = self._capture_records(log)
        try:

            def make_503():
                r = MagicMock(spec=requests.Response)
                r.status_code = 503
                r.headers = {"Retry-After": "0.5"}
                r.json.return_value = {"error": {"code": "insufficient_resources", "message": "VRAM exhausted"}}
                return r

            with patch("time.sleep"):
                with patch(
                    "benchrig.core.client.requests.post",
                    side_effect=[make_503() for _ in range(4)],
                ):
                    with self.assertRaises(PrismBusyError) as ctx:
                        _post_with_503_retry("http://x", json={}, max_retries=3)
        finally:
            restore()

        self.assertEqual(ctx.exception.attempts, 4)
        exhausted = [r for r in records if getattr(r, "event", "") == "retry.exhausted"]
        self.assertEqual(len(exhausted), 1)
        self.assertEqual(exhausted[0].levelno, logging.WARNING)
        self.assertEqual(exhausted[0].attempt, 4)
        self.assertIn("insufficient_resources", exhausted[0].reason)


if __name__ == "__main__":
    unittest.main()
