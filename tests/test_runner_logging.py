"""Tests for capacity.exhausted log event (plan.md Phase 11, item 11.5)."""

import logging
import unittest
from unittest.mock import MagicMock, patch


class CapacityExhaustedLoggingTests(unittest.TestCase):
    def _capture_records(self, log):
        records: list[logging.LogRecord] = []
        handler = next(h for h in log.handlers if getattr(h, "name", "") == "benchrig.json")
        original_emit = handler.emit

        def cap(record):
            records.append(record)

        handler.emit = cap
        return records, lambda: setattr(handler, "emit", original_emit)

    def test_capacity_exhausted_logged_with_reason(self):
        """A scenario with `insufficient_resources` triggers `capacity.exhausted` at WARNING with model and reason."""
        from benchrig.core.client import PrismClient
        from benchrig.core.logging import setup_logging
        from benchrig.core.runner import BenchmarkRunner

        log = setup_logging("INFO", run_id="cap-test")
        records, restore = self._capture_records(log)
        try:
            mock_client = MagicMock(spec=PrismClient)
            mock_client.engine_name = "ONNX Runtime GenAI"
            mock_client.name = "prism"
            mock_client.display_name = "Prism"
            mock_client.unload_model.return_value = True
            mock_client.get_running_models.return_value = []

            runner = BenchmarkRunner(client=mock_client, config={"benchmark": {}})

            # Response carries a string `error` that contains the marker `insufficient_resources`
            # (the substring the runner's `_capacity_error` checks for).
            failing_response = {
                "success": False,
                "error": "insufficient_resources: Insufficient VRAM",
                "eval_count": 0,
                "prompt_eval_count": 0,
                "ttft_sec": 0.0,
                "eval_tok_per_sec": 0.0,
                "prompt_tok_per_sec": 0.0,
                "finish_reason": None,
                "truncated": False,
            }
            scenarios = [
                {
                    "id": "speed_short",
                    "name": "Short prompt",
                    "prompt": "Hi",
                    "max_tokens": 1,
                    "options": {},
                    "iterations": 1,
                }
            ]

            with patch("benchrig.core.runner.time.sleep", return_value=None):
                with patch.object(mock_client, "generate", return_value=failing_response):
                    runner._run_suite(
                        "speed",
                        "Speed Test",
                        "fixture-model",
                        scenarios,
                        lambda *_a, **_kw: {},
                    )
        finally:
            restore()

        cap_records = [r for r in records if getattr(r, "event", "") == "capacity.exhausted"]
        self.assertEqual(len(cap_records), 1)
        self.assertEqual(cap_records[0].levelno, logging.WARNING)
        self.assertEqual(cap_records[0].model, "fixture-model")
        self.assertIn("insufficient_resources", cap_records[0].reason)


if __name__ == "__main__":
    unittest.main()
