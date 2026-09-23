"""Tests for the structured logging module (plan.md Phase 11, item 11.1).

Run order matters: the contextvar-based run_id tests rely on each test resetting
its token cleanly; ``unittest`` runs tests in declaration order, so test isolation
within this file is preserved as long as each test that calls ``RUN_ID.set(...)``
calls ``RUN_ID.reset(token)`` in ``finally``.
"""

import io
import json
import logging
import unittest


class JsonFormatterShapeTests(unittest.TestCase):
    def test_record_has_no_taskname_noise(self):
        """Python 3.12 adds LogRecord.taskName; it is not a per-event extra and must not leak into the JSON."""
        from benchrig.core.logging import JsonFormatter

        record = logging.LogRecord("benchrig.t", logging.INFO, __file__, 1, "m", None, None)
        record.taskName = None  # what 3.12+ sets on every record
        self.assertNotIn("taskName", json.loads(JsonFormatter().format(record)))


class JsonFormatterTests(unittest.TestCase):
    def test_json_formatter_emits_required_fields(self):
        """A record goes through JsonFormatter -> one JSON object with ts/level/event/message."""
        from benchrig.core.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="benchrig.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        record.event = "test.greeting"
        output = formatter.format(record)
        obj = json.loads(output)
        self.assertEqual(obj["level"], "INFO")
        self.assertEqual(obj["event"], "test.greeting")
        self.assertEqual(obj["message"], "hello world")
        # ISO 8601 UTC with millisecond precision, ending in Z.
        self.assertRegex(obj["ts"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

    def test_json_formatter_includes_per_event_extras(self):
        """Custom attributes set on the record (e.g. `model`, `attempt`) land in the JSON output verbatim."""
        from benchrig.core.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="benchrig.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="x",
            args=(),
            exc_info=None,
        )
        record.event = "http.request_started"
        record.model = "phi-4-mini"
        record.attempt = 2
        obj = json.loads(formatter.format(record))
        self.assertEqual(obj["model"], "phi-4-mini")
        self.assertEqual(obj["attempt"], 2)
        self.assertEqual(obj["event"], "http.request_started")

    def test_json_formatter_falls_back_to_record_name_when_event_unset(self):
        """When no explicit `event` attribute is on the record, the formatter uses the logger name."""
        from benchrig.core.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="benchrig.core.client",
            level=logging.WARNING,
            pathname=__file__,
            lineno=10,
            msg="x",
            args=(),
            exc_info=None,
        )
        obj = json.loads(formatter.format(record))
        self.assertEqual(obj["event"], "benchrig.core.client")


class RunIdFilterTests(unittest.TestCase):
    def test_run_id_filter_injects_contextvar_value(self):
        """When the RUN_ID contextvar is set, RunIdFilter stamps it onto the record."""
        from benchrig.core.logging import RUN_ID, RunIdFilter

        token = RUN_ID.set("run-abc-123")
        try:
            flt = RunIdFilter()
            record = logging.LogRecord(
                name="benchrig.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=10,
                msg="x",
                args=(),
                exc_info=None,
            )
            self.assertTrue(flt.filter(record))
            self.assertEqual(record.run_id, "run-abc-123")
        finally:
            RUN_ID.reset(token)

    def test_run_id_filter_omits_field_when_contextvar_unset(self):
        """When RUN_ID is at default (None), the filter leaves the record alone (no `run_id` key)."""
        from benchrig.core.logging import RUN_ID, RunIdFilter

        flt = RunIdFilter()
        record = logging.LogRecord(
            name="benchrig.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="x",
            args=(),
            exc_info=None,
        )
        # Clear any prior set from earlier tests (defensive in test ordering).
        RUN_ID.set(None)
        try:
            self.assertTrue(flt.filter(record))
            self.assertFalse(
                hasattr(record, "run_id") and getattr(record, "run_id", None) is not None,
                "filter should not add a `run_id` attribute when the contextvar is unset",
            )
        finally:
            RUN_ID.set(None)  # reset to default explicitly


class SetupLoggingTests(unittest.TestCase):
    def test_setup_logging_returns_benchrig_logger(self):
        """setup_logging returns the same logger across calls (named 'benchrig')."""
        from benchrig.core.logging import setup_logging

        logger = setup_logging("INFO")
        self.assertEqual(logger.name, "benchrig")

    def test_setup_logging_replaces_existing_stderr_handler(self):
        """Idempotent: re-calling setup_logging does not duplicate the JSON handler."""
        from benchrig.core.logging import _JSON_HANDLER_NAME, setup_logging

        logger1 = setup_logging("WARNING")
        count1 = sum(1 for h in logger1.handlers if getattr(h, "name", "") == _JSON_HANDLER_NAME)
        logger2 = setup_logging("WARNING")
        count2 = sum(1 for h in logger2.handlers if getattr(h, "name", "") == _JSON_HANDLER_NAME)
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 1)

    def test_setup_logging_level_sets_logger_level(self):
        """setup_logging(level='DEBUG') sets logger.level to logging.DEBUG."""
        from benchrig.core.logging import setup_logging

        logger = setup_logging("DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)


class EndToEndJsonLineTests(unittest.TestCase):
    """Smoke test: a record emitted through setup_logging surfaces as one parseable JSON line on stderr."""

    def test_record_through_setup_logging_is_parseable_json_with_run_id(self):
        from benchrig.core.logging import _JSON_HANDLER_NAME, setup_logging

        # Capture the actual stderr handler's stream; replace with StringIO; restore after.
        logger = setup_logging("INFO", run_id="e2e-run-001")
        handler = next(h for h in logger.handlers if getattr(h, "name", "") == _JSON_HANDLER_NAME)
        original = handler.stream
        buf = io.StringIO()
        handler.stream = buf
        try:
            logger.info("hello %s", "world", extra={"event": "smoke.test", "model": "fixture"})
        finally:
            handler.stream = original

        line = buf.getvalue().strip()
        obj = json.loads(line)
        self.assertEqual(obj["event"], "smoke.test")
        self.assertEqual(obj["message"], "hello world")
        self.assertEqual(obj["run_id"], "e2e-run-001")
        self.assertEqual(obj["model"], "fixture")
        self.assertEqual(obj["level"], "INFO")


if __name__ == "__main__":
    unittest.main()
