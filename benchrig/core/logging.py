"""Structured logging on stderr (plan.md Phase 11, item 11.1).

JSON-shaped records, one per line, with stable field names so downstream log aggregators can index
them. Stdlib ``logging`` only - no OpenTelemetry, no file output. Stderr is the destination; stdout stays
for the rich UX that ``rich.console.print`` provides in ``benchrig/cli.py``.

Stable record shape::

    {"ts": "<ISO 8601 UTC, ms precision>",
     "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>",
     "event": "<dotted name>",
     "run_id": "<uuid4 or absent>",
     "message": "<formatted>",
     ...per-event extras (model, runtime, attempt, duration_sec, ttft_sec, ...) ...}
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# A contextvar so ``logging.getLogger("benchrig").info(...)`` calls inherit ``run_id`` from the calling
# context (set by ``cli.py::main`` at startup). Tools that do not set it see ``None``.
RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("benchrig_run_id", default=None)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record. Output is a single line (no trailing newline)."""

    # Standard ``logging.LogRecord`` attributes; everything else is treated as a per-event extra.
    _STD_ATTRS = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "taskName",  # added by Python 3.12; not a per-event extra
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")
        # ``datetime.isoformat`` on a UTC datetime emits ``+00:00``. The plan-defined JSON shape
        # uses the canonical ``Z`` so downstream JSON parsers (jq, log aggregators) recognise
        # the timestamp as UTC at a glance.
        if ts.endswith("+00:00"):
            ts = ts[:-6] + "Z"
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "event": getattr(record, "event", record.name),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._STD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


class RunIdFilter(logging.Filter):
    """Inject ``run_id`` from the ``RUN_ID`` contextvar onto every record that passes through."""

    def filter(self, record: logging.LogRecord) -> bool:
        rid = RUN_ID.get()
        if rid is not None:
            record.run_id = rid
        return True


# Sentinel name so callers can locate the JSON handler on the benchrig logger (used by setup_logging
# for idempotency and by tests). Always a constant - never per-instance.
_JSON_HANDLER_NAME = "benchrig.json"


def setup_logging(level: str = "WARNING", run_id: str | None = None) -> logging.Logger:
    """Configure the ``benchrig`` logger with a JSON stderr handler. Idempotent.

    Args:
        level: log level name (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL``).
            Default ``WARNING`` so the colored stdout UX is unchanged.
        run_id: optional UUID4 set on the ``RUN_ID`` contextvar for this run. Pass it once at
            startup; per-record filters will pick it up automatically.

    Returns:
        The ``benchrig`` logger.
    """
    logger = logging.getLogger("benchrig")

    # Reuse an existing JSON handler if present, otherwise create one. Never duplicate.
    existing = [h for h in logger.handlers if getattr(h, "name", "") == _JSON_HANDLER_NAME]
    if existing:
        handler = existing[0]
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.name = _JSON_HANDLER_NAME
        handler.setFormatter(JsonFormatter())
        handler.addFilter(RunIdFilter())
        logger.addHandler(handler)

    numeric_level = logging.getLevelName(level.upper())
    logger.setLevel(numeric_level)

    if run_id is not None:
        RUN_ID.set(run_id)

    return logger
