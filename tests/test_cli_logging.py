"""Tests for the CLI surface of structured logging (plan.md Phase 11, item 11.2).

Covers the ``--log-level`` flag, the ``BENCHRIG_LOG_LEVEL`` env var, the invalid-level error path,
and the run.started/run.completed JSON events emitted by ``run_benchmarks``.
"""

import os
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch


class LogLevelFlagTests(unittest.TestCase):
    def test_log_level_flag_default_is_warning(self):
        """`--log-level` absent and no env var: default WARNING (silent stderr)."""
        from benchrig.cli import build_parser

        saved = os.environ.pop("BENCHRIG_LOG_LEVEL", None)
        try:
            parser = build_parser()
            args = parser.parse_args([])
            self.assertEqual(args.log_level, "WARNING")
        finally:
            if saved is not None:
                os.environ["BENCHRIG_LOG_LEVEL"] = saved

    def test_log_level_flag_accepts_debug(self):
        from benchrig.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--log-level", "DEBUG"])
        self.assertEqual(args.log_level, "DEBUG")

    def test_env_var_overrides_flag_default(self):
        """BENCHRIG_LOG_LEVEL=DEBUG without --log-level: resolved to DEBUG."""
        from benchrig.cli import build_parser

        with patch.dict(os.environ, {"BENCHRIG_LOG_LEVEL": "DEBUG"}, clear=False):
            parser = build_parser()
            args = parser.parse_args([])
            self.assertEqual(args.log_level, "DEBUG")

    def test_flag_overrides_env_var(self):
        """If both are set, --log-level wins (env only seeds the default)."""
        from benchrig.cli import build_parser

        with patch.dict(os.environ, {"BENCHRIG_LOG_LEVEL": "DEBUG"}, clear=False):
            parser = build_parser()
            args = parser.parse_args(["--log-level", "ERROR"])
            self.assertEqual(args.log_level, "ERROR")

    def test_log_level_flag_is_case_insensitive(self):
        """The env var accepts `debug`; the flag must too."""
        from benchrig.cli import build_parser

        self.assertEqual(build_parser().parse_args(["--log-level", "debug"]).log_level, "DEBUG")

    def test_critical_is_not_an_accepted_level(self):
        """spec.md Phase 11: the flag takes DEBUG, INFO, WARNING or ERROR."""
        from benchrig import cli

        with self.assertRaises(SystemExit) as cm:
            cli.build_parser().parse_args(["--log-level", "CRITICAL"])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit):
            cli._resolve_log_level("CRITICAL")

    def test_invalid_level_message_goes_to_stderr(self):
        """A usage error belongs on stderr, so stdout stays clean for piping."""
        import contextlib
        import io

        from benchrig import cli

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit):
                cli._resolve_log_level("BANANA")
        self.assertIn("BANANA", err.getvalue())
        self.assertEqual(out.getvalue(), "")

    def test_invalid_level_exits_non_zero(self):
        """A garbage --log-level value exits non-zero with a clear error."""
        from benchrig import cli

        with self.assertRaises(SystemExit) as cm:
            cli._resolve_log_level("BANANA")
        self.assertEqual(cm.exception.code, 2)


class RunLifecycleEventTests(unittest.TestCase):
    """`run.started` and `run.completed` events emitted from run_benchmarks."""

    def _capture_run_benchmarks_log_calls(self, argv=(), runs=1, targets=(("ollama", "stub"),), scorecards=()):
        """Patch the heavy collaborators of run_benchmarks and capture every log call.

        Returns a list of kwargs that were passed to ``logger.info(...)``. We capture at
        the logger-call level (not after formatting) so the test asserts the *event*
        attribute on the record and is independent of the JSON formatter.
        """
        from benchrig.cli import run_benchmarks

        captured = []
        mock_log = MagicMock()
        mock_log.info = MagicMock(side_effect=lambda msg, *a, **k: captured.append(k))

        with patch("benchrig.cli.logging.getLogger") as mock_get_logger:
            mock_get_logger.return_value = mock_log
            with patch("benchrig.cli.resolve_target_models", return_value=list(targets)):
                with patch("benchrig.cli._warn_slow_provider", return_value=None):
                    with patch("benchrig.cli.display_system_banner", return_value=None):
                        with patch("benchrig.cli.evaluate_model", return_value=[]):
                            with patch("benchrig.cli.build_scorecards", return_value=list(scorecards)):
                                with patch("benchrig.cli.display_leaderboard", return_value=None):
                                    with patch(
                                        "benchrig.cli.display_token_savings",
                                        return_value=None,
                                    ):
                                        with patch(
                                            "benchrig.cli.show_1to1_comparison",
                                            return_value=None,
                                        ):
                                            with patch(
                                                "benchrig.cli.save_outputs",
                                                return_value=("/tmp/md", "/tmp/json"),
                                            ):
                                                with patch(
                                                    "benchrig.cli.get_system_specs",
                                                    return_value={},
                                                ):
                                                    args = self._dummy_args(argv=argv, runs=runs)
                                                    config = {}
                                                    clients = {"ollama": self._stub_client()}
                                                    run_benchmarks(
                                                        args,
                                                        config,
                                                        clients,
                                                        "ollama",
                                                    )

        return captured

    def _stub_client(self):
        c = MagicMock()
        c.display_name = "Ollama"
        c.is_reachable.return_value = False
        c.engine_name = "llama.cpp"
        c.name = "ollama"
        return c

    def _dummy_args(self, argv=(), runs=1):
        return Namespace(
            baseline=None,
            pair=None,
            models="installed",
            suite="all",
            runs=runs,
            scenarios_dir=None,
            output_dir="results",
            csv=None,
            chart=None,
            argv=list(argv),
        )

    def test_run_started_event_fires_with_argv_runtime_num_models_runs(self):
        """At the top of run_benchmarks, the first log call carries event=run.started with all four fields."""
        captured = self._capture_run_benchmarks_log_calls(("--check", "--runs", "3"), runs=3)

        started = [k for k in captured if k.get("extra", {}).get("event") == "run.started"]
        self.assertEqual(len(started), 1, f"expected exactly one run.started; got: {captured[:5]}…")
        self.assertEqual(started[0]["extra"]["runtime"], "ollama")
        self.assertEqual(started[0]["extra"]["num_models"], 1)
        self.assertEqual(started[0]["extra"]["runs"], 3)
        self.assertEqual(started[0]["extra"]["argv"], ["--check", "--runs", "3"])

    def test_run_completed_event_fires_with_total_duration(self):
        """At the end of run_benchmarks, run.completed is logged with total_duration_sec > 0."""
        captured = self._capture_run_benchmarks_log_calls()

        completed = [k for k in captured if k.get("extra", {}).get("event") == "run.completed"]
        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(completed[0]["extra"]["total_duration_sec"], 0.0)

    def test_run_completed_reports_models_ok_and_models_skipped(self):
        """spec.md Phase 11: run.completed carries models_ok and models_skipped, not internal counters."""
        captured = self._capture_run_benchmarks_log_calls(
            targets=[("ollama", "a"), ("ollama", "b")],
            scorecards=[{"model": "a", "runtime": "ollama"}],
        )
        (completed,) = [k for k in captured if k.get("extra", {}).get("event") == "run.completed"]
        extra = completed["extra"]
        self.assertEqual((extra["models_ok"], extra["models_skipped"]), (1, 1))
        self.assertNotIn("scorecards_produced", extra)
        self.assertNotIn("targets_planned", extra)

    def test_run_completed_still_fires_when_the_run_exits_early(self):
        """`finally` semantics: no models to benchmark -> sys.exit(1), run.completed still logged."""
        from benchrig.cli import run_benchmarks

        captured = []
        mock_log = MagicMock()
        mock_log.info = MagicMock(side_effect=lambda msg, *a, **k: captured.append(k))
        with patch("benchrig.cli.logging.getLogger", return_value=mock_log):
            with patch("benchrig.cli.resolve_target_models", return_value=[]):
                with self.assertRaises(SystemExit):
                    run_benchmarks(self._dummy_args(), {}, {"ollama": self._stub_client()}, "ollama")
        completed = [k for k in captured if k.get("extra", {}).get("event") == "run.completed"]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["extra"]["models_ok"], 0)


class NonBenchmarkBranchTests(unittest.TestCase):
    """`--check` and `--pull-recommended` are not benchmark runs: no run.* events (spec.md Phase 11)."""

    def _events_for(self, flag):
        import logging
        import sys

        from benchrig import cli

        records: list[logging.LogRecord] = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Collect(level=logging.DEBUG)
        logger = logging.getLogger("benchrig")
        logger.addHandler(handler)
        try:
            with (
                patch.object(sys, "argv", ["benchrig", flag, "--log-level", "INFO"]),
                patch("benchrig.cli._bootstrap_cuda_env"),
                patch("benchrig.cli.load_config", return_value={}),
                patch("benchrig.cli.create_runtime_client", return_value=MagicMock()),
                patch("benchrig.cli.run_system_check"),
                patch("benchrig.cli.pull_recommended_models"),
            ):
                cli.main()
        finally:
            logger.removeHandler(handler)
        return [getattr(r, "event", "") for r in records]

    def test_check_emits_no_run_events(self):
        self.assertNotIn("run.started", self._events_for("--check"))

    def test_pull_recommended_emits_no_run_events(self):
        self.assertNotIn("run.started", self._events_for("--pull-recommended"))


if __name__ == "__main__":
    unittest.main()
