"""`_suite_skip_notice` (plan.md item 4.6 follow-up): when the capacity short-circuit in
`BenchmarkRunner` leaves a suite with zero results, the CLI must say why instead of silently printing
"Running <suite> tests..." followed by nothing.
"""

import unittest
from unittest.mock import MagicMock

from benchrig.cli import _suite_skip_notice
from benchrig.core.runner import BenchmarkRunner


def _runner_with_reason(reason: str | None) -> BenchmarkRunner:
    runner = MagicMock(spec=BenchmarkRunner)
    runner.capacity_exhausted_reason = reason
    return runner


class SuiteSkipNoticeTests(unittest.TestCase):
    def test_no_notice_when_the_suite_actually_produced_results(self):
        runner = _runner_with_reason("insufficient_resources: ...")
        self.assertIsNone(_suite_skip_notice(runner, [{"suite": "coding"}], [{"id": "c1"}]))

    def test_no_notice_when_there_were_no_scenarios_to_run(self):
        runner = _runner_with_reason(None)
        self.assertIsNone(_suite_skip_notice(runner, [], []))

    def test_notice_when_results_are_empty_because_capacity_was_exhausted(self):
        runner = _runner_with_reason("insufficient_resources: model does not fit")
        reason = _suite_skip_notice(runner, [], [{"id": "c1"}])
        self.assertEqual(reason, "insufficient_resources: model does not fit")

    def test_no_notice_when_results_are_empty_but_runner_has_no_recorded_reason(self):
        """Belt-and-suspenders: an empty suite for some other reason must not fabricate a VRAM message."""
        runner = _runner_with_reason(None)
        self.assertIsNone(_suite_skip_notice(runner, [], [{"id": "c1"}]))


class RunnerExposesCapacityReasonTests(unittest.TestCase):
    def test_capacity_exhausted_reason_defaults_to_none(self):
        runner = BenchmarkRunner(client=MagicMock(), config={})
        self.assertIsNone(runner.capacity_exhausted_reason)

    def test_capacity_exhausted_reason_reflects_internal_state(self):
        runner = BenchmarkRunner(client=MagicMock(), config={})
        runner._capacity_exhausted_reason = "insufficient_resources: nope"
        self.assertEqual(runner.capacity_exhausted_reason, "insufficient_resources: nope")


if __name__ == "__main__":
    unittest.main()
