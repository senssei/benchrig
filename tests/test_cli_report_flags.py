"""Tests that `--csv`/`--chart` are actually wired end-to-end through the CLI (plan.md Phase 1,
items 1.1-1.3). Items 1.1/1.2 claimed the flags were wired into ``benchrig/cli.py`` and shipped, but
`save_outputs()` never called the writers and `build_parser()` never defined the flags — this file
pins the fix.
"""

import os
import tempfile
import unittest


def card(model, runtime="ollama", engine="llama.cpp", **extra):
    return {
        "model": model,
        "runtime": runtime,
        "engine": engine,
        "composite_score": 90.0,
        "coding_pass_rate": 100.0,
        "reasoning_accuracy": 66.7,
        "avg_eval_tok_sec": 100.0,
        "avg_ttft_sec": 0.1,
        "peak_vram_mb": 5000.0,
        "total_runs": 3,
        **extra,
    }


class CliFlagsExistTests(unittest.TestCase):
    def test_csv_and_chart_flags_are_registered(self):
        from benchrig import cli

        parser = cli.build_parser()
        options = {o for a in parser._actions for o in a.option_strings}
        self.assertIn("--csv", options, "--csv is not registered on the argument parser")
        self.assertIn("--chart", options, "--chart is not registered on the argument parser")

    def test_parsed_args_expose_csv_and_chart(self):
        from benchrig import cli

        args = cli.build_parser().parse_args(["--csv", "out.csv", "--chart", "out.png"])
        self.assertEqual(args.csv, "out.csv")
        self.assertEqual(args.chart, "out.png")


class SaveOutputsWritesRequestedArtifactsTests(unittest.TestCase):
    def test_save_outputs_writes_csv_and_chart_when_requested(self):
        from benchrig import cli

        scorecards = [card("phi4-mini:latest"), card("deepseek-r1:14b", runtime="foundry")]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "scorecards.csv")
            chart_path = os.path.join(tmp, "scorecards.png")
            md_path, _raw_json_path = cli.save_outputs(
                tmp,
                specs={},
                scorecards=scorecards,
                results=[],
                total_duration=1.0,
                csv_path=csv_path,
                chart_path=chart_path,
            )

            self.assertTrue(os.path.isfile(csv_path), "--csv path was not written by save_outputs")
            self.assertTrue(os.path.isfile(chart_path), "--chart path was not written by save_outputs")
            with open(md_path, encoding="utf-8") as f:
                report = f.read()
            self.assertIn("scorecards.png", report, "markdown report does not link the produced chart")
            self.assertIn("scorecards.csv", report, "markdown report does not link the produced CSV")

    def test_save_outputs_skips_csv_and_chart_when_not_requested(self):
        from benchrig import cli

        scorecards = [card("phi4-mini:latest")]
        with tempfile.TemporaryDirectory() as tmp:
            cli.save_outputs(tmp, specs={}, scorecards=scorecards, results=[], total_duration=1.0)
            self.assertFalse(os.path.isfile(os.path.join(tmp, "scorecards.csv")))
            self.assertFalse(os.path.isfile(os.path.join(tmp, "scorecards.png")))


if __name__ == "__main__":
    unittest.main()
