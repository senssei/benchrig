"""Tests for the PNG chart export of scorecards (plan.md Phase 1, item 1.2)."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch


def card(model, runtime="ollama", engine="llama.cpp", composite_score=80.0, **extra):
    return {
        "model": model,
        "runtime": runtime,
        "engine": engine,
        "composite_score": composite_score,
        "coding_pass_rate": 100.0,
        "reasoning_accuracy": 66.7,
        "avg_eval_tok_sec": 100.0,
        "avg_ttft_sec": 0.1,
        "peak_vram_mb": 5000.0,
        "total_runs": 3,
        **extra,
    }


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ChartExportTests(unittest.TestCase):
    def test_chart_export_writes_png_with_one_bar_per_scorecard(self):
        # Import inside the test so a missing module reads as "not found" / red, not a collection error.
        from benchrig.reporting.charts import write_scorecards_chart

        scorecards = [
            card("phi4-mini:latest", runtime="ollama", composite_score=80.0),
            card("deepseek-r1:14b", runtime="foundry", composite_score=65.0),
            card("qwen2.5:7b", runtime="prism", composite_score=72.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scorecards.png")
            returned = write_scorecards_chart(scorecards, path)

            self.assertEqual(returned, path)
            with open(path, "rb") as fh:
                sig = fh.read(len(PNG_SIGNATURE))
            self.assertEqual(sig, PNG_SIGNATURE, "output is not a PNG file")
            # Non-trivial size — an empty figure or a save-failure stub would be a few hundred bytes at most.
            self.assertGreater(os.path.getsize(path), 1000, f"PNG too small ({os.path.getsize(path)} bytes)")

    def test_make_scorecard_figure_has_one_bar_per_scorecard(self):
        """Inspect the rendered figure: count ``Rectangle`` patches on the axes (= bars)."""
        from benchrig.reporting.charts import make_scorecard_figure

        scorecards = [
            card("phi4-mini:latest", composite_score=80.0),
            card("deepseek-r1:14b", runtime="foundry", composite_score=65.0),
            card("qwen2.5:7b", runtime="prism", composite_score=72.0),
        ]
        fig = make_scorecard_figure(scorecards)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        # Every bar on a bar() call lands as a Rectangle patch with get_height().
        bars = [p for p in ax.patches if hasattr(p, "get_height")]
        self.assertEqual(len(bars), len(scorecards), f"expected {len(scorecards)} bars, got {len(bars)}")
        heights = sorted(round(b.get_height(), 1) for b in bars)
        self.assertEqual(heights, sorted(sc["composite_score"] for sc in scorecards))
        # Tick labels = one per scorecard, in order; format is "<model> (<runtime>)".
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()],
            [f"{sc['model']} ({sc.get('runtime', 'unknown')})" for sc in scorecards],
        )

    def test_bar_labels_are_rotated_to_avoid_overlap(self):
        """Bug fix (plan.md Phase 7): horizontal labels overlap into an unreadable strip once there are more
        than a couple of scorecards. Labels must be rotated 30° with right alignment (spec.md Phase 1)."""
        from benchrig.reporting.charts import make_scorecard_figure

        scorecards = [
            card("mistral-7b-instruct-v0.2-q4_0", runtime="prism", composite_score=71.0),
            card("phi-4-mini", runtime="foundry", composite_score=45.0),
            card("nomic-embed-text-v1.5", runtime="ollama", composite_score=45.0),
        ]
        fig = make_scorecard_figure(scorecards)
        ax = fig.axes[0]
        for tick_label in ax.get_xticklabels():
            self.assertEqual(tick_label.get_rotation(), 30, "bar label is not rotated; will overlap")
            self.assertEqual(tick_label.get_horizontalalignment(), "right")

    def test_charts_module_does_not_import_matplotlib_at_module_load(self):
        # Drop matplotlib from sys.modules to make the assertion meaningful.
        for mod in [m for m in list(sys.modules) if m == "matplotlib" or m.startswith("matplotlib.")]:
            del sys.modules[mod]
        # Importing the module must not pull matplotlib in (spec.md I3).
        import benchrig.reporting.charts  # noqa: F401

        leaked = sorted(m for m in sys.modules if m == "matplotlib" or m.startswith("matplotlib."))
        self.assertEqual(leaked, [], f"matplotlib imported eagerly at module load: {leaked}")

    def test_charts_missing_matplotlib_raises_clear_error(self):
        from benchrig.reporting.charts import write_scorecards_chart

        # Pretend matplotlib is not installed: pyplot import inside the function will fail.
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with self.assertRaises(RuntimeError) as ctx:
                write_scorecards_chart([card("phi4-mini:latest")], "/tmp/should-not-exist.png")
        self.assertIn("matplotlib", str(ctx.exception).lower())

    def test_charts_writes_no_partial_file_when_matplotlib_missing(self):
        from benchrig.reporting.charts import write_scorecards_chart

        target = "/tmp/__benchrig_no_partial__.png"
        if os.path.exists(target):
            os.remove(target)
        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
            with self.assertRaises(RuntimeError):
                write_scorecards_chart([card("phi4-mini:latest")], target)
        self.assertFalse(os.path.exists(target), "partial PNG was left on disk after error")


if __name__ == "__main__":
    unittest.main()
