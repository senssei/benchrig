"""Tests for the Markdown report embedding chart/CSV links (plan.md Phase 1, item 1.3)."""

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


class MarkdownAttachmentTests(unittest.TestCase):
    def test_markdown_report_links_to_chart_and_csv_when_produced(self):
        # Import inside the test so a missing module reads as "not found" / red.
        from benchrig.reporting.markdown import generate_markdown_report

        scorecards = [card("phi4-mini:latest"), card("deepseek-r1:14b", runtime="foundry")]
        with tempfile.TemporaryDirectory() as tmp:
            chart_path = os.path.join(tmp, "scorecards.png")
            csv_path = os.path.join(tmp, "scorecards.csv")
            md_path = os.path.join(tmp, "report.md")
            content = generate_markdown_report(
                scorecards,
                [],
                {},
                output_path=md_path,
                chart_path=chart_path,
                csv_path=csv_path,
            )

        # Chart embedded as an image near the top, with alt text "Composite scores".
        chart_mark = f"![Composite scores]({chart_path})"
        self.assertIn(chart_mark, content, "chart not embedded as Markdown image")
        # CSV linked as a regular Markdown link.
        csv_mark = f"[CSV]({csv_path})"
        self.assertIn(csv_mark, content, "CSV not linked from the Markdown report")
        # Both attachments appear before the auto-generated footer.
        self.assertLess(content.index(chart_mark), content.index("Generated automatically"))
        self.assertLess(content.index(csv_mark), content.index("Generated automatically"))

    def test_markdown_report_has_no_attachment_links_when_paths_omitted(self):
        from benchrig.reporting.markdown import generate_markdown_report

        scorecards = [card("phi4-mini:latest")]
        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "report.md")
            content = generate_markdown_report(scorecards, [], {}, output_path=md_path)
        # Neither image embed nor CSV link when nothing was produced.
        self.assertNotIn("![]", content, "unexpected image embed without --chart")
        self.assertNotIn(".csv)", content, "unexpected CSV link without --csv")

    def test_markdown_report_links_chart_only_when_only_chart_produced(self):
        from benchrig.reporting.markdown import generate_markdown_report

        scorecards = [card("phi4-mini:latest")]
        with tempfile.TemporaryDirectory() as tmp:
            chart_path = os.path.join(tmp, "scorecards.png")
            md_path = os.path.join(tmp, "report.md")
            content = generate_markdown_report(
                scorecards,
                [],
                {},
                output_path=md_path,
                chart_path=chart_path,
            )
        self.assertIn(f"![Composite scores]({chart_path})", content)
        self.assertNotIn(".csv)", content, "CSV link appeared even though --csv was not requested")

    def test_markdown_report_links_csv_only_when_only_csv_produced(self):
        from benchrig.reporting.markdown import generate_markdown_report

        scorecards = [card("phi4-mini:latest")]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "scorecards.csv")
            md_path = os.path.join(tmp, "report.md")
            content = generate_markdown_report(
                scorecards,
                [],
                {},
                output_path=md_path,
                csv_path=csv_path,
            )
        self.assertIn(f"[CSV]({csv_path})", content)
        self.assertNotIn("![]", content, "image embed appeared even though --chart was not requested")


if __name__ == "__main__":
    unittest.main()
