"""Unit tests for the sandboxed code runner."""

import glob
import os
import tempfile
import time
import unittest

from benchrig.core.sandbox import extract_python_code, run_code_with_tests


class ExtractCodeTests(unittest.TestCase):
    def test_prefers_blocks_with_definitions(self):
        text = "```python\nprint('hi')\n```\ntext\n```python\ndef f():\n    pass\n```"
        self.assertEqual(extract_python_code(text), "def f():\n    pass")

    def test_falls_back_to_raw_text(self):
        self.assertEqual(extract_python_code("  x = 1  "), "x = 1")

    def test_does_not_dedent_a_continuation_fence_that_shares_indentation_with_an_earlier_block(self):
        """Review finding: dedenting every block independently breaks a model that shows a class in one fence and
        adds a method to it in a second, indented fence — the second block's indent is relative to the class, not
        markdown noise, and must survive the join so `subtract` stays a method instead of becoming a free function."""
        text = (
            "```python\n"
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "```\n\n"
            "```python\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n"
            "```\n"
        )
        code = extract_python_code(text)
        res = run_code_with_tests(
            code, ["assert Calculator().add(1, 2) == 3", "assert Calculator().subtract(5, 3) == 2"]
        )
        self.assertTrue(res["passed"], res["error"])

    def test_dedents_a_fence_indented_under_a_markdown_list(self):
        """spec.md Phase 9 item 9.2: a ```python fence nested under a list item keeps the model's indentation
        relative to the list margin; only the outer .strip() (not a per-line dedent) leaves later same-level
        statements at the list's leftover margin instead of column 0, which raises IndentationError."""
        text = (
            "Here is the solution:\n\n"
            "   ```python\n"
            "   def add(a, b):\n"
            "       return a + b\n"
            "\n"
            "   print(add(1, 2))\n"
            "   ```\n"
        )
        code = extract_python_code(text)
        compile(code, "<extracted>", "exec")  # raises IndentationError before the fix
        self.assertEqual(code, "def add(a, b):\n    return a + b\n\nprint(add(1, 2))")


class RunCodeTests(unittest.TestCase):
    def test_partial_pass_counts_individual_assertions(self):
        res = run_code_with_tests("def f(x):\n    return x + 1", ["assert f(1) == 2", "assert f(1) == 3"])
        self.assertEqual((res["passed_tests"], res["total_tests"]), (1, 2))
        self.assertFalse(res["passed"])
        self.assertEqual(res["pass_ratio"], 0.5)

    def test_all_pass(self):
        res = run_code_with_tests("def f(x):\n    return x", ["assert f(1) == 1"])
        self.assertTrue(res["passed"])
        self.assertEqual(res["error"], "")

    def test_syntax_error_is_reported(self):
        res = run_code_with_tests("def broken(:", ["assert True"])
        self.assertFalse(res["passed"])
        self.assertIn("SyntaxError", res["error"])

    def test_timeout_terminates_and_reports(self):
        start = time.perf_counter()
        res = run_code_with_tests("while True:\n    pass", ["assert True"], timeout_sec=1)
        self.assertLess(time.perf_counter() - start, 5)
        self.assertFalse(res["passed"])
        self.assertIn("timed out", res["error"])

    def test_temp_script_is_removed(self):
        pattern = os.path.join(tempfile.gettempdir(), "tmp*.py")
        before = set(glob.glob(pattern))
        run_code_with_tests("x = 1", ["assert x == 1"])
        self.assertEqual(set(glob.glob(pattern)) - before, set())

    def test_long_code_preview_is_truncated(self):
        res = run_code_with_tests("x = 1\n" + "# pad\n" * 200, ["assert x == 1"])
        self.assertTrue(res["extracted_code"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
