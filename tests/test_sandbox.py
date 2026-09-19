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
