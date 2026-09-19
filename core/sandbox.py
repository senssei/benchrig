"""Sandboxed Python code execution for verifying unit tests and coding tasks."""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

EXTRACTED_CODE_PREVIEW_CHARS = 300


def extract_python_code(text: str) -> str:
    """Extract Python code block from markdown or raw LLM output."""
    # Try ```python ... ```
    pattern = r"```(?:python|py)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        # If multiple code blocks, join them or pick the longest one containing function/def
        code_blocks_with_def = [b for b in matches if "def " in b or "class " in b]
        if code_blocks_with_def:
            return "\n\n".join(code_blocks_with_def).strip()
        return matches[-1].strip()

    # Fallback: if no code block markers, return text stripped
    return text.strip()


def _build_test_script(solution_code: str, test_assertions: list[str]) -> str:
    """Concatenate the model solution with a harness that runs every assertion independently."""
    lines = [
        "import sys",
        "import math",
        "import collections",
        "import itertools",
        "import re",
        "import json",
        "",
        "# Model Solution:",
        solution_code,
        "",
        "# Automated Test Suite Runner:",
        "passed_count = 0",
        f"total_count = {len(test_assertions)}",
        "failures = []",
    ]
    for idx, assertion in enumerate(test_assertions):
        lines.extend(
            [
                "try:",
                f"    {assertion}",
                "    passed_count += 1",
                "except Exception as e:",
                f"    failures.append(f'Test {idx + 1} failed: {{e}}')",
            ]
        )
    lines.extend(
        [
            "",
            "print(f'__RESULT__:passed={passed_count}:total={total_count}')",
            "if failures:",
            "    print('__FAILURES__:' + ' | '.join(failures), file=sys.stderr)",
            "    sys.exit(1 if passed_count == 0 else 0)",
        ]
    )
    return "\n".join(lines)


def _run_isolated(script_path: str, timeout_sec: float) -> tuple[int, str, str]:
    """
    Run a script in its own process group and return (returncode, stdout, stderr).

    On timeout the whole group is killed (not just the direct child) so runaway
    grandchildren cannot outlive the test; raises subprocess.TimeoutExpired.
    """
    proc = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.communicate()
        raise
    return proc.returncode, stdout, stderr


def _sandbox_result(
    solution_code: str,
    *,
    passed: bool = False,
    passed_tests: int = 0,
    total_tests: int = 0,
    execution_time_sec: float = 0.0,
    error: str = "",
) -> dict[str, Any]:
    """Build the result dict returned by run_code_with_tests."""
    truncated = len(solution_code) > EXTRACTED_CODE_PREVIEW_CHARS
    return {
        "passed": passed,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "pass_ratio": round(passed_tests / total_tests, 2) if total_tests > 0 else 0.0,
        "execution_time_sec": round(execution_time_sec, 3),
        "error": error,
        "extracted_code": solution_code[:EXTRACTED_CODE_PREVIEW_CHARS] + ("..." if truncated else ""),
    }


def run_code_with_tests(
    solution_code: str,
    test_assertions: list[str],
    timeout_sec: float = 5.0,
) -> dict[str, Any]:
    """
    Execute extracted solution code concatenated with test assertions in an isolated process.

    Returns:
        passed (bool), total_tests (int), passed_tests (int), pass_ratio (float),
        execution_time_sec (float), error (str), extracted_code (str preview)
    """
    total_tests = len(test_assertions)
    script = _build_test_script(solution_code, test_assertions)

    start_t = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        returncode, stdout, stderr = _run_isolated(script_path, timeout_sec)
        exec_time = time.perf_counter() - start_t

        passed_tests = 0
        res_match = re.search(r"__RESULT__:passed=(\d+):total=(\d+)", stdout)
        if res_match:
            passed_tests = int(res_match.group(1))
            total_tests = int(res_match.group(2))

        all_passed = passed_tests == total_tests and returncode == 0

        error_msg = ""
        if not all_passed:
            if stderr:
                error_msg = stderr.strip().split("\n")[-1]
            elif returncode != 0:
                error_msg = f"Process exited with code {returncode}"

        return _sandbox_result(
            solution_code,
            passed=all_passed,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time_sec=exec_time,
            error=error_msg,
        )
    except subprocess.TimeoutExpired:
        return _sandbox_result(
            solution_code,
            total_tests=total_tests,
            execution_time_sec=timeout_sec,
            error=f"Execution timed out (> {timeout_sec}s)",
        )
    except Exception as e:  # sandbox must never crash the benchmark run
        return _sandbox_result(solution_code, total_tests=total_tests, error=str(e))
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
