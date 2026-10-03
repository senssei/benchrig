"""
Isolated subprocess execution for verifying Python coding tasks and unit tests.

Uses dedicated process groups with strict timeouts and out-of-band JSON token
verification. Note on isolation: this provides process-group containment and
clean timeout termination, not OS-level sandboxing (e.g. seccomp or containers).
Untrusted code executes under the user's Python interpreter and could attempt
system calls (subprocess, socket, ctypes).
"""

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

from benchrig.core.reasoning_parser import extract_thinking_and_answer

EXTRACTED_CODE_PREVIEW_CHARS = 300


def extract_python_code(text: str) -> str:
    """Extract Python code block from markdown or raw LLM output, ignoring any `<think>` reasoning trace."""
    text = extract_thinking_and_answer(text)["answer"]
    # Try ```python ... ```
    pattern = r"```(?:python|py)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if len(matches) == 1:
        # A single fence nested under a markdown list/bullet keeps the list's margin on every line; dedent it
        # (before stripping) so that margin does not survive as a spurious indent on some lines but not others
        # (`.strip()` alone only trims the outer edges of the whole block, e.g. `IndentationError: unindent does
        # not match any outer indentation level` on a later same-level statement). Only done for a single fence:
        # with more than one, a later block may be a continuation fragment (e.g. a method added to a class shown
        # in an earlier fence) whose indentation is relative to that block, not markdown noise, and dedenting it
        # on its own would strip the very indentation that keeps it nested.
        matches = [textwrap.dedent(matches[0])]
    if matches:
        # If multiple code blocks, join them or pick the longest one containing function/def
        code_blocks_with_def = [b for b in matches if "def " in b or "class " in b]
        if code_blocks_with_def:
            return "\n\n".join(code_blocks_with_def).strip()
        return matches[-1].strip()

    # Fallback: if no code block markers, return text stripped
    return text.strip()


def has_complete_code_block(text: str) -> bool:
    """True if the answer part of `text` (outside any `<think>` trace) contains a closed ``` code fence."""
    answer = extract_thinking_and_answer(text)["answer"]
    return re.search(r"```(?:python|py)?\n.*?```", answer, re.DOTALL | re.IGNORECASE) is not None


def _build_test_script(
    solution_code: str,
    test_assertions: list[str],
    result_path: str,
    token: str,
) -> str:
    """Concatenate the model solution with a harness that runs every assertion independently and reports out-of-band."""
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
        f"_RESULT_PATH = {repr(result_path)}",
        f"_TOKEN = {repr(token)}",
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
                "except (Exception, SystemExit) as e:",
                f"    failures.append(f'Test {idx + 1} failed: {{type(e).__name__}}: {{e}}')",
            ]
        )
    lines.extend(
        [
            "",
            "try:",
            "    with open(_RESULT_PATH, 'w', encoding='utf-8') as _rf:",
            "        json.dump({",
            "            'token': _TOKEN,",
            "            'completed': True,",
            "            'passed_count': passed_count,",
            "            'total_count': total_count,",
            "            'failures': failures,",
            "        }, _rf)",
            "except Exception as _e:",
            "    print(f'Failed to write result file: {_e}', file=sys.stderr)",
            "",
            "if failures:",
            "    print('__FAILURES__:' + ' | '.join(failures), file=sys.stderr)",
            "    sys.exit(1)",
            "sys.exit(0)",
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
    Execute extracted solution code concatenated with test assertions in an isolated subprocess.

    Runs in a dedicated process group with strict timeouts (killed via SIGKILL on POSIX)
    and out-of-band JSON completion token verification.

    Returns:
        passed (bool), total_tests (int), passed_tests (int), pass_ratio (float),
        execution_time_sec (float), error (str), extracted_code (str preview)
    """
    total_tests = len(test_assertions)
    token = secrets.token_hex(16)

    script_path = None
    result_path = None

    start_t = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as sf:
        script_path = sf.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as rf:
        result_path = rf.name
        # Pre-seed result file with completed=False to detect incomplete runs or early exits
        json.dump({"token": token, "completed": False}, rf)

    try:
        script = _build_test_script(solution_code, test_assertions, result_path, token)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        returncode, stdout, stderr = _run_isolated(script_path, timeout_sec)
        exec_time = time.perf_counter() - start_t

        passed_tests = 0
        completed = False
        failures: list[str] = []

        try:
            with open(result_path, encoding="utf-8") as rf:
                result_data = json.load(rf)
            if isinstance(result_data, dict) and result_data.get("token") == token:
                completed = bool(result_data.get("completed", False))
                if completed:
                    passed_tests = int(result_data.get("passed_count", 0))
                    failures = [str(x) for x in result_data.get("failures", [])]
        except Exception:
            completed = False

        all_passed = total_tests > 0 and completed and passed_tests == total_tests and returncode == 0

        error_msg = ""
        if not all_passed:
            if failures:
                error_msg = " | ".join(failures)
            elif stderr:
                error_msg = stderr.strip().split("\n")[-1]
            elif returncode != 0:
                error_msg = f"Process exited with code {returncode}"
            else:
                error_msg = "Execution terminated before test suite completed"

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
        for p in (script_path, result_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
