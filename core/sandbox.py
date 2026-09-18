"""Sandboxed Python code execution for verifying unit tests and coding tasks."""

import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional


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


def run_code_with_tests(
    solution_code: str,
    test_assertions: List[str],
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Execute extracted solution code concatenated with test assertions in an isolated process.
    
    Returns:
        passed (bool), total_tests (int), passed_tests (int), error (str), execution_time (float)
    """
    test_harness = [
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
        test_harness.extend([
            f"try:",
            f"    {assertion}",
            f"    passed_count += 1",
            f"except Exception as e:",
            f"    failures.append(f'Test {idx+1} failed: {{e}}')",
        ])

    test_harness.extend([
        "",
        "print(f'__RESULT__:passed={passed_count}:total={total_count}')",
        "if failures:",
        "    print('__FAILURES__:' + ' | '.join(failures), file=sys.stderr)",
        "    sys.exit(1 if passed_count == 0 else 0)",
    ])

    full_script = "\n".join(test_harness)

    start_t = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        exec_time = time.perf_counter() - start_t
        stdout = proc.stdout
        stderr = proc.stderr

        # Parse test results
        passed_tests = 0
        total_tests = len(test_assertions)

        res_match = re.search(r"__RESULT__:passed=(\d+):total=(\d+)", stdout)
        if res_match:
            passed_tests = int(res_match.group(1))
            total_tests = int(res_match.group(2))

        all_passed = (passed_tests == total_tests) and (proc.returncode == 0)

        error_msg = ""
        if not all_passed:
            if stderr:
                error_msg = stderr.strip().split("\n")[-1]
            elif proc.returncode != 0:
                error_msg = f"Process exited with code {proc.returncode}"

        return {
            "passed": all_passed,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "pass_ratio": round(passed_tests / total_tests, 2) if total_tests > 0 else 0.0,
            "execution_time_sec": round(exec_time, 3),
            "error": error_msg,
            "extracted_code": solution_code[:300] + ("..." if len(solution_code) > 300 else ""),
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "passed_tests": 0,
            "total_tests": len(test_assertions),
            "pass_ratio": 0.0,
            "execution_time_sec": timeout_sec,
            "error": f"Execution timed out (> {timeout_sec}s)",
            "extracted_code": solution_code[:300],
        }
    except Exception as e:
        return {
            "passed": False,
            "passed_tests": 0,
            "total_tests": len(test_assertions),
            "pass_ratio": 0.0,
            "execution_time_sec": 0.0,
            "error": str(e),
            "extracted_code": solution_code[:300],
        }
