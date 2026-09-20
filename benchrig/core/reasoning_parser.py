"""Reasoning and logic evaluator with <think> tag extraction (DeepSeek-R1 support)."""

import re
from typing import Any

# For `regex` checks only the end of the answer counts: a conclusion is stated last, and an unbounded pattern would
# otherwise be satisfied by words scattered through the reasoning steps of a wrong answer.
CONCLUSION_WINDOW_CHARS = 300


def normalize_for_pattern(text: str) -> str:
    """Collapse markdown, bullets, punctuation and line breaks to single spaces, keeping letters and digits (any script).

    A conclusion written on one line, on separate lines or as bullets then reads the same: `- **Box 1**: Oranges` becomes
    `Box 1 Oranges`.
    """
    return re.sub(r"[\W_]+", " ", text).strip()


def simplify_latex(text: str) -> str:
    r"""Rewrite the LaTeX that reasoning models like to answer with into plain text: `\boxed{\dfrac{1}{6}}` -> `1/6`."""
    text = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.replace("\\text", "").replace("\\$", "$")


def contains_expected(haystack: str, pattern: str) -> bool:
    """Whether `pattern` occurs in `haystack`; numbers only match as whole numbers.

    A plain substring test accepted `11/60` and `1/60` for `1/6`, and `1114.60` for `114.6`. Patterns that contain a digit
    must not be glued to another digit, a decimal point or a fraction slash on either side; other patterns are plain
    substrings, as before. Both arguments are compared in lowercase.
    """
    haystack, pattern = haystack.lower(), pattern.lower()
    if not re.search(r"\d", pattern):
        return pattern in haystack
    return re.search(rf"(?<![\d./,]){re.escape(pattern)}(?!\d|[.,]\d|/\d)", haystack) is not None


def extract_final_answer(text: str) -> str | None:
    """The text after the last `Final answer` marker (same line, or the next when it is on a line of its own), if any."""
    found = re.findall(r"final answer\W*(?:is\W*)?([^\n]+)", text, re.IGNORECASE)
    return found[-1].strip() if found else None


def normalize_final_answer(text: str) -> str:
    """Plain, comparable form of a final answer: no markdown, LaTeX, `$`, thousands separators or trailing full stop."""
    text = simplify_latex(text)
    text = re.sub(r"[*_`$]|\\[()\[\]]", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.sub(r"\s+", " ", text).strip().rstrip(".").strip()


def extract_thinking_and_answer(text: str) -> dict[str, Any]:
    """
    Split a response into its thinking trace and its final answer.

    Supports the `<think>...</think>` tags used by DeepSeek-R1 and similar reasoning models, including two forms that
    a plain regex misses:
    - an unterminated `<think>` (the token budget ran out while the model was still thinking): there is no answer, so
      `answer` is empty and `truncated_thinking` is True. Treating the reasoning trace as the answer would let a trace
      that merely mentions the right words pass;
    - a bare `</think>` with no opening tag (some templates put `<think>` in the prompt).
    """
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return {
            "has_think_tags": True,
            "thinking": match.group(1).strip(),
            "answer": text.replace(match.group(0), "").strip(),
            "truncated_thinking": False,
        }
    open_at = text.find("<think>")
    if open_at != -1:  # opened but never closed
        return {
            "has_think_tags": True,
            "thinking": text[open_at + len("<think>") :].strip(),
            "answer": text[:open_at].strip(),
            "truncated_thinking": True,
        }
    close_at = text.find("</think>")
    if close_at != -1:  # closed without an opening tag
        return {
            "has_think_tags": True,
            "thinking": text[:close_at].strip(),
            "answer": text[close_at + len("</think>") :].strip(),
            "truncated_thinking": False,
        }
    return {"has_think_tags": False, "thinking": "", "answer": text.strip(), "truncated_thinking": False}


def evaluate_reasoning_answer(
    response_text: str,
    expected_answer: str,
    check_type: str = "exact_or_contains",
    accepted_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate if reasoning response arrived at the correct conclusion.

    check_type:
    - "exact_or_contains": answer contains expected string
    - "numeric": extracts number (or boxed number) and compares mathematically
    - "regex": tests the normalized end of the answer (see `normalize_for_pattern`) against regular expressions
    - "final_answer": the line after the last `Final answer` marker must fully match one of `accepted_patterns`
    """
    parsed = extract_thinking_and_answer(response_text)
    answer = parsed["answer"]
    matchable = simplify_latex(answer)  # for the string and pattern checks; the numeric check reads \boxed itself

    correct = False
    extracted_val = ""

    if check_type == "numeric":
        # Look for \boxed{X} or the last standalone number in answer
        boxed_match = re.findall(r"\\boxed\{([^}]+)\}", answer)
        if boxed_match:
            extracted_val = boxed_match[-1].strip()
        else:
            # Find numbers in the answer
            nums = re.findall(r"[-+]?\d*\.?\d+", answer)
            if nums:
                extracted_val = nums[-1]

        try:
            exp_num = float(expected_answer.strip())
            act_num = float(extracted_val) if extracted_val else None
            if act_num is not None and abs(act_num - exp_num) < 1e-4:
                correct = True
        except ValueError:
            correct = extracted_val == expected_answer.strip()

    elif check_type == "regex" and accepted_patterns:
        conclusion = normalize_for_pattern(matchable)[-CONCLUSION_WINDOW_CHARS:]
        for pat in accepted_patterns:
            if re.search(pat, conclusion, re.IGNORECASE):
                correct = True
                extracted_val = pat
                break

    elif check_type == "final_answer":
        final = extract_final_answer(matchable)
        if final is not None:
            extracted_val = normalize_final_answer(final)
            correct = any(re.fullmatch(pat, extracted_val, re.IGNORECASE) for pat in accepted_patterns or [])

    else:  # exact_or_contains
        exp_clean = expected_answer.strip().lower()
        ans_clean = matchable.lower()
        if contains_expected(ans_clean, exp_clean):
            correct = True
            extracted_val = exp_clean
        elif accepted_patterns:
            for pat in accepted_patterns:
                if contains_expected(ans_clean, pat):
                    correct = True
                    extracted_val = pat
                    break

    return {
        "correct": correct,
        "has_think_tags": parsed["has_think_tags"],
        "think_chars": len(parsed["thinking"]),
        "think_preview": (parsed["thinking"][:150] + "...") if parsed["thinking"] else "",
        "extracted_answer": extracted_val or (answer[:80] + "..."),
        # The end of the answer (where conclusions are stated), so a result can be audited without re-running the model.
        "answer_excerpt": answer if len(answer) <= 400 else "..." + answer[-400:],
        "truncated_thinking": parsed["truncated_thinking"],
        "expected_answer": expected_answer,
    }
