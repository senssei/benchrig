"""Reasoning and logic evaluator with <think> tag extraction (DeepSeek-R1 support)."""

import logging
import re
from fractions import Fraction
from typing import Any

_log = logging.getLogger(__name__)

# For `regex` checks only the end of the answer counts: a conclusion is stated last, and an unbounded pattern would
# otherwise be satisfied by words scattered through the reasoning steps of a wrong answer.
CONCLUSION_WINDOW_CHARS = 300

# English number words for the `"numeric_text"` evaluator. Case-sensitive on purpose: the Phase 12.3 test
# pins that `Forty-Two` does NOT match `forty-two` without an explicit case-insensitive opt-in.
_WORD_TO_DIGIT = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}


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


def _try_fraction(s: str) -> Fraction | None:
    """Return `Fraction(s)` if `s` parses as a number, else `None`. Whitespace must already be trimmed."""
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def _text_to_int(s: str) -> int | None:
    """Convert an English number phrase to an integer (case-sensitive). Returns `None` if not recognized.

    Supports: `zero`, `one`, ..., `nineteen`, `twenty`, `thirty`, ..., `ninety`, `hundred`, `thousand`,
    `million`, and hyphenated forms like `forty-two`, `twenty-one`. `Forty-Two` does NOT match.
    """
    s = s.strip().replace("-", " ")
    tokens = s.split()
    if not tokens:
        return None
    for t in tokens:
        if t not in _WORD_TO_DIGIT and t not in _WORD_SCALES:
            return None
    total = 0
    current = 0
    for t in tokens:
        if t in _WORD_TO_DIGIT:
            current += _WORD_TO_DIGIT[t]
        elif t == "hundred":
            current = (current or 1) * 100
        else:  # thousand / million
            total += (current or 1) * _WORD_SCALES[t]
            current = 0
    total += current
    return total if total > 0 or tokens == ["zero"] else None


def _to_number(s: str) -> Fraction | None:
    """Parse `s` as a number: digits first (`Fraction`), else English number words. Returns `None` if neither.

    Whitespace is trimmed first; both sides are converted. Hyphens are split (`forty-two` → `forty two`).
    """
    s = s.strip()
    frac = _try_fraction(s)
    if frac is not None:
        return frac
    word = _text_to_int(s)
    if word is not None:
        return Fraction(word)
    return None


def evaluate_reasoning_answer(
    response_text: str,
    expected_answer: str,
    check_type: str = "exact_or_contains",
    accepted_patterns: list[str] | None = None,
    evaluator: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate if reasoning response arrived at the correct conclusion.

    check_type:
    - "exact_or_contains": answer contains expected string
    - "numeric": extracts number (or boxed number) and compares mathematically
    - "regex": tests the normalized end of the answer (see `normalize_for_pattern`) against regular expressions
    - "final_answer": the line after the last `Final answer` marker must fully match one of `accepted_patterns`

    evaluator (Phase 12.3, opt-in per scenario; absent means today's behavior):
    - "numeric": isolate the answer (extract value after `Final answer:` marker if present, trim whitespace);
      if both sides parse as a `Fraction`, compare exactly; otherwise fall through to `check_type`.
    - "numeric_text": same as `"numeric"` plus English-text-to-number on both sides first.
    - any other non-None value: treated as None, emits `evaluator.unknown` structured-log event with
      `scenario_id` (when provided) and `evaluator_value` (the rejected value).

    scenario_id: optional scenario identifier (e.g. `"reasoning_bat_ball"`) included in the
    `evaluator.unknown` structured-log event payload so a misspelled field can be traced.
    """
    if evaluator is not None and (not isinstance(evaluator, str) or evaluator not in ("numeric", "numeric_text")):
        _log.warning(
            "evaluator.unknown",
            extra={
                "event": "evaluator.unknown",
                "evaluator_value": evaluator,
                "scenario_id": scenario_id,
            },
        )
        evaluator = None

    parsed = extract_thinking_and_answer(response_text)
    answer = parsed["answer"]
    matchable = simplify_latex(answer)  # for the string and pattern checks; the numeric check reads \boxed itself

    correct = False
    extracted_val = ""
    numeric_layer_applied = False

    if evaluator in ("numeric", "numeric_text"):
        # Isolate the value after a `Final answer:` marker (if present); fall back to the full answer.
        # Applied to both numeric and numeric_text so a `final_answer` scenario with a prose prefix
        # actually exercises the numeric layer instead of being skipped.
        ans_candidate = (extract_final_answer(matchable) or answer).strip()
        exp_candidate = expected_answer.strip()
        if evaluator == "numeric_text":
            ans_frac = _to_number(ans_candidate)
            exp_frac = _to_number(exp_candidate)
        else:
            ans_frac = _try_fraction(ans_candidate)
            exp_frac = _try_fraction(exp_candidate)
        if ans_frac is not None and exp_frac is not None:
            correct = ans_frac == exp_frac
            extracted_val = ans_candidate
            numeric_layer_applied = True

    if not numeric_layer_applied:
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
