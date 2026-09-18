"""Reasoning and logic evaluator with <think> tag extraction (DeepSeek-R1 support)."""

import re
from typing import Any, Dict, List, Optional


def extract_thinking_and_answer(text: str) -> Dict[str, str]:
    """
    Extract thinking trace and final answer.
    Supports <think>...</think> tags used by DeepSeek-R1 and similar reasoning models.
    """
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        answer = text.replace(match.group(0), "").strip()
        return {
            "has_think_tags": True,
            "thinking": thinking,
            "answer": answer,
        }
    return {
        "has_think_tags": False,
        "thinking": "",
        "answer": text.strip(),
    }


def evaluate_reasoning_answer(
    response_text: str,
    expected_answer: str,
    check_type: str = "exact_or_contains",
    accepted_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate if reasoning response arrived at the correct conclusion.
    
    check_type:
    - "exact_or_contains": answer contains expected string
    - "numeric": extracts number (or boxed number) and compares mathematically
    - "regex": tests against regular expressions
    """
    parsed = extract_thinking_and_answer(response_text)
    answer = parsed["answer"]

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
            correct = (extracted_val == expected_answer.strip())

    elif check_type == "regex" and accepted_patterns:
        for pat in accepted_patterns:
            if re.search(pat, answer, re.IGNORECASE):
                correct = True
                extracted_val = pat
                break

    else:  # exact_or_contains
        exp_clean = expected_answer.strip().lower()
        ans_clean = answer.lower()
        if exp_clean in ans_clean:
            correct = True
            extracted_val = exp_clean
        elif accepted_patterns:
            for pat in accepted_patterns:
                if pat.lower() in ans_clean:
                    correct = True
                    extracted_val = pat
                    break

    return {
        "correct": correct,
        "has_think_tags": parsed["has_think_tags"],
        "think_chars": len(parsed["thinking"]),
        "think_preview": (parsed["thinking"][:150] + "...") if parsed["thinking"] else "",
        "extracted_answer": extracted_val or (answer[:80] + "..."),
        "expected_answer": expected_answer,
    }
