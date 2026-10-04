"""Phase 12.3 — numeric-equivalence layer for reasoning ground truth.

`evaluate_reasoning_answer` gains an opt-in `evaluator` keyword. When the scenario asks for
`"numeric"`, leading/trailing whitespace is trimmed and both sides are compared exactly as
`Fraction` values (so `"42"`, `"42.0"`, and `"  42  "` all match `"42"`). When the scenario asks
for `"numeric_text"`, English number words (`forty-two`, `five`, ...) are also converted before
the same numeric comparison. An unknown `evaluator` value is treated as absent and emits a
`evaluator.unknown` structured-log event.
"""

import unittest

from benchrig.core.reasoning_parser import evaluate_reasoning_answer


def _eval(response, expected, *, evaluator=None, check_type="numeric"):
    return evaluate_reasoning_answer(
        response_text=response,
        expected_answer=expected,
        check_type=check_type,
        accepted_patterns=None,
        evaluator=evaluator,
    )


class NumericEquivalenceTests(unittest.TestCase):
    # --- evaluator="numeric": tolerant whitespace + Fraction-exact numeric compare ---

    def test_numeric_matches_exact_integer(self):
        self.assertTrue(_eval("Final answer: 42", "42", evaluator="numeric")["correct"])

    def test_numeric_matches_decimal_variant(self):
        self.assertTrue(_eval("Final answer: 42.0", "42", evaluator="numeric")["correct"])

    def test_numeric_matches_whitespace_padded(self):
        self.assertTrue(_eval("Final answer:   42   ", "42", evaluator="numeric")["correct"])

    def test_numeric_matches_decimal_padded(self):
        self.assertTrue(_eval("Final answer:   42.0   ", "42", evaluator="numeric")["correct"])

    def test_numeric_does_not_match_text_answer(self):
        # Without "numeric_text", "forty-two" must NOT match "42".
        self.assertFalse(_eval("Final answer: forty-two", "42", evaluator="numeric")["correct"])

    def test_numeric_falls_through_when_one_side_not_numeric(self):
        # Three-boxes puzzle: expected "Box 1: Oranges" has no digits; numeric layer is a no-op
        # and the underlying check_type (final_answer) decides. With final_answer + no marker in
        # the response, the answer is rejected.
        result = evaluate_reasoning_answer(
            response_text="Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples",
            expected_answer="Box 1: Oranges",
            check_type="final_answer",
            accepted_patterns=[
                "box 1 (?:(?!apples|oranges)\\w+ ){0,3}oranges box 2 (?:(?!apples|oranges)\\w+ ){0,3}(?:apples and oranges|oranges and apples) box 3 (?:(?!apples|oranges)\\w+ ){0,3}apples(?: (?!and )|$)"
            ],
            evaluator="numeric",
        )
        self.assertFalse(result["correct"])

    # --- evaluator="numeric_text": English words also count ---

    def test_numeric_text_matches_text_to_digit(self):
        self.assertTrue(_eval("Final answer: forty-two", "42", evaluator="numeric_text")["correct"])

    def test_numeric_text_matches_digit_to_text(self):
        self.assertTrue(_eval("Final answer: 42", "forty-two", evaluator="numeric_text")["correct"])

    def test_numeric_text_case_sensitive(self):
        # "Forty-Two" must NOT match "forty-two" — no opt-in case-insensitive in this item.
        self.assertFalse(_eval("Final answer: Forty-Two", "forty-two", evaluator="numeric_text")["correct"])

    def test_numeric_text_other_words(self):
        cases = [
            ("five", "5"),
            ("twelve", "12"),
            ("twenty-one", "21"),
            ("ninety", "90"),
        ]
        for word, digit in cases:
            self.assertTrue(
                _eval(f"Final answer: {word}", digit, evaluator="numeric_text")["correct"],
                f"{word!r} should match {digit!r}",
            )

    # --- absent evaluator: behavior unchanged ---

    def test_absent_evaluator_legacy_decimal_path(self):
        # Pre-existing numeric path: float("42.0") == float("42") → True.
        self.assertTrue(_eval("Final answer: 42.0", "42")["correct"])

    def test_absent_evaluator_legacy_text_no_match(self):
        # Pre-existing numeric path: "forty-two" → ValueError → string compare → not equal.
        self.assertFalse(_eval("Final answer: forty-two", "42")["correct"])

    # --- Non-numeric ground truth: regex and exact_or_contains unaffected by evaluator ---

    def test_regex_scenario_unaffected_by_evaluator(self):
        text = "Conclusion:\n- **Box 1**: Oranges\n- **Box 2**: Apples and Oranges\n- **Box 3**: Apples"
        result = evaluate_reasoning_answer(
            response_text=text,
            expected_answer="Box 1: Oranges",
            check_type="regex",
            accepted_patterns=[
                "box 1 (?:(?!apples|oranges)\\w+ ){0,3}oranges box 2 (?:(?!apples|oranges)\\w+ ){0,3}(?:apples and oranges|oranges and apples) box 3 (?:(?!apples|oranges)\\w+ ){0,3}apples(?: (?!and )|$)"
            ],
            evaluator="numeric",
        )
        self.assertTrue(result["correct"])

    def test_exact_or_contains_scenario_unaffected_by_evaluator(self):
        # exact_or_contains with a non-numeric expected_answer keeps current behavior;
        # the numeric layer is a no-op because the expected answer has no digits.
        result = evaluate_reasoning_answer(
            response_text="The answer is Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples.",
            expected_answer="Box 1: Oranges",
            check_type="exact_or_contains",
            accepted_patterns=None,
            evaluator="numeric",
        )
        self.assertTrue(result["correct"])

    # --- Unknown evaluator value: fallback + structured-log event ---

    def test_unknown_evaluator_value_falls_back_and_logs(self):
        with self.assertLogs("benchrig.core.reasoning_parser", level="WARNING") as cm:
            result = evaluate_reasoning_answer(
                response_text="Final answer: 42.0",
                expected_answer="42",
                check_type="numeric",
                accepted_patterns=None,
                evaluator="not_a_real_value",
                scenario_id="reasoning_bat_ball",
            )
        # Underlying check_type is "numeric"; with no evaluator normalization, the legacy
        # float-compare path runs and 42.0 == 42 → True.
        self.assertTrue(result["correct"], "unknown evaluator must fall back to underlying check_type")
        # Assert the structured-log event has the right name AND the right field shape.
        matching = [r for r in cm.records if getattr(r, "event", None) == "evaluator.unknown"]
        self.assertEqual(len(matching), 1, f"expected one evaluator.unknown record, got {len(matching)}: {cm.output}")
        rec = matching[0]
        self.assertEqual(rec.evaluator_value, "not_a_real_value")
        self.assertEqual(rec.scenario_id, "reasoning_bat_ball")

    # --- Critical-path pins: the new layer is on the path, not a no-op ---

    def test_non_string_evaluator_value_falls_back_and_logs(self):
        """A non-string `evaluator` (e.g. an int) must also fall back to today's path and emit
        `evaluator.unknown` — the type contract is `str | None` and is enforced even though
        Python does not enforce type annotations at runtime.
        """
        with self.assertLogs("benchrig.core.reasoning_parser", level="WARNING") as cm:
            result = evaluate_reasoning_answer(
                response_text="Final answer: 42.0",
                expected_answer="42",
                check_type="numeric",
                accepted_patterns=None,
                evaluator=42,  # type: ignore[arg-type]
                scenario_id="reasoning_bat_ball",
            )
        self.assertTrue(result["correct"], "non-string evaluator must fall back to underlying check_type")
        matching = [r for r in cm.records if getattr(r, "event", None) == "evaluator.unknown"]
        self.assertEqual(len(matching), 1, f"expected one evaluator.unknown record, got {len(matching)}")
        self.assertEqual(matching[0].evaluator_value, 42)
        self.assertEqual(matching[0].scenario_id, "reasoning_bat_ball")

    def test_numeric_short_circuits_final_answer_with_empty_patterns(self):
        """A `final_answer` scenario with empty `accepted_patterns` would reject via the
        legacy path (no patterns to match). The numeric layer must accept on its own,
        proving the new branch is on the critical path rather than a no-op fallback.
        """
        result = evaluate_reasoning_answer(
            response_text="Final answer: 42",
            expected_answer="42",
            check_type="final_answer",
            accepted_patterns=[],
            evaluator="numeric",
        )
        self.assertTrue(result["correct"])

    def test_numeric_text_case_sensitive_direct_pin(self):
        """`Forty-Two` (capitalized) must NOT match `42` under `evaluator="numeric_text"`.

        Pinned directly with `expected_answer="42"` (a digit) so a regression that
        accidentally case-folds the English-number-word conversion would fail this test.
        """
        result = evaluate_reasoning_answer(
            response_text="Final answer: Forty-Two",
            expected_answer="42",
            check_type="numeric",
            accepted_patterns=None,
            evaluator="numeric_text",
        )
        self.assertFalse(result["correct"], "Forty-Two must NOT match 42 under numeric_text")

    def test_numeric_text_no_op_when_neither_side_is_numeric(self):
        """The `numeric_text` layer is a no-op when neither side parses as a number.
        Underlying `check_type="exact_or_contains"` decides; both sides have no digits.
        """
        result = evaluate_reasoning_answer(
            response_text="oranges",
            expected_answer="Box 1: Oranges",
            check_type="exact_or_contains",
            accepted_patterns=None,
            evaluator="numeric_text",
        )
        self.assertFalse(result["correct"], "non-numeric answer must not match via numeric_text no-op")
