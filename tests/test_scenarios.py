"""The bundled scenarios are checked against independent solvers, and the answer checkers against tricky answers."""

import json
import re
import unittest
from fractions import Fraction
from importlib.resources import files
from itertools import product

from benchrig.core.reasoning_parser import (
    contains_expected,
    evaluate_reasoning_answer,
    extract_final_answer,
    normalize_final_answer,
)

SCENARIO_DIR = files("benchrig").joinpath("data/scenarios")


def load(name):
    return json.loads(SCENARIO_DIR.joinpath(name).read_text(encoding="utf-8"))


def check(scenario, text):
    return evaluate_reasoning_answer(
        text, scenario["expected_answer"], scenario["check_type"], scenario.get("accepted_patterns")
    )["correct"]


def final(answer):
    return f"Some reasoning first.\nFinal answer: {answer}"


def solve_trains():
    start_b = 9 * 60
    gap_km = 432 - 72 * (start_b - (8 * 60 + 15)) / 60
    minutes = start_b + gap_km / (72 + 90) * 60
    return f"{int(minutes // 60):02d}:{int(round(minutes % 60)):02d}"


def solve_knights():
    solutions = [
        (a, b, c)
        for a, b, c in product([True, False], repeat=3)  # True = knight
        if (a == (not b)) and (b == c) and (c == (a != b))
    ]
    assert len(solutions) == 1, "the puzzle must have exactly one solution"
    return ", ".join(f"{name} is a {'knight' if v else 'knave'}" for name, v in zip("ABC", solutions[0], strict=True))


def solve_digits():
    return str(sum(1 for n in range(1000, 10000) if n % 5 == 0 and len(set(str(n))) == 4))


def solve_recurrence():
    a = 3
    for _ in range(9):
        a = 2 * a - 1
    return str(a)


def solve_price():
    return str(int(Fraction(80) * Fraction(125, 100) * Fraction(80, 100) * Fraction(110, 100)))


def solve_letters():
    return str("strawberry raspberry blueberry".count("r"))


def solve_bat_ball():
    (ball,) = [b for b in range(0, 111) if (b + 100) + b == 110]  # bat = ball + 100 cents, together 110 cents
    return str(ball)


def solve_multiples():
    return str(sum(1 for n in range(1, 101) if n % 3 == 0 or n % 5 == 0))


def solve_strawberry():
    return str("strawberry".count("r"))


SOLVERS = {
    "reasoning_trains": solve_trains,
    "reasoning_knights": solve_knights,
    "reasoning_distinct_digits": solve_digits,
    "reasoning_recurrence": solve_recurrence,
    "reasoning_price_chain": solve_price,
    "reasoning_letter_count": solve_letters,
    "reasoning_bat_ball": solve_bat_ball,
    "reasoning_multiples": solve_multiples,
    "reasoning_strawberry": solve_strawberry,
}


class ScenarioFileTests(unittest.TestCase):
    def test_every_scenario_has_what_its_suite_needs(self):
        for name, keys in (
            ("reasoning.json", {"id", "name", "prompt", "check_type", "expected_answer", "options"}),
            ("polish.json", {"id", "name", "prompt", "check_type", "expected_answer", "options"}),
            ("coding.json", {"id", "name", "prompt", "test_assertions", "options"}),
            ("speed.json", {"id", "name", "prompt"}),
        ):
            scenarios = load(name)
            ids = [s["id"] for s in scenarios]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in {name}")
            for sc in scenarios:
                self.assertTrue(keys <= sc.keys(), f"{name}:{sc.get('id')} misses {keys - sc.keys()}")
                for pattern in sc.get("accepted_patterns", []):
                    re.compile(pattern)

    def test_final_answer_scenarios_ask_for_the_marker(self):
        for sc in load("reasoning.json"):
            if sc["check_type"] == "final_answer":
                self.assertIn("Final answer:", sc["prompt"], sc["id"])
                self.assertTrue(sc["accepted_patterns"], sc["id"])

    def test_the_suite_has_twelve_reasoning_scenarios(self):
        self.assertEqual(len(load("reasoning.json")), 12)


class SolverAgreementTests(unittest.TestCase):
    """The expected answers come from independent code, so a wrong expectation cannot slip in as it once did."""

    def test_every_new_scenario_has_a_solver(self):
        new = {s["id"] for s in load("reasoning.json") if s["check_type"] == "final_answer"}
        self.assertEqual(new, set(SOLVERS))

    def test_the_solver_answer_is_accepted_and_matches_the_stated_expectation(self):
        for sc in load("reasoning.json"):
            solver = SOLVERS.get(sc["id"])
            if not solver:
                continue
            answer = solver()
            self.assertEqual(normalize_final_answer(sc["expected_answer"]).lower(), answer.lower(), sc["id"])
            self.assertTrue(check(sc, final(answer)), f"{sc['id']} rejects its own solution {answer!r}")

    def test_wrong_answers_are_rejected(self):
        wrong = {
            "reasoning_trains": ["11:30", "11:00", "12:20"],
            "reasoning_knights": [
                "A is a knight, B is a knave, C is a knight",
                "A is a knave, B is a knave, C is a knight",
            ],
            "reasoning_distinct_digits": ["900", "9520", "95"],
            "reasoning_recurrence": ["1024", "1023", "513"],
            "reasoning_price_chain": ["80", "89", "88.5"],
            "reasoning_letter_count": ["7", "9", "80"],
            "reasoning_bat_ball": ["10", "0.05", "15"],  # 10 cents is the classic wrong answer
            "reasoning_multiples": ["46", "48", "53", "33"],
            "reasoning_strawberry": ["2", "4", "10"],
        }
        by_id = {s["id"]: s for s in load("reasoning.json")}
        for sid, answers in wrong.items():
            for answer in answers:
                self.assertFalse(check(by_id[sid], final(answer)), f"{sid} accepted {answer!r}")

    def test_accepted_spellings(self):
        by_id = {s["id"]: s for s in load("reasoning.json")}
        cases = {
            "reasoning_trains": ["11:20", "**11:20**", "11:20 AM", "11.20"],
            "reasoning_knights": [
                "A: knave, B: knight, C: knight",
                "**A is a knave, B is a knight, C is a knight.**",
                "A knave, B knight, C knight",
            ],
            "reasoning_distinct_digits": ["952", "**952**", "$952$", "\\boxed{952}"],
            "reasoning_price_chain": ["88", "88.00", "€88", "88 euros", "$88"],
            "reasoning_recurrence": ["1025", "1,025"],
            "reasoning_bat_ball": ["5", "5 cents", "**5**"],
            "reasoning_multiples": ["47", "47."],
            "reasoning_strawberry": ["3", "**3**"],
        }
        for sid, answers in cases.items():
            for answer in answers:
                self.assertTrue(check(by_id[sid], final(answer)), f"{sid} rejects {answer!r}")


class FinalAnswerTests(unittest.TestCase):
    SC = {"check_type": "final_answer", "expected_answer": "952", "accepted_patterns": ["952"]}

    def test_marker_variants(self):
        for text in (
            "Final answer: 952",
            "**Final answer:** 952",
            "**Final Answer:**\n952",
            "final answer is 952.",
            "FINAL ANSWER = 952",
            "So it works.\n\nFinal answer: **952**\n",
        ):
            self.assertTrue(check(self.SC, text), text)

    def test_the_last_marker_counts(self):
        self.assertTrue(check(self.SC, "Final answer: 900\nOn reflection I was wrong.\nFinal answer: 952"))
        self.assertFalse(check(self.SC, "Final answer: 952\nHmm, actually.\nFinal answer: 900"))

    def test_no_marker_is_not_an_answer_even_if_the_number_appears(self):
        self.assertFalse(check(self.SC, "I think the number is 952."))

    def test_the_answer_must_match_fully(self):
        for text in ("Final answer: 9520", "Final answer: 952 or 953", "Final answer: about 952"):
            self.assertFalse(check(self.SC, text), text)

    def test_helpers(self):
        self.assertEqual(extract_final_answer("a\nFinal answer: 12\nb"), "12")
        self.assertIsNone(extract_final_answer("nothing here"))
        self.assertEqual(normalize_final_answer("**$1,025.**"), "1025")
        self.assertEqual(normalize_final_answer("\\boxed{\\dfrac{1}{6}}"), "1/6")


class NumberBoundaryTests(unittest.TestCase):
    def test_numbers_are_matched_as_whole_numbers(self):
        for haystack, pattern, expected in (
            ("the probability is 11/60", "1/6", False),
            ("21/60", "1/6", False),
            ("1/60", "1/6", False),
            ("1114.60", "114.6", False),
            ("1,114.6", "114.6", False),
            ("114.60", "114.6", False),
            ("the answer is 1/6.", "1/6", True),
            ("1/6", "1/6", True),
            ("total: $114.60", "$114.60", True),
            ("114.6.", "114.6", True),
            ("(114.6)", "114.6", True),
            ("0.1667 or so", "0.1667", True),
        ):
            self.assertEqual(contains_expected(haystack, pattern), expected, (haystack, pattern))

    def test_words_are_still_plain_substrings(self):
        self.assertTrue(contains_expected("do tego dużego sklepu", "dużego"))
        self.assertTrue(contains_expected("Zasoby i wątki", "wątk"))

    def test_the_bundled_probability_scenario_no_longer_accepts_lookalikes(self):
        venn = next(s for s in load("reasoning.json") if s["id"] == "reasoning_venn_probability")
        for text in ("It is 11/60", "The result is 1/60", "5/300"):
            self.assertFalse(check(venn, text), text)
        for text in ("The probability is 1/6.", "5/30 = 1/6", "\\boxed{\\dfrac{1}{6}}", "0.167"):
            self.assertTrue(check(venn, text), text)

    def test_the_bundled_tax_scenario_no_longer_accepts_1114(self):
        tax = next(s for s in load("reasoning.json") if s["id"] == "reasoning_discount_tax")
        self.assertFalse(check(tax, "The total is $1114.60"))
        self.assertTrue(check(tax, "The total is $114.60."))


if __name__ == "__main__":
    unittest.main()
