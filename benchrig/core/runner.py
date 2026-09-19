"""Benchmark execution engine that orchestrates test suites, hardware monitoring, and scoring."""

import time
from collections.abc import Callable, Iterable
from typing import Any

from benchrig.core.client import BaseRuntimeClient
from benchrig.core.hardware import HardwareSampler, get_system_specs
from benchrig.core.reasoning_parser import evaluate_reasoning_answer
from benchrig.core.sandbox import extract_python_code, run_code_with_tests

INTER_TEST_PAUSE_SEC = 0.5
SANDBOX_TIMEOUT_SEC = 6.0

# Generation speed considered "strong" (100%) for 7B-8B local models.
REFERENCE_TOK_PER_SEC = 60.0
# Composite-score penalty applied when a run tripped the VRAM warning threshold.
VRAM_WARNING_PENALTY = 20.0

# Frontier-model reference pricing used to estimate cloud spend avoided
# (Claude 3.5 Sonnet / GPT-4o class: $3.00 per 1M prompt, $15.00 per 1M completion tokens).
PROMPT_USD_PER_TOKEN = 3.00 / 1_000_000
COMPLETION_USD_PER_TOKEN = 15.00 / 1_000_000

DEFAULT_COMPOSITE_WEIGHTS = {"coding": 0.40, "reasoning": 0.30, "performance": 0.30}

# Filler text (computer architecture & operating systems) repeated to fill a context window.
CONTEXT_FILLER_PARAGRAPH = (
    "Modern operating systems rely on virtual memory to provide isolation, protection, "
    "and efficient memory utilization across processes. The memory management unit (MMU) "
    "translates virtual addresses into physical addresses using multi-level page tables. "
    "To accelerate translations, the CPU caches recent mappings in the Translation Lookaside "
    "Buffer (TLB). When a virtual page is not mapped to physical RAM, a page fault occurs, "
    "causing the OS kernel to page in the required data from swap or storage. "
    "Furthermore, modern GPUs utilize unified memory architectures and asynchronous copy engines "
    "to maximize bandwidth across high-speed PCIe and NVLink interconnects. "
)


class BenchmarkRunner:
    """Coordinates benchmark suites, gathers metrics, and calculates scores."""

    def __init__(
        self,
        client: BaseRuntimeClient,
        config: dict[str, Any],
        progress_callback: Callable[[str, str], None] | None = None,
    ):
        self.client = client
        self.config = config
        self.progress_callback = progress_callback
        self.specs = get_system_specs()

    def _notify(self, status: str, detail: str = ""):
        if self.progress_callback:
            self.progress_callback(status, detail)

    def _create_sampler(self) -> HardwareSampler:
        """Create HardwareSampler configured with client and dynamic thresholds."""
        hw_conf = self.config.get("hardware", {})
        interval = hw_conf.get("sample_interval_sec", 0.15)
        threshold = hw_conf.get("vram_warning_threshold_mb", "auto")
        return HardwareSampler(
            interval_sec=interval,
            client=self.client,
            warning_threshold_setting=threshold,
        )

    def _generate_with_telemetry(
        self, model: str, prompt: str, options: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one streamed generation while sampling hardware; return (response, hardware)."""
        sampler = self._create_sampler()
        sampler.start()
        resp = self.client.generate(model=model, prompt=prompt, options=options, measure_ttft=True)
        return resp, sampler.stop()

    def _base_record(
        self,
        suite: str,
        test_id: str,
        name: str,
        model: str,
        resp: dict[str, Any],
        hardware: dict[str, Any],
    ) -> dict[str, Any]:
        """Fields shared by every result record, regardless of suite."""
        return {
            "suite": suite,
            "test_id": test_id,
            "name": name,
            "model": model,
            "runtime": getattr(self.client, "name", "ollama"),
            "engine": getattr(self.client, "engine_name", "llama.cpp"),
            "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
            "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
            "ttft_sec": resp.get("ttft_sec", 0.0),
            "prompt_eval_count": resp.get("prompt_eval_count", 0),
            "hardware": hardware,
        }

    def _run_suite(
        self,
        suite: str,
        label: str,
        model: str,
        scenarios: list[dict[str, Any]],
        score: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Run each scenario's prompt and build one result record per scenario.

        ``score(scenario, response)`` returns the suite-specific fields; it must include
        ``success`` (already combined with ``response["success"]``).
        """
        results = []
        for sc in scenarios:
            self._notify(label, f"[{model}] {sc['name']}")
            resp, hw = self._generate_with_telemetry(model, sc["prompt"], sc.get("options", {}))

            record = self._base_record(suite, sc["id"], sc["name"], model, resp, hw)
            record["eval_count"] = resp.get("eval_count", 0)
            record.update(score(sc, resp))
            results.append(record)
            time.sleep(INTER_TEST_PAUSE_SEC)
        return results

    @staticmethod
    def _evaluate_answer(sc: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
        """Grade a free-text answer against the scenario's expected answer."""
        return evaluate_reasoning_answer(
            response_text=resp.get("response", ""),
            expected_answer=sc.get("expected_answer", ""),
            check_type=sc.get("check_type", "exact_or_contains"),
            accepted_patterns=sc.get("accepted_patterns", []),
        )

    def warmup(self, model: str):
        """Warm up model to ensure weights are placed in VRAM."""
        self._notify("Warmup", f"Warming up {model}...")
        self.client.generate(
            model=model,
            prompt="Hello, respond with OK.",
            options={"num_predict": 5},
            measure_ttft=False,
        )

    def run_speed_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run raw speed & throughput scenarios."""
        return self._run_suite(
            "speed",
            "Speed Test",
            model,
            scenarios,
            lambda sc, resp: {
                "success": resp["success"],
                "total_time_sec": resp.get("total_time_sec", 0.0),
            },
        )

    def run_coding_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run coding benchmark scenarios with automated unit test assertions."""

        def score(sc: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
            test_res = run_code_with_tests(
                solution_code=extract_python_code(resp.get("response", "")),
                test_assertions=sc.get("test_assertions", []),
                timeout_sec=SANDBOX_TIMEOUT_SEC,
            )
            return {
                "success": resp["success"] and test_res["passed"],
                "passed": test_res["passed"],
                "passed_tests": test_res["passed_tests"],
                "total_tests": test_res["total_tests"],
                "pass_ratio": test_res["pass_ratio"],
                "sandbox_error": test_res.get("error", ""),
            }

        return self._run_suite("coding", "Coding Test", model, scenarios, score)

    def run_reasoning_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run reasoning scenarios and evaluate against expected ground truth."""

        def score(sc: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
            ev = self._evaluate_answer(sc, resp)
            return {
                "success": resp["success"] and ev["correct"],
                "correct": ev["correct"],
                "has_think_tags": ev["has_think_tags"],
                "think_chars": ev["think_chars"],
                "extracted_answer": ev["extracted_answer"],
                "expected_answer": ev["expected_answer"],
            }

        return self._run_suite("reasoning", "Reasoning Test", model, scenarios, score)

    def run_polish_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run Polish NLP comprehension and grammar scenarios."""

        def score(sc: dict[str, Any], resp: dict[str, Any]) -> dict[str, Any]:
            ev = self._evaluate_answer(sc, resp)
            return {
                "success": resp["success"] and ev["correct"],
                "correct": ev["correct"],
                "extracted_answer": ev["extracted_answer"],
                "expected_answer": ev["expected_answer"],
            }

        return self._run_suite("polish", "Polish Test", model, scenarios, score)

    def run_context_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run context scaling scenarios (512 to 8k) to check VRAM saturation."""
        results = []
        for sc in scenarios:
            ctx_size = sc["context_size"]
            multiplier = sc.get("prompt_multiplier", 1)
            # Repeat paragraph to fill the requested context window
            repeated_text = (CONTEXT_FILLER_PARAGRAPH * multiplier)[: ctx_size * 3]
            prompt = f"{repeated_text}\n\nQuestion: {sc['instruction']}"

            self._notify("Context Scaling", f"[{model}] {sc['name']} (num_ctx: {ctx_size})")
            resp, hw = self._generate_with_telemetry(
                model,
                prompt,
                {"num_ctx": ctx_size, "num_predict": 128, "temperature": 0.2},
            )

            record = self._base_record("context", f"context_{ctx_size}", sc["name"], model, resp, hw)
            record["context_size"] = ctx_size
            record["success"] = resp["success"]
            results.append(record)
            time.sleep(INTER_TEST_PAUSE_SEC)
        return results

    def compute_model_scorecard(
        self,
        model: str,
        all_test_results: list[dict[str, Any]],
        runtime: str | None = None,
    ) -> dict[str, Any]:
        """Compute aggregated scorecard and composite score for a model."""
        model_results = [
            r
            for r in all_test_results
            if r["model"] == model and (runtime is None or r.get("runtime", "ollama") == runtime)
        ]
        if not model_results:
            return {}

        sc_runtime = model_results[0].get("runtime", getattr(self.client, "name", "ollama"))
        sc_engine = model_results[0].get("engine", getattr(self.client, "engine_name", "llama.cpp"))

        # 1. Performance metrics (only records that actually generated tokens)
        speed_records = [r for r in model_results if r.get("eval_tok_per_sec", 0) > 0]
        avg_eval_tok_sec = _mean(r["eval_tok_per_sec"] for r in speed_records)
        avg_prompt_tok_sec = _mean(r["prompt_tok_per_sec"] for r in speed_records)
        avg_ttft = _mean(r["ttft_sec"] for r in speed_records)

        # Hardware metrics
        peak_vram = max((r.get("hardware", {}).get("vram_peak_mb", 0.0) for r in model_results), default=0.0)
        vram_warning = any(r.get("hardware", {}).get("vram_warning", False) for r in model_results)
        total_vram = max((r.get("hardware", {}).get("vram_total_mb", 0.0) for r in model_results), default=0.0)
        if total_vram == 0.0 and self.specs.get("gpu_vram_total_mb"):
            try:
                total_vram = float(self.specs["gpu_vram_total_mb"])
            except ValueError:
                total_vram = 0.0

        # 2. Coding metrics
        coding_records = [r for r in model_results if r.get("suite") == "coding"]
        total_coding_tests = sum(r.get("total_tests", 0) for r in coding_records)
        passed_coding_tests = sum(r.get("passed_tests", 0) for r in coding_records)
        coding_pass_rate = _percent(passed_coding_tests, total_coding_tests)

        # 3. Reasoning metrics
        reasoning_records = [r for r in model_results if r.get("suite") == "reasoning"]
        correct_reasoning = sum(1 for r in reasoning_records if r.get("correct", False))
        reasoning_accuracy = _percent(correct_reasoning, len(reasoning_records))

        # 4. Normalized speed & VRAM efficiency (0 - 100)
        speed_factor = min(100.0, (avg_eval_tok_sec / REFERENCE_TOK_PER_SEC) * 100.0)
        vram_penalty = VRAM_WARNING_PENALTY if vram_warning else 0.0
        efficiency_score = max(0.0, speed_factor - vram_penalty)

        # Composite score
        weights = self.config.get("benchmark", {}).get("composite_weights", DEFAULT_COMPOSITE_WEIGHTS)
        composite_score = (
            coding_pass_rate * weights.get("coding", DEFAULT_COMPOSITE_WEIGHTS["coding"])
            + reasoning_accuracy * weights.get("reasoning", DEFAULT_COMPOSITE_WEIGHTS["reasoning"])
            + efficiency_score * weights.get("performance", DEFAULT_COMPOSITE_WEIGHTS["performance"])
        )

        # 5. Token savings & cloud cost estimation
        total_eval_tokens = sum(r.get("eval_count", 0) for r in model_results)
        total_prompt_tokens = sum(r.get("prompt_eval_count", 0) for r in model_results)
        est_cost_saved_usd = total_prompt_tokens * PROMPT_USD_PER_TOKEN + total_eval_tokens * COMPLETION_USD_PER_TOKEN

        return {
            "model": model,
            "runtime": sc_runtime,
            "engine": sc_engine,
            "composite_score": round(composite_score, 1),
            "coding_pass_rate": round(coding_pass_rate, 1),
            "reasoning_accuracy": round(reasoning_accuracy, 1),
            "avg_eval_tok_sec": round(avg_eval_tok_sec, 1),
            "avg_prompt_tok_sec": round(avg_prompt_tok_sec, 1),
            "avg_ttft_sec": round(avg_ttft, 2),
            "peak_vram_mb": round(peak_vram, 1),
            "total_vram_mb": round(total_vram, 1),
            "total_eval_tokens": total_eval_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_tokens_saved": total_eval_tokens + total_prompt_tokens,
            "est_cost_saved_usd": round(est_cost_saved_usd, 4),
            "memory_type": self.specs.get("memory_type", "VRAM"),
            "gpu_type": self.specs.get("gpu_type", "unknown"),
            "vram_warning": vram_warning,
            "total_runs": len(model_results),
        }


def _mean(values: Iterable[float]) -> float:
    """Arithmetic mean, or 0.0 for an empty iterable."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percent(part: float, whole: float) -> float:
    """``part`` as a percentage of ``whole``, or 0.0 when ``whole`` is zero."""
    return part / whole * 100.0 if whole > 0 else 0.0
