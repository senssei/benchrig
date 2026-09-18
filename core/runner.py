"""Benchmark execution engine that orchestrates test suites, hardware monitoring, and scoring."""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from core.client import OllamaClient
from core.hardware import HardwareSampler, get_system_specs
from core.reasoning_parser import evaluate_reasoning_answer
from core.sandbox import extract_python_code, run_code_with_tests


class BenchmarkRunner:
    """Coordinates benchmark suites, gathers metrics, and calculates scores."""

    def __init__(
        self,
        client: OllamaClient,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, str], None]] = None,
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

    def warmup(self, model: str):
        """Warm up model to ensure weights are placed in VRAM."""
        self._notify("Warmup", f"Warming up {model}...")
        self.client.generate(
            model=model,
            prompt="Hello, respond with OK.",
            options={"num_predict": 5},
            measure_ttft=False,
        )

    def run_speed_suite(self, model: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run raw speed & throughput scenarios."""
        results = []
        for sc in scenarios:
            self._notify("Speed Test", f"[{model}] {sc['name']}")
            sampler = self._create_sampler()
            sampler.start()

            resp = self.client.generate(
                model=model,
                prompt=sc["prompt"],
                options=sc.get("options", {}),
                measure_ttft=True,
            )
            hw = sampler.stop()

            results.append({
                "suite": "speed",
                "test_id": sc["id"],
                "name": sc["name"],
                "model": model,
                "success": resp["success"],
                "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
                "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
                "ttft_sec": resp.get("ttft_sec", 0.0),
                "eval_count": resp.get("eval_count", 0),
                "total_time_sec": resp.get("total_time_sec", 0.0),
                "hardware": hw,
            })
            time.sleep(0.5)
        return results

    def run_coding_suite(self, model: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run coding benchmark scenarios with automated unit test assertions."""
        results = []
        for sc in scenarios:
            self._notify("Coding Test", f"[{model}] {sc['name']}")
            sampler = self._create_sampler()
            sampler.start()

            resp = self.client.generate(
                model=model,
                prompt=sc["prompt"],
                options=sc.get("options", {}),
                measure_ttft=True,
            )
            hw = sampler.stop()

            extracted_code = extract_python_code(resp.get("response", ""))
            test_res = run_code_with_tests(
                solution_code=extracted_code,
                test_assertions=sc.get("test_assertions", []),
                timeout_sec=6.0,
            )

            results.append({
                "suite": "coding",
                "test_id": sc["id"],
                "name": sc["name"],
                "model": model,
                "success": resp["success"] and test_res["passed"],
                "passed": test_res["passed"],
                "passed_tests": test_res["passed_tests"],
                "total_tests": test_res["total_tests"],
                "pass_ratio": test_res["pass_ratio"],
                "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
                "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
                "ttft_sec": resp.get("ttft_sec", 0.0),
                "eval_count": resp.get("eval_count", 0),
                "sandbox_error": test_res.get("error", ""),
                "hardware": hw,
            })
            time.sleep(0.5)
        return results

    def run_reasoning_suite(self, model: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run reasoning scenarios and evaluate against expected ground truth."""
        results = []
        for sc in scenarios:
            self._notify("Reasoning Test", f"[{model}] {sc['name']}")
            sampler = self._create_sampler()
            sampler.start()

            resp = self.client.generate(
                model=model,
                prompt=sc["prompt"],
                options=sc.get("options", {}),
                measure_ttft=True,
            )
            hw = sampler.stop()

            eval_res = evaluate_reasoning_answer(
                response_text=resp.get("response", ""),
                expected_answer=sc.get("expected_answer", ""),
                check_type=sc.get("check_type", "exact_or_contains"),
                accepted_patterns=sc.get("accepted_patterns", []),
            )

            results.append({
                "suite": "reasoning",
                "test_id": sc["id"],
                "name": sc["name"],
                "model": model,
                "success": resp["success"] and eval_res["correct"],
                "correct": eval_res["correct"],
                "has_think_tags": eval_res["has_think_tags"],
                "think_chars": eval_res["think_chars"],
                "extracted_answer": eval_res["extracted_answer"],
                "expected_answer": eval_res["expected_answer"],
                "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
                "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
                "ttft_sec": resp.get("ttft_sec", 0.0),
                "eval_count": resp.get("eval_count", 0),
                "hardware": hw,
            })
            time.sleep(0.5)
        return results

    def run_polish_suite(self, model: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run Polish NLP comprehension and grammar scenarios."""
        results = []
        for sc in scenarios:
            self._notify("Polish Test", f"[{model}] {sc['name']}")
            sampler = self._create_sampler()
            sampler.start()

            resp = self.client.generate(
                model=model,
                prompt=sc["prompt"],
                options=sc.get("options", {}),
                measure_ttft=True,
            )
            hw = sampler.stop()

            eval_res = evaluate_reasoning_answer(
                response_text=resp.get("response", ""),
                expected_answer=sc.get("expected_answer", ""),
                check_type=sc.get("check_type", "exact_or_contains"),
                accepted_patterns=sc.get("accepted_patterns", []),
            )

            results.append({
                "suite": "polish",
                "test_id": sc["id"],
                "name": sc["name"],
                "model": model,
                "success": resp["success"] and eval_res["correct"],
                "correct": eval_res["correct"],
                "extracted_answer": eval_res["extracted_answer"],
                "expected_answer": eval_res["expected_answer"],
                "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
                "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
                "ttft_sec": resp.get("ttft_sec", 0.0),
                "eval_count": resp.get("eval_count", 0),
                "hardware": hw,
            })
            time.sleep(0.5)
        return results

    def run_context_suite(self, model: str, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run context scaling scenarios (512 to 8k) to check VRAM saturation."""
        results = []
        # Sample base context text (computer architecture & operating systems)
        base_paragraph = (
            "Modern operating systems rely on virtual memory to provide isolation, protection, "
            "and efficient memory utilization across processes. The memory management unit (MMU) "
            "translates virtual addresses into physical addresses using multi-level page tables. "
            "To accelerate translations, the CPU caches recent mappings in the Translation Lookaside "
            "Buffer (TLB). When a virtual page is not mapped to physical RAM, a page fault occurs, "
            "causing the OS kernel to page in the required data from swap or storage. "
            "Furthermore, modern GPUs utilize unified memory architectures and asynchronous copy engines "
            "to maximize bandwidth across high-speed PCIe and NVLink interconnects. "
        )

        for sc in scenarios:
            ctx_size = sc["context_size"]
            multiplier = sc.get("prompt_multiplier", 1)
            # Repeat paragraph to fill the requested context window
            repeated_text = (base_paragraph * multiplier)[: ctx_size * 3]
            prompt = f"{repeated_text}\n\nQuestion: {sc['instruction']}"

            self._notify("Context Scaling", f"[{model}] {sc['name']} (num_ctx: {ctx_size})")
            sampler = self._create_sampler()
            sampler.start()

            resp = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_ctx": ctx_size,
                    "num_predict": 128,
                    "temperature": 0.2,
                },
                measure_ttft=True,
            )
            hw = sampler.stop()

            results.append({
                "suite": "context",
                "test_id": f"context_{ctx_size}",
                "name": sc["name"],
                "context_size": ctx_size,
                "model": model,
                "success": resp["success"],
                "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
                "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
                "ttft_sec": resp.get("ttft_sec", 0.0),
                "prompt_eval_count": resp.get("prompt_eval_count", 0),
                "hardware": hw,
            })
            time.sleep(0.5)
        return results

    def compute_model_scorecard(self, model: str, all_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated scorecard and composite score for a model."""
        model_results = [r for r in all_test_results if r["model"] == model]
        if not model_results:
            return {}

        # 1. Performance metrics
        speed_records = [r for r in model_results if "eval_tok_per_sec" in r and r["eval_tok_per_sec"] > 0]
        avg_eval_tok_sec = (
            sum(r["eval_tok_per_sec"] for r in speed_records) / len(speed_records)
            if speed_records
            else 0.0
        )
        avg_prompt_tok_sec = (
            sum(r["prompt_tok_per_sec"] for r in speed_records) / len(speed_records)
            if speed_records
            else 0.0
        )
        avg_ttft = (
            sum(r["ttft_sec"] for r in speed_records) / len(speed_records)
            if speed_records
            else 0.0
        )

        # Hardware metrics
        peak_vram = max(
            [r.get("hardware", {}).get("vram_peak_mb", 0.0) for r in model_results] or [0.0]
        )
        vram_warning = any(
            r.get("hardware", {}).get("vram_warning", False) for r in model_results
        )

        # 2. Coding metrics
        coding_records = [r for r in model_results if r.get("suite") == "coding"]
        total_coding_tests = sum(r.get("total_tests", 0) for r in coding_records)
        passed_coding_tests = sum(r.get("passed_tests", 0) for r in coding_records)
        coding_pass_rate = (
            (passed_coding_tests / total_coding_tests * 100.0)
            if total_coding_tests > 0
            else 0.0
        )

        # 3. Reasoning metrics
        reasoning_records = [r for r in model_results if r.get("suite") == "reasoning"]
        total_reasoning = len(reasoning_records)
        correct_reasoning = sum(1 for r in reasoning_records if r.get("correct", False))
        reasoning_accuracy = (
            (correct_reasoning / total_reasoning * 100.0)
            if total_reasoning > 0
            else 0.0
        )

        # 4. Normalized Speed & VRAM Efficiency (0 - 100)
        # Reference: 60 tok/s is considered a strong baseline (100%) for 7B-8B local models
        speed_factor = min(100.0, (avg_eval_tok_sec / 60.0) * 100.0)
        # VRAM efficiency: if vram_warning is true, penalize 20 points
        vram_penalty = 20.0 if vram_warning else 0.0
        efficiency_score = max(0.0, speed_factor - vram_penalty)

        # Composite score
        weights = self.config.get("benchmark", {}).get("composite_weights", {
            "coding": 0.40,
            "reasoning": 0.30,
            "performance": 0.30,
        })

        composite_score = (
            (coding_pass_rate * weights.get("coding", 0.40))
            + (reasoning_accuracy * weights.get("reasoning", 0.30))
            + (efficiency_score * weights.get("performance", 0.30))
        )

        total_vram = max(
            [r.get("hardware", {}).get("vram_total_mb", 0.0) for r in model_results] or [0.0]
        )
        if total_vram == 0.0 and self.specs.get("gpu_vram_total_mb"):
            try:
                total_vram = float(self.specs.get("gpu_vram_total_mb", 0.0))
            except ValueError:
                total_vram = 0.0

        return {
            "model": model,
            "composite_score": round(composite_score, 1),
            "coding_pass_rate": round(coding_pass_rate, 1),
            "reasoning_accuracy": round(reasoning_accuracy, 1),
            "avg_eval_tok_sec": round(avg_eval_tok_sec, 1),
            "avg_prompt_tok_sec": round(avg_prompt_tok_sec, 1),
            "avg_ttft_sec": round(avg_ttft, 2),
            "peak_vram_mb": round(peak_vram, 1),
            "total_vram_mb": round(total_vram, 1),
            "memory_type": self.specs.get("memory_type", "VRAM"),
            "gpu_type": self.specs.get("gpu_type", "unknown"),
            "vram_warning": vram_warning,
            "total_runs": len(model_results),
        }
