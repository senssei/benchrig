"""Benchmark execution engine that orchestrates test suites, hardware monitoring, and scoring."""

import time
from collections.abc import Callable, Iterable
from typing import Any

from benchrig.core.client import BaseRuntimeClient
from benchrig.core.hardware import HardwareSampler, get_system_specs
from benchrig.core.reasoning_parser import contains_expected, evaluate_reasoning_answer
from benchrig.core.sandbox import extract_python_code, has_complete_code_block, run_code_with_tests

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

# Models that spend their token budget on a `<think>` trace before answering get a larger `num_predict`
# (suites that need a complete answer only), otherwise the answer or the code is cut off and scored as wrong.
DEFAULT_THINKING_MODELS = ("deepseek-r1", "qwq", "magistral", "qwen3")
# Measured on deepseek-r1:14b coding (one run each): x6 cut off 1-2 of 4 scenarios, x12 cut off 1, x20 cut off none (the scenario
# that loops at x12 finished in 4609 tokens at x20); the real need is a long tail, so the default leaves room above 4.6k tokens.
DEFAULT_THINKING_TOKEN_MULTIPLIER = 12
# Ollama `think` per suite for models that can think (None = the model's own default). Speed and context measure throughput
# and prompt processing, so thinking is off there: otherwise a thinking model's latency is mostly thinking time.
DEFAULT_THINK_BY_SUITE: dict[str, bool | None] = {"speed": False, "context": False}
# Suites where the answer is scored, so it must be complete. `speed` and `context` generate a fixed number of tokens on
# purpose, so stopping at the token limit there is expected and is not reported as truncation.
ANSWER_SUITES = ("coding", "reasoning", "polish")

# Context suite: prompts are sized from the context window (`num_ctx`) so the label matches what is sent.
CONTEXT_CHARS_PER_TOKEN = 4  # rough English estimate; the real count is recorded as `prompt_tokens_actual`
DEFAULT_CONTEXT_FILL_RATIO = 0.75  # leaves room for the answer and the instruction inside the window
CONTEXT_WARMUP_PROMPT = "hi"

# Baseline VRAM: wait until the reading stops moving (a previous model may still be unloading).
VRAM_SETTLE_TOLERANCE_MB = 50.0
VRAM_SETTLE_TIMEOUT_SEC = 10.0
VRAM_SETTLE_POLL_SEC = 0.5

# Scorecard metrics whose min-max over repeated runs (`--runs N`) is reported.
SPREAD_METRICS = ("composite_score", "coding_pass_rate", "reasoning_accuracy", "avg_eval_tok_sec", "avg_ttft_sec")

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


def build_context_prompt(
    ctx_size: int,
    instruction: str,
    fill_ratio: float = DEFAULT_CONTEXT_FILL_RATIO,
    needle: str | None = None,
    depth: float = 0.5,
) -> str:
    """Prompt of roughly `ctx_size * fill_ratio` tokens: the filler paragraph repeated, then the instruction.

    With a `needle` (a fact the question asks about) the sentence is placed at `depth` (0 = start, 1 = end) of the filler,
    at a sentence boundary, so the model can only answer by using the whole context.
    """
    target_chars = max(1, int(ctx_size * fill_ratio * CONTEXT_CHARS_PER_TOKEN))
    repeats = target_chars // len(CONTEXT_FILLER_PARAGRAPH) + 1
    filler = (CONTEXT_FILLER_PARAGRAPH * repeats)[:target_chars]
    if needle:
        at = filler.find(". ", int(len(filler) * min(max(depth, 0.0), 1.0)))
        at = len(filler) if at == -1 else at + 2
        filler = f"{filler[:at]}{needle} {filler[at:]}"
    return f"{filler}\n\nQuestion: {instruction}"


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
        self.vram_baseline_mb: float | None = None  # GPU memory in use before the model is loaded
        # True when a model was still loaded at that moment: the baseline then contains model memory (Prism, which has no
        # unload, still holds the previous model), so peak minus baseline would understate this model's footprint.
        self.vram_baseline_dirty = False
        self.run_index = 0  # repetition (`--runs N`), set by the caller; 0 is the first
        self.cold_start_sec: float | None = (
            None  # duration of the first (warm-up) request, which includes loading the model
        )
        self.gpu_fit_pct: float | None = None  # share of the loaded model that is in GPU memory (Ollama only)

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

    def measure_vram_baseline(self) -> float:
        """Record GPU memory in use *before* the model is loaded, so reports can show the model's own footprint.

        Waits until two consecutive readings agree (a previous model may still be unloading) or the timeout passes.
        Whatever else is using the GPU (a desktop, other model servers, the WSL2 host) is part of this baseline.
        """
        provider = self._create_sampler().provider
        previous = provider.read_gpu()[0]
        deadline = time.monotonic() + VRAM_SETTLE_TIMEOUT_SEC
        while time.monotonic() < deadline:
            time.sleep(VRAM_SETTLE_POLL_SEC)
            current = provider.read_gpu()[0]
            settled = abs(current - previous) <= VRAM_SETTLE_TOLERANCE_MB
            previous = current
            if settled:
                break
        self.vram_baseline_mb = previous
        try:
            self.vram_baseline_dirty = bool(self.client.get_running_models())
        except Exception:
            self.vram_baseline_dirty = False
        return previous

    def _sampling_defaults(self) -> dict[str, Any]:
        """`execution_alignment_1to1` as default request options, so every runtime samples the same way.

        The seed advances with the repetition, so `--runs 3` gives three different (but reproducible) samples instead of the
        same one three times.
        """
        aligned = self.config.get("execution_alignment_1to1") or {}
        defaults = {key: aligned[key] for key in ("temperature", "top_p") if key in aligned}
        if "seed" in aligned:
            defaults["seed"] = aligned["seed"] + self.run_index
        if "context_tokens" in aligned:
            defaults["num_ctx"] = aligned["context_tokens"]
        return defaults

    def _with_sampling(self, options: dict[str, Any]) -> dict[str, Any]:
        """Scenario options over the aligned defaults (a scenario's own setting always wins)."""
        return {**self._sampling_defaults(), **options}

    def _for_run(self, prompt: str) -> str:
        """Prompt for the current repetition.

        Ollama caches the prompt prefix between requests (a repeated prompt measured 0.006 s of prefill instead of 0.12 s),
        which would make repetitions 2..N look faster than the first. A per-repetition marker at the start changes the
        very first token and so defeats the cache. The first repetition is sent unchanged.
        """
        return prompt if self.run_index == 0 else f"(request {self.run_index + 1})\n{prompt}"

    def _apply_think(self, suite: str, options: dict[str, Any]) -> dict[str, Any]:
        """Add the suite's `think` setting (`benchmark.think`); a scenario's own `think` option wins."""
        if "think" in options:
            return options
        think = {**DEFAULT_THINK_BY_SUITE, **(self.config.get("benchmark", {}).get("think") or {})}.get(suite)
        return options if think is None else {**options, "think": think}

    def _effective_options(self, suite: str, model: str, options: dict[str, Any]) -> dict[str, Any]:
        """Scenario options, with the token budget raised for thinking models on suites that need a full answer."""
        options = self._apply_think(suite, self._with_sampling(options))
        bench = self.config.get("benchmark", {})
        patterns = bench.get("thinking_models", DEFAULT_THINKING_MODELS)
        multiplier = bench.get("thinking_token_multiplier", DEFAULT_THINKING_TOKEN_MULTIPLIER)
        is_thinking = any(str(p).lower() in model.lower() for p in patterns)
        # Applied even with `think: false`: deepseek-r1:14b with thinking off still ran into the base budget (0 of 4 tests, the
        # 512-token limit reached without a complete code block), so a thinking-capable model keeps the larger budget.
        if not (is_thinking and suite in ANSWER_SUITES and options.get("num_predict") and multiplier > 1):
            return options
        return {**options, "num_predict": int(options["num_predict"] * multiplier)}

    @staticmethod
    def _is_truncated(resp: dict[str, Any], options: dict[str, Any]) -> bool:
        """True if generation stopped because the token budget ran out, not because the model finished."""
        reason = resp.get("finish_reason")
        if reason:
            return reason == "length"
        budget = options.get("num_predict") or options.get("max_tokens")  # engines that report no finish reason
        return bool(budget) and resp.get("eval_count", 0) >= budget

    def _generate_with_telemetry(
        self, model: str, prompt: str, options: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one streamed generation while sampling hardware; return (response, hardware)."""
        sampler = self._create_sampler()
        sampler.start()
        resp = self.client.generate(model=model, prompt=self._for_run(prompt), options=options, measure_ttft=True)
        hardware = sampler.stop()
        if self.vram_baseline_mb is not None:
            hardware["vram_baseline_mb"] = round(self.vram_baseline_mb, 1)
            if self.vram_baseline_dirty:
                hardware["vram_baseline_dirty"] = True  # no model-only figure: it would be too low
            else:
                hardware["vram_model_mb"] = round(
                    max(0.0, hardware.get("vram_peak_mb", 0.0) - self.vram_baseline_mb), 1
                )
        return resp, hardware

    def _base_record(
        self,
        suite: str,
        test_id: str,
        name: str,
        model: str,
        resp: dict[str, Any],
        hardware: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fields shared by every result record, regardless of suite."""
        ttft = resp.get("ttft_sec", 0.0)
        prompt_tokens = resp.get("prompt_eval_count", 0)
        record = {
            "suite": suite,
            "test_id": test_id,
            "name": name,
            "model": model,
            "runtime": getattr(self.client, "name", "ollama"),
            # The engine that served this model (a multi-engine server such as Prism reports it per model).
            "engine": resp.get("engine") or getattr(self.client, "engine_name", "llama.cpp"),
            "eval_tok_per_sec": resp.get("eval_tok_per_sec", 0.0),
            "prompt_tok_per_sec": resp.get("prompt_tok_per_sec", 0.0),
            # Prefill computed the same way for every runtime (prompt tokens / time to first token, so it includes
            # request overhead); `prompt_tok_per_sec` is what the engine itself reports where it reports one.
            "prefill_eff_tok_per_sec": round(prompt_tokens / ttft, 2) if ttft > 0 else 0.0,
            "prefill_source": getattr(self.client, "prefill_source", "client_ttft"),
            "run": self.run_index,
            "ttft_sec": ttft,
            "prompt_eval_count": prompt_tokens,
            "load_time_sec": resp.get("load_time_sec", 0.0),
            "finish_reason": resp.get("finish_reason"),
            "truncated": suite in ANSWER_SUITES and self._is_truncated(resp, options or {}),
            "hardware": hardware,
        }
        if resp.get("device"):  # execution device reported by the server (e.g. Prism: "cuda" / "cpu")
            record["device"] = resp["device"]
        if self.cold_start_sec is not None:
            record["cold_start_sec"] = self.cold_start_sec
        if self.gpu_fit_pct is not None:
            record["gpu_fit_pct"] = self.gpu_fit_pct
        power = hardware.get("power_avg_w", 0.0) if isinstance(hardware, dict) else 0.0
        if power and power > 0 and resp.get("eval_tok_per_sec", 0) > 0:
            record["tokens_per_joule"] = round(resp["eval_tok_per_sec"] / power, 3)  # (tokens/s) / (J/s)
        if resp.get(
            "usage_estimated"
        ):  # token counts (and so prefill and decode speeds) are estimates, not the server's
            record["usage_estimated"] = True
        if resp.get("error"):  # the request itself failed (as opposed to a wrong answer)
            record["error"] = str(resp["error"])
        if resp.get("think") is not None:  # the `think` setting that was sent (Ollama, thinking-capable models)
            record["think"] = resp["think"]
        if resp.get("thinking_chars"):
            record["thinking_chars"] = resp["thinking_chars"]
            if resp.get("answer_ttft_sec") is not None:
                record["answer_ttft_sec"] = resp["answer_ttft_sec"]
                record["think_time_sec"] = round(max(0.0, resp["answer_ttft_sec"] - ttft), 3)
        return record

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
            options = self._effective_options(suite, model, sc.get("options", {}))
            resp, hw = self._generate_with_telemetry(model, sc["prompt"], options)

            record = self._base_record(suite, sc["id"], sc["name"], model, resp, hw, options)
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
        """Warm up model to ensure weights are placed in VRAM.

        The duration of this first request is the cold start (it includes loading the model when it was not resident), and
        right after it is the moment to ask the runtime how much of the model sits in GPU memory.
        """
        self._notify("Warmup", f"Warming up {model}...")
        resp = self.client.generate(
            model=model,
            prompt="Hello, respond with OK.",
            options={"num_predict": 5},
            measure_ttft=False,
        )
        if resp.get("success"):
            self.cold_start_sec = resp.get("total_time_sec")
        self.gpu_fit_pct = self.measure_gpu_fit(model)

    def measure_gpu_fit(self, model: str) -> float | None:
        """`size_vram / size` of the loaded model in percent, for runtimes that report it (Ollama `/api/ps`); else None.

        Below 100 the model does not fit in GPU memory and part of it runs on the CPU, which explains slow decode speeds.
        """
        try:
            loaded = self.client.get_running_models()
        except Exception:
            return None
        for entry in loaded:
            if entry.get("name") in (model, f"{model}:latest") or entry.get("model") == model:
                size, in_vram = entry.get("size", 0), entry.get("size_vram")
                if size and in_vram is not None:
                    return round(min(100.0, in_vram / size * 100.0), 1)
        return None

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
            text = resp.get("response", "")
            test_res = run_code_with_tests(
                solution_code=extract_python_code(text),
                test_assertions=sc.get("test_assertions", []),
                timeout_sec=SANDBOX_TIMEOUT_SEC,
            )
            error = test_res.get("error", "")
            options = self._effective_options("coding", model, sc.get("options", {}))
            if resp.get("success") and self._is_truncated(resp, options) and not has_complete_code_block(text):
                # The model was cut off before finishing its code: say so instead of reporting "name is not defined".
                error = f"Response truncated at num_predict={options.get('num_predict')} before the code was complete"
            return {
                "success": resp["success"] and test_res["passed"],
                "passed": test_res["passed"],
                "passed_tests": test_res["passed_tests"],
                "total_tests": test_res["total_tests"],
                "pass_ratio": test_res["pass_ratio"],
                "sandbox_error": error,
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
                "think_chars": max(ev["think_chars"], resp.get("thinking_chars", 0)),
                "truncated_thinking": ev["truncated_thinking"],
                "extracted_answer": ev["extracted_answer"],
                "answer_excerpt": ev["answer_excerpt"],
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
                "answer_excerpt": ev["answer_excerpt"],
                "expected_answer": ev["expected_answer"],
            }

        return self._run_suite("polish", "Polish Test", model, scenarios, score)

    def run_context_suite(self, model: str, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run context scaling scenarios (512 to 8k) to check latency and memory as the context window grows.

        Each step sets `num_ctx` to its label and sends a prompt of about `fill_ratio` of that window. On runtimes that
        reload the model when `num_ctx` changes (Ollama), an untimed request with the same `num_ctx` runs first, so the
        timed request measures prefill instead of model load time.
        """
        results = []
        for sc in scenarios:
            ctx_size = sc["context_size"]
            prompt = build_context_prompt(
                ctx_size,
                sc["instruction"],
                sc.get("fill_ratio", DEFAULT_CONTEXT_FILL_RATIO),
                sc.get("needle"),
                sc.get("needle_depth", 0.5),
            )
            options = self._apply_think(
                "context", self._with_sampling({"num_ctx": ctx_size, "num_predict": 128, "temperature": 0.2})
            )

            self._notify("Context Scaling", f"[{model}] {sc['name']} (num_ctx: {ctx_size})")
            if getattr(self.client, "reloads_on_context_change", False):
                # A different, short prompt: it loads the model at this context size without caching the long prompt.
                self.client.generate(
                    model=model,
                    prompt=CONTEXT_WARMUP_PROMPT,
                    options={"num_ctx": ctx_size, "num_predict": 1, "temperature": 0.0},
                    measure_ttft=False,
                )
            resp, hw = self._generate_with_telemetry(model, prompt, options)

            record = self._base_record("context", f"context_{ctx_size}", sc["name"], model, resp, hw, options)
            record["context_size"] = ctx_size
            record["prompt_tokens_actual"] = resp.get("prompt_eval_count", 0)
            record["success"] = resp["success"]
            if sc.get("expected"):  # the answer must contain the fact hidden in the context
                record["retrieved"] = bool(
                    resp["success"] and contains_expected(resp.get("response", ""), sc["expected"])
                )
                record["success"] = record["retrieved"]
            results.append(record)
            time.sleep(INTER_TEST_PAUSE_SEC)
        self._restore_default_context(model)
        return results

    def _restore_default_context(self, model: str) -> None:
        """Leave the model loaded with the default context size, as this suite found it.

        The last context step leaves Ollama holding the model at `num_ctx` 8192; the next suite (the first suite of the next
        repetition) would then start with a reload, and that reload would show up as a multi-second TTFT.
        """
        if getattr(self.client, "reloads_on_context_change", False):
            options = {"num_predict": 1, "temperature": 0.0}
            if "num_ctx" in self._sampling_defaults():
                options["num_ctx"] = self._sampling_defaults()["num_ctx"]
            self.client.generate(model=model, prompt=CONTEXT_WARMUP_PROMPT, options=options, measure_ttft=False)

    def compute_model_scorecard(
        self,
        model: str,
        all_test_results: list[dict[str, Any]],
        runtime: str | None = None,
        _split_runs: bool = True,
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
        avg_eval_tok_sec_mean = _mean(r["eval_tok_per_sec"] for r in speed_records)
        # Weighted by tokens (total tokens / total generation time): a 16-token answer must not count as much as a 512-token one.
        timed = [r for r in speed_records if r.get("eval_count", 0) > 0]
        generation_sec = sum(r["eval_count"] / r["eval_tok_per_sec"] for r in timed)
        avg_eval_tok_sec = (
            sum(r["eval_count"] for r in timed) / generation_sec if generation_sec > 0 else avg_eval_tok_sec_mean
        )
        # Energy: tokens per joule of GPU power (whole-request average power), where power was sampled.
        powered = [r for r in timed if r.get("tokens_per_joule")]
        joules = sum(r["eval_count"] / r["tokens_per_joule"] for r in powered)
        tokens_per_joule = sum(r["eval_count"] for r in powered) / joules if joules > 0 else None
        cold_starts = [r["cold_start_sec"] for r in model_results if "cold_start_sec" in r]
        gpu_fits = [r["gpu_fit_pct"] for r in model_results if "gpu_fit_pct" in r]
        avg_prompt_tok_sec = _mean(r["prompt_tok_per_sec"] for r in speed_records)
        avg_ttft = _mean(r["ttft_sec"] for r in speed_records)
        avg_prefill_eff = _mean(r.get("prefill_eff_tok_per_sec", 0.0) for r in speed_records)

        # Hardware metrics
        peak_vram = max((r.get("hardware", {}).get("vram_peak_mb", 0.0) for r in model_results), default=0.0)
        baselines = sorted(
            r["hardware"]["vram_baseline_mb"] for r in model_results if "vram_baseline_mb" in r.get("hardware", {})
        )
        vram_baseline = baselines[len(baselines) // 2] if baselines else None
        baseline_dirty = any(r.get("hardware", {}).get("vram_baseline_dirty") for r in model_results)
        vram_model = max(0.0, peak_vram - vram_baseline) if vram_baseline is not None and not baseline_dirty else None
        truncated_runs = sum(1 for r in model_results if r.get("truncated"))
        usage_estimated = any(r.get("usage_estimated") for r in model_results)
        asked = [r for r in model_results if r.get("suite") == "context" and "retrieved" in r]
        context_retrieval = _percent(sum(1 for r in asked if r["retrieved"]), len(asked)) if asked else None
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

        run_ids = sorted({r.get("run", 0) for r in model_results})
        spread = None
        if _split_runs and len(run_ids) > 1:
            # min-max of each metric over the repetitions (scored per repetition, not over all requests at once)
            per_run = [
                self.compute_model_scorecard(
                    model, [r for r in model_results if r.get("run", 0) == run], runtime=runtime, _split_runs=False
                )
                for run in run_ids
            ]
            spread = {key: [min(c[key] for c in per_run), max(c[key] for c in per_run)] for key in SPREAD_METRICS}

        return {
            "model": model,
            "runtime": sc_runtime,
            "engine": sc_engine,
            "runs": len(run_ids),
            "spread": spread,
            "composite_score": round(composite_score, 1),
            "coding_pass_rate": round(coding_pass_rate, 1),
            "reasoning_accuracy": round(reasoning_accuracy, 1),
            "avg_eval_tok_sec": round(avg_eval_tok_sec, 1),
            "avg_eval_tok_sec_mean": round(avg_eval_tok_sec_mean, 1),
            "tokens_per_joule": round(tokens_per_joule, 3) if tokens_per_joule is not None else None,
            "cold_start_sec": round(cold_starts[0], 2) if cold_starts else None,
            "gpu_fit_pct": gpu_fits[0] if gpu_fits else None,
            "avg_prompt_tok_sec": round(avg_prompt_tok_sec, 1),
            "avg_prefill_eff_tok_sec": round(avg_prefill_eff, 1),
            "avg_ttft_sec": round(avg_ttft, 2),
            "peak_vram_mb": round(peak_vram, 1),
            "vram_baseline_mb": round(vram_baseline, 1) if vram_baseline is not None else None,
            "vram_model_mb": round(vram_model, 1) if vram_model is not None else None,
            "vram_baseline_dirty": baseline_dirty,
            "truncated_runs": truncated_runs,
            "usage_estimated": usage_estimated,
            "context_retrieval_pct": round(context_retrieval, 1) if context_retrieval is not None else None,
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
