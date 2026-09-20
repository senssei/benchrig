# 🧪 Benchmark Suites & Evaluation Methodology

This document details the design, scenarios, scoring formulas, and validation methodologies employed by the 5 specialized benchmark suites in **BenchRig**.

---

## 1. ⚡ Speed & Latency Suite (`speed`)

The speed suite evaluates inference latency, maximum decoder velocity, and prompt ingestion throughput under streaming conditions.

### Scenarios
1. **Raw Decode Speed (Short Prompt)**:
   - Measures maximum decoding velocity when the context window is near-empty.
   - Evaluates pure token sampling speed unconstrained by large KV-cache buffers.
2. **Throughput Generation (512 tokens)**:
   - Tasks the model with generating an extended completion to measure sustained throughput over time.
   - Reveals thermal throttling, memory bandwidth saturation, or engine queue degradation.

### Metrics Measured
* **Time-to-First-Token (TTFT)**: High-resolution timestamp delta from HTTP socket dispatch to the arrival of the first completion byte.
* **Prefill Speed (Prompt Tokens/s)**: Hardware throughput while ingesting and calculating initial attention keys/values for the prompt.
* **Decode Speed (Completion Tokens/s)**: Sequential auto-regressive generation speed of output tokens.

---

## 2. 💻 Coding & Algorithmic Sandbox Suite (`coding`)

The coding suite measures whether an LLM produces syntactically valid and deterministically functional Python code without human intervention.

### Isolated Subprocess Sandbox (`benchrig/core/sandbox.py`)
Rather than relying on lexical matching (e.g. BLEU/ROUGE), BenchRig executes the model's generated code inside an **isolated Python sandbox**:
1. Strips conversational prose, backticks (` ```python `), and trailing explanations.
2. Appends an automated test harness with strict `assert` statements covering edge cases.
3. Spawns an isolated `subprocess.run` with a strict 10-second timeout.
4. Validates stdout, stderr, and assertion return codes.

**Reasoning traces and cut-off answers.** A model that thinks first (`<think>...</think>`, e.g. DeepSeek-R1) can spend its whole
token budget before writing any code. Code inside the `<think>` trace is ignored, and a response that stopped at the token
limit without a complete code block is reported as `Response truncated at num_predict=N before the code was complete` (with
`truncated: true` in the result) instead of a misleading `name '...' is not defined`. To give such models room, the budget
(`num_predict`) of the coding, reasoning and polish suites is multiplied for models matching `benchmark.thinking_models`
(default `deepseek-r1`, `qwq`, `magistral`, `qwen3`) by `benchmark.thinking_token_multiplier` (default 12); see
[Configuration](configuration.md). Scorecards count cut-off responses as `truncated_runs` and the reports flag them with `⚠`. Only the suites that score an answer
(coding, reasoning, polish) are checked: the speed and context suites generate a fixed number of tokens on purpose, so stopping
at the limit there is not truncation (`finish_reason` is still recorded).

**Thinking models and the `think` option.** Ollama streams a thinking model's reasoning in a separate `thinking` field. BenchRig
reads it: `ttft_sec` is the time to the first token of *any* kind (the latency), `answer_ttft_sec` the time to the first answer
token, `think_time_sec` the difference, and `thinking_chars` the amount of thinking. (Earlier versions reported only the first
answer token, so a thinking model showed seconds of "TTFT", or 0.0 s when it never answered.)

Whether a model thinks is set with Ollama's `think` option, per suite, in `benchmark.think` (see [Configuration](configuration.md)):
by default **speed and context run with thinking off** so throughput and prompt-processing latency are not mixed with thinking
time, and coding, reasoning and polish use the model's own default. Turning thinking off for coding measures the model
without reasoning (much faster), but the larger token budget still applies to these models: in one run `deepseek-r1:14b` with thinking
off and the base budget (512 tokens) never produced a complete code block (0 of 4 tests), while with its default thinking it passed
4 of 4 on the same scenario (4541 tokens, 14752 characters of thinking, 73 s to the first answer token, but a `ttft_sec` of 0.11 s).
The option is sent only to models that Ollama lists as able to think; other runtimes ignore it.

**How much budget do thinking models need?** Measured on `deepseek-r1:14b`, coding suite, one run each: a multiplier of 6 cut off
1 to 2 of the 4 scenarios, 12 cut off 1, and 20 cut off none; the scenario that ran out at 12 (6144 tokens) finished in 4609
tokens at 20. The need is a long tail, hence the default of 12; use `--runs 3` and check `truncated_runs` for your models.

### Algorithmic Scenarios

| Scenario | Algorithmic Focus | Assertions Tested |
| :--- | :--- | :---: |
| **Nested Dict Flattening** | Deep recursion, key path concatenation, delimiter handling | 4 |
| **Merge Overlapping Intervals** | O(N log N) sorting, greedy interval consolidation, edge intervals | 5 |
| **Valid Balanced Brackets** | Stack data structure, pair mapping, empty and malformed string handling | 6 |
| **LRU Cache Implementation** | Doubly-linked hash map / `OrderedDict`, O(1) get/put operations | 2 |

---

## 3. 🧠 Reasoning & Thought Chain Suite (`reasoning`)

Designed to test models that generate reasoning chains (such as DeepSeek-R1 or Qwen 2.5 with reasoning prompts).

### Methodology
* **`<think>` Parsing**: The parser in [`benchrig/core/reasoning_parser.py`](https://github.com/senssei/benchrig/blob/main/benchrig/core/reasoning_parser.py) inspects whether the model separates internal reasoning (`<think>...</think>`) from the final response.
* **Token Budget Telemetry**: Tracks the proportion of tokens spent on internal exploration vs the final delivered answer.
* **Deterministic Verification**: Extracts the final answer and checks exact equality against ground-truth mathematical and logical solutions.

!!! note "What counts as an answer"
    - The answer is what follows `</think>`. A response cut off *inside* `<think>` has no answer and is scored as wrong
      (`truncated_thinking: true`); text that only appears in the reasoning trace never counts, even if it contains the expected
      words.
    - `regex` checks look only at the **end of the answer** (the last 300 characters), after markdown, bullets, punctuation and line
      breaks are collapsed to spaces. A conclusion written as `Box 1: Oranges, Box 2: ...` on one line, on separate lines or as
      bullets reads the same, while words from the right answer that merely appear earlier in the reasoning do not count. The
      bundled patterns use bounded gaps (a few filler words) instead of `.*`, so they cannot be satisfied by scattered mentions.
    - LaTeX in answers is simplified before matching (`\boxed{\dfrac{1}{6}}` reads as `1/6`), because reasoning models often
      answer that way.
    - Every result records `answer_excerpt` (the last 400 characters of the answer) so a score can be audited from
      `results/runs/*.json` without re-running the model.
    - `final_answer` scenarios (six of the nine) take the last `Final answer: ...` line and require it to match an accepted pattern
      in full; no marker means no answer. `exact_or_contains` matches numbers as whole numbers, so `11/60` no longer counts as `1/6`
      and `1114.60` no longer counts as `114.6` (both used to pass).
    - The expected answers of the new scenarios are computed by independent code in `tests/test_scenarios.py`, which also rejects near
      misses, so a wrong expectation cannot rank models wrongly.

### Scenarios

| Scenario | Check | Expected |
| :--- | :--- | :--- |
| Multi-Step Discount & Tax Calculation | contains number | 114.6 |
| Three Mislabeled Boxes Logic Puzzle | regex on the conclusion | Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples |
| Set Theory & Probability | contains number | 1/6 |
| Two Trains Meeting Time | `Final answer` | 11:20 |
| Knights and Knaves (3 people) | `Final answer` | A is a knave, B and C are knights |
| Distinct-Digit Multiples of Five (4-digit numbers) | `Final answer` | 952 |
| Linear Recurrence (a10 of a1=3, a(n+1)=2a(n)-1) | `Final answer` | 1025 |
| Chained Percentage Changes (80, +25%, -20%, +10%) | `Final answer` | 88 |
| Letter Counting (r in "strawberry raspberry blueberry") | `Final answer` | 8 |

In one run of `phi4-mini`, `qwen2.5-coder:7b` and `gemma3:12b` the new scenarios split the models (Letter Counting 0 of 3, Distinct-Digit
Multiples 2 of 3) where the old three mostly passed for everyone. Scores are still coarse with nine scenarios; add your own for a finer
ranking ([Custom scenarios](tutorials/custom-scenarios.md)).

---

## 4. 📐 Context Saturation Suite (`context`)

Assesses how inference performance degrades as context length increases.

### Methodology
* Runs five steps, labelled **512**, **1024**, **2048**, **4096** and **8192** (`context_scaling.json`). Each step sets the request's
  `num_ctx` to the label and sends a prompt of about `fill_ratio` (0.75) of that window, built from a repeated filler paragraph.
* **A fact is hidden in the text** (`needle`, placed at `needle_depth`, by default halfway, at a sentence boundary) and the question asks
  for it (`instruction`, answer expected in `expected`). A step passes when the answer contains the expected number as a whole
  number. Each step hides a different code, so it cannot be memorised. The suite therefore measures whether the model *uses* its
  context, not only how fast it reads it; the share of steps passed is the scorecard's `context_retrieval_pct`.
* On runtimes that reload the model when `num_ctx` changes (Ollama), an untimed request with the same `num_ctx` runs first, so
  the timed request measures prompt processing and not model load. The load time is recorded as `load_time_sec`.
* Samples Time-to-First-Token (TTFT) and memory growth across each step.
* Identifies the threshold where the model's KV-cache exceeds physical GPU VRAM and triggers spilling into system RAM.

!!! note "Reading the context numbers"
    - **The label is the context window, not the prompt length.** Prompt size is estimated from characters (about 4 per token), and
      real tokenizers differ, so the real token count is recorded as `prompt_tokens_actual`. For Phi-4-mini the five steps sent
      289, 565, 1107, 2199 and 4374 tokens (about 55% of each window). Use `fill_ratio` in the scenario file to change the fill.
    - **TTFT now grows with the prompt.** Before the warm-up was added, the Ollama numbers here were dominated by model reload
      (about 4 s per step measured directly, about 7 s in the suite), because every step changed `num_ctx`. Results from earlier
      versions of this suite are not comparable.
    - **Where a runtime cannot fit the window**, the step is recorded as a failure with the engine's error; that is a finding, not a
      harness fault.
    - **The scorecard's `avg_ttft_sec` still averages every suite**, including the large-context steps, so compare TTFT per suite
      (`results/runs/*.json`) when the prompts differ in size.

---

## 5. 🇵🇱 Polish NLP Linguistic Suite (`polish`)

Evaluates linguistic precision in morphologically rich and complex languages (Polish).

### Scenarios
* **Grammatical Declension**: Testing correct nominal and adjectival case inflections across all 7 Polish cases (*przypadki*).
* **Idiomatic Nuance**: Translating and interpreting idioms and culturally specific expressions.
* **Text Summarization**: Preserving factual precision under strict word-count limits.
