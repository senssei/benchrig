# 🧪 Custom Scenarios & Benchmark Authoring

This guide explains how test scenarios are structured, how deterministic assertions are executed in isolated sandboxes, and how to author custom benchmarks within the [`scenarios/`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/) directory.

---

## 📂 Scenario Catalog Overview

All test cases in **BenchRig** are defined as modular JSON arrays in the `scenarios/` directory:

| Scenario File | Target Suite | Evaluation Methodology |
| :--- | :--- | :--- |
| [`coding.json`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/coding.json) | `coding` | Python code block extraction, AST validation, execution in isolated subprocesses against unit assertion arrays. |
| [`reasoning.json`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/reasoning.json) | `reasoning` | Twelve multi-step math and logic puzzles evaluated against ground truth (a `Final answer:` line, whole-number matching or regex); tracks thinking metrics. |
| [`speed.json`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/speed.json) | `speed` | Fixed-token prefill and decode prompts to measure raw token throughput and Time to First Token (TTFT). |
| [`context_scaling.json`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/context_scaling.json) | `context` | Scaled input contexts (512 to 8192 tokens) with a hidden fact to retrieve (`needle`, `instruction`, `expected`), measuring whether the model uses the context, TTFT degradation and memory growth. |
| [`polish.json`](https://github.com/senssei/benchrig/blob/main/benchrig/data/scenarios/polish.json) | `polish` | Polish language grammatical inflections, noun cases (biernik, dopełniacz), and syntax validation. |

---

## 💻 1. Authoring Coding Scenarios (`coding.json`)

Coding scenarios evaluate algorithmic accuracy by executing the generated Python code against deterministic unit test assertions in an isolated subprocess.

### JSON Schema
```json
{
  "id": "unique_string_identifier",
  "name": "Human-Readable Test Name",
  "prompt": "Full prompt specifying function signature, constraints, and instructions to return code in ```python ... ``` fences.",
  "options": {
    "temperature": 0.1,
    "num_predict": 512
  },
  "test_assertions": [
    "assert my_func(input1) == expected1",
    "assert my_func(input2) == expected2"
  ]
}
```

### Execution Flow in `SandboxRunner` ([`benchrig/core/sandbox.py`](https://github.com/senssei/benchrig/blob/main/benchrig/core/sandbox.py))
1. **Code Extraction**: The runner parses the model response, stripping out markdown formatting (```` ```python ... ``` ````).
2. **Harness Assembly**: Combines the extracted code with all statements in `test_assertions`.
3. **Isolated Subprocess**: Writes the harness to an ephemeral temporary file and executes it using `python3` with a strict execution timeout (default: 10s).
4. **Assertion Result**:
   - If all assertions pass: `PASS (N/N tests)`.
   - If an assertion fails or an unhandled exception occurs: `FAIL (Traceback captured)`.
   - If execution exceeds timeout: `TIMEOUT`.

### Example
```json
{
  "id": "code_reverse_words",
  "name": "Reverse Words in String",
  "prompt": "Write a Python function `reverse_words(s: str) -> str` that reverses the order of words in a string while preserving single spaces between words and trimming extra whitespace.\nRespond only with valid Python code inside ```python ... ```.",
  "options": {
    "temperature": 0.1,
    "num_predict": 256
  },
  "test_assertions": [
    "assert reverse_words('the sky is blue') == 'blue is sky the'",
    "assert reverse_words('  hello world  ') == 'world hello'",
    "assert reverse_words('a good   example') == 'example good a'"
  ]
}
```

---

## 🧠 2. Authoring Reasoning Scenarios (`reasoning.json`)

Reasoning scenarios evaluate multi-step deduction, math problem solving, and chain-of-thought models (e.g. DeepSeek-R1, QwQ).

### JSON Schema
```json
{
  "id": "reasoning_unique_id",
  "name": "Scenario Title",
  "prompt": "Detailed reasoning puzzle requiring step-by-step derivation.",
  "options": {
    "temperature": 0.1,
    "num_predict": 1024
  },
  "check_type": "final_answer | exact_or_contains | regex | numeric",
  "expected_answer": "Standard ground truth",
  "accepted_patterns": [
    "pattern1",
    "pattern2"
  ]
}
```

### Check Types

| `check_type` | The answer is correct when... |
| :--- | :--- |
| `final_answer` (recommended) | The **last** line after a `Final answer` marker fully matches one of the `accepted_patterns` (`re.fullmatch`, case-insensitive). Markdown, LaTeX (`\boxed{\dfrac{1}{6}}` reads as `1/6`), `$`, thousands separators and a trailing full stop are removed first. No marker means no answer. End the prompt with `End with exactly one line in the form: Final answer: <answer>`. |
| `exact_or_contains` | `expected_answer` or one of `accepted_patterns` occurs in the answer. Patterns that contain a digit only match as whole numbers (`1/6` does not match `11/60`, `114.6` does not match `1114.60`); other patterns are plain substrings. |
| `regex` | One of `accepted_patterns` matches the **end** of the answer (last 300 characters) after markdown, bullets, punctuation and line breaks are collapsed to single spaces. Use bounded gaps, not `.*`. |
| `numeric` | The `\boxed{...}` number, or else the last number in the answer, equals `expected_answer`. |

A response that stops inside an unclosed `<think>` has no answer, and text that only appears in the thinking never counts.

The `options` may also carry `think` (`true` or `false`) to turn thinking on or off for that scenario on Ollama models that can think;
see [Configuration](configuration.md).

### Verifying Your Expected Answer
A wrong expectation silently ranks models wrongly, so compute the answer with independent code (a brute force or a formula) and
check that the scenario accepts it and rejects near misses. `tests/test_scenarios.py` does this for every bundled `final_answer`
scenario and is the pattern to copy.

### Answer Extraction & `<think>` Tag Parsing
Reasoning models often encapsulate chain-of-thought reasoning inside `<think>...</think>` tags. The `ReasoningParser` ([`benchrig/core/reasoning_parser.py`](https://github.com/senssei/benchrig/blob/main/benchrig/core/reasoning_parser.py)):
1. Detects `<think>` blocks and separates the thinking stream from the final answer.
2. Calculates the **Thinking Token Count** and thinking duration.
3. Normalizes and validates the final response against `accepted_patterns` using either case-insensitive substring matching or regular expressions.

---

## ⏱ 3. Authoring Speed Scenarios (`speed.json`)

Speed scenarios measure maximum token generation throughput and prompt processing (prefill) speed without complex logic overhead.

### JSON Schema
```json
{
  "id": "speed_unique_id",
  "name": "Throughput Test Name",
  "prompt": "Instructions designed to elicit a continuous stream of tokens.",
  "options": {
    "temperature": 0.0,
    "num_predict": 256
  }
}
```

---

## 📐 4. Context Scaling Scenarios (`context_scaling.json`)

Context scaling scenarios measure how prompt prefill speed and Time to First Token (TTFT) degrade as the input context window expands from 512 tokens to 8k tokens.

### Schema
```json
{
  "id": "ctx_4096",
  "name": "Context Scaling 4k",
  "context_tokens": 4096,
  "prompt_file": "scenarios/prompts/long_context_4k.txt",
  "query": "Summarize the key conclusion from the text above."
}
```

---

## 🛠 Best Practices for Scenario Design

1. **Deterministic Prompts**: Set `"temperature": 0.1` or `0.0` to ensure reproducibility across runs.
2. **Explicit Signatures**: Provide full function headers with Python type hints in coding prompts.
3. **Comprehensive Assertions**: Include boundary cases (empty collections `[]`, `{}`, negative values, large inputs).
4. **Isolated Test State**: Do not write assertions that depend on external internet access or filesystem state.
