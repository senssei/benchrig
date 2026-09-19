# 🧪 Tutorial 4: Authoring Custom Benchmark Scenarios

This tutorial guides you through authoring and registering custom evaluation scenarios for **BenchRig**—including deterministic coding challenges, multi-step logical reasoning tests, and domain-specific benchmarks.

---

## 📂 Scenario Architecture in BenchRig

All benchmark scenarios reside in the `scenarios/` directory as JSON files:
- **`scenarios/coding.json`**: Algorithmic problems executed in isolated sandboxes against unit test assertions.
- **`scenarios/reasoning.json`**: Logical puzzles with chain-of-thought (`<think>`) and answer extraction.
- **`scenarios/speed.json`**: Variable-length prompts for testing prefill and decode throughput.
- **`scenarios/context_scaling.json`**: Stepwise context tests (512 to 8,192 tokens) measuring KV cache degradation.
- **`scenarios/polish.json`**: Linguistic, declension, and grammatical verification.

---

## 💻 1. Authoring a Custom Coding Scenario

Coding scenarios are evaluated by [`benchrig/core/sandbox.py`](../../benchrig/core/sandbox.py). The sandbox extracts the Python function from the model's response, appends your test harness, and executes the combined code in an isolated subprocess.

### Schema:
```json
{
  "name": "Human-readable scenario title",
  "category": "algorithms | data-structures | concurrency",
  "difficulty": "easy | medium | hard",
  "prompt": "Clear instruction specifying function signature, arguments, and return types.",
  "test_code": "Python harness that verifies the implementation and reports __FAILURES__."
}
```

### Example: Rate Limiter Coding Scenario
Let's create a custom scenario testing a token bucket rate limiter.

Add the following object to `scenarios/coding.json`:

```json
{
  "name": "Token Bucket Rate Limiter",
  "category": "concurrency",
  "difficulty": "medium",
  "prompt": "Write a Python class `TokenBucket(rate, capacity)` that implements a token bucket rate limiter. It should have a method `consume(tokens=1) -> bool` returning True if tokens were consumed, False otherwise. Use `time.time()` for timestamps.",
  "test_code": "\nimport time\n\nfailures = []\npassed = 0\n\ntry:\n    tb = TokenBucket(rate=10, capacity=10)\n    if not tb.consume(5):\n        failures.append('Initial consumption failed')\n    else:\n        passed += 1\n        \n    if not tb.consume(5):\n        failures.append('Consuming remaining capacity failed')\n    else:\n        passed += 1\n        \n    if tb.consume(1):\n        failures.append('Over-consuming tokens should return False')\n    else:\n        passed += 1\n        \n    # Simulate time passing\n    time.sleep(0.2)\n    if not tb.consume(2):\n        failures.append('Tokens did not refill after time elapsed')\n    else:\n        passed += 1\nexcept Exception as e:\n    failures.append(f'Exception raised during execution: {e}')\n\nif failures:\n    print(f'__FAILURES__:' + ' | '.join(failures))\nelse:\n    print(f'__PASSED__:{passed}/4')\n"
}
```

### How `benchrig/core/sandbox.py` Interprets Your Test:
1. **Assertion Counting**: The runner looks for `__PASSED__:<passed>/<total>`.
2. **Failure Reporting**: If failures occur, the runner captures `__FAILURES__:<details>` and embeds the exact diagnostic message in `results/1TO1_COMPARISON_REPORT.md`.
3. **Execution Sandbox**: Runs in an isolated subprocess with a 10-second timeout. Any infinite loop (`while True`) or hanging socket is aborted cleanly without crashing the benchmark.

---

## 🧠 2. Authoring a Custom Reasoning Scenario

Reasoning scenarios evaluate a model's logical deduction and chain-of-thought capability. The engine [`benchrig/core/reasoning_parser.py`](../../benchrig/core/reasoning_parser.py) measures:
- Whether the model generates thinking tags (`<think>...</think>`).
- Total tokens spent on reasoning vs final answer synthesis.
- Correctness of the extracted final answer against `expected_answer`.

### Schema:
```json
{
  "name": "River Crossing Logic Puzzle",
  "category": "logic",
  "difficulty": "medium",
  "prompt": "A farmer must cross a river with a wolf, a goat, and a cabbage. His boat can only carry himself and one item. If left alone, the wolf eats the goat, or the goat eats the cabbage. What is the minimum number of crossings needed? Provide your final answer as an integer inside \\boxed{}.",
  "expected_answer": "7"
}
```

### Verification Heuristics:
1. **LaTeX Boxed Notation**: If the model encloses its answer in `\boxed{7}`, `benchrig/core/reasoning_parser.py` extracts it automatically.
2. **Natural Text Extraction**: The parser checks keywords like `"Answer: 7"`, `"final answer is 7"`, or standalone numeric values on the last line.

---

## 🚀 3. Running Custom Scenarios

### Step 1: Execute via CLI
Run your updated scenario suite immediately:

```bash
# Evaluate coding suite with your new scenario:
benchrig --runtime ollama --models qwen2.5-coder:7b --suite coding

# Evaluate reasoning suite:
benchrig --runtime ollama --models llama3.1:8b --suite reasoning
```

### Step 2: Verify in Detailed Assertion Output
Inspect the scenario assertion breakdown in `results/LATEST_SUMMARY.md`:

```markdown
| Scenario Name | phi3:mini Status | qwen2.5-coder:7b Status | Speed Comparison |
|:---|:---:|:---:|:---:|
| **Token Bucket Rate Limiter** | ❌ FAIL (1/4) | ✅ PASS (4/4) | 94.0 vs 128.6 t/s |
```

---

## 💡 Best Practices for Reliable Scenarios

1. **Idempotence**: Scenarios should not rely on external networks or internet connectivity.
2. **Minimal External Dependencies**: Standard library imports (`math`, `collections`, `itertools`, `re`, `time`) ensure cross-platform compatibility.
3. **Robust Harnesses**: Always wrap test calls in `try...except` blocks within `test_code` so that a `TypeError` on Test 1 does not abort execution before Test 2 and Test 3 run.
