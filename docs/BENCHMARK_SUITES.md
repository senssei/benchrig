# 🧪 Benchmark Suites & Evaluation Methodology

This document details the design, scenarios, scoring formulas, and validation methodologies employed by the 5 specialized benchmark suites in **Ollama BenchRig**.

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

### Isolated Subprocess Sandbox (`core/sandbox.py`)
Rather than relying on lexical matching (e.g. BLEU/ROUGE), Ollama BenchRig executes the model's generated code inside an **isolated Python sandbox**:
1. Strips conversational prose, backticks (` ```python `), and trailing explanations.
2. Appends an automated test harness with strict `assert` statements covering edge cases.
3. Spawns an isolated `subprocess.run` with a strict 10-second timeout.
4. Validates stdout, stderr, and assertion return codes.

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
* **`<think>` Parsing**: The parser in [`core/reasoning_parser.py`](../core/reasoning_parser.py) inspects whether the model separates internal reasoning (`<think>...</think>`) from the final response.
* **Token Budget Telemetry**: Tracks the proportion of tokens spent on internal exploration vs the final delivered answer.
* **Deterministic Verification**: Extracts the final answer and checks exact equality against ground-truth mathematical and logical solutions.

---

## 4. 📐 Context Saturation Suite (`context`)

Assesses how inference performance degrades as context length increases.

### Methodology
* Progressively submits structured prompts scaling from **512**, **1024**, **2048**, **4096**, up to **8192** tokens.
* Samples Time-to-First-Token (TTFT) and memory growth across each step.
* Identifies the threshold where the model's KV-cache exceeds physical GPU VRAM and triggers spilling into system RAM.

---

## 5. 🇵🇱 Polish NLP Linguistic Suite (`polish`)

Evaluates linguistic precision in morphologically rich and complex languages (Polish).

### Scenarios
* **Grammatical Declension**: Testing correct nominal and adjectival case inflections across all 7 Polish cases (*przypadki*).
* **Idiomatic Nuance**: Translating and interpreting idioms and culturally specific expressions.
* **Text Summarization**: Preserving factual precision under strict word-count limits.
