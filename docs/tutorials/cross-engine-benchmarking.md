# ⚖️ Tutorial 3: Fair 1:1 Cross-Engine Benchmarking (Ollama vs MS Foundry / Prism)

This tutorial explains how to conduct fair, scientifically rigorous **1:1 head-to-head benchmarks** comparing **Ollama** (`llama.cpp`) against **Microsoft Foundry Local** (`ONNX Runtime GenAI`) on identical hardware.

---

## 🎯 The Importance of 1:1 Parameter Alignment

Comparing local LLM inference engines fairly requires eliminating non-deterministic variances. If one engine runs with a context window of 2048 and temperature 0.7, while another runs at context 8192 and greedy sampling (temperature 0.0), differences in throughput and code quality stem from hyperparameter divergence rather than kernel performance.

BenchRig establishes an **Execution Alignment Baseline** defined in `config.yaml`:

```yaml
execution_alignment_1to1:
  context_tokens: 4096     # Identical KV cache pre-allocation
  temperature: 0.1         # Low sampling temperature for deterministic code
  top_p: 0.9               # Standard nucleus cutoff
  seed: 42                 # Base RNG seed; repetition N of --runs uses seed + N
```

These values are applied to every request on every runtime (a scenario's own options win). With `--runs N` the repetitions after the first
also get a short prompt marker, because Ollama caches the prompt prefix and would otherwise answer repeated prompts almost without
prefill; the scorecards then show the min-max of each headline metric over the repetitions.

---

## 🏎 Standard 1:1 Model Comparison Pairs

| Pair ID | Architecture | Ollama Model (`llama.cpp`) | MS Foundry Model (`ONNX Runtime GenAI`) |
| :--- | :--- | :--- | :--- |
| **`phi_mini`** | Phi-3 / 3.5 Mini (3.8B) | `ollama:phi3:mini` | `foundry:phi-3.5-mini` |
| **`phi4_mini`** | Phi-4 Mini (3.8B) | `ollama:phi4-mini:latest` | `foundry:phi-4-mini` |
| **`qwen_coder_7b`** | Qwen 2.5 Coder 7B | `ollama:qwen2.5-coder:7b` | `foundry:qwen2.5-coder-7b` |
| **`qwen_small`** | Qwen 3 (0.5B - 0.6B) | `ollama:qwen2.5:0.5b` | `foundry:qwen3-0.6b` |

---

## ⚡ Zero-Redundancy Workflow: Using Cached Baselines

A critical feature of BenchRig is the **`--baseline` flag**. When benchmarking new Foundry configurations or direct ONNX GPU kernels, you **never need to re-run Ollama** if a verified baseline run already exists.

### Step 1: Collect or Locate Your Ollama Baseline Run
When an Ollama run completes, it is saved in `results/runs/benchmark_<timestamp>.json`. Inspect existing runs:
```bash
python3 -c "
import json, glob
for path in sorted(glob.glob('results/runs/*.json')):
    with open(path) as f:
        data = json.load(f)
    models = [sc.get('model') for sc in data.get('scorecards', [])]
    print(f'{path} -> {models}')
"
```

### Step 2: Evaluate MS Foundry Against the Baseline
Run the benchmark exclusively for Foundry, passing the baseline file:

```bash
benchrig \
  --runtime foundry \
  --models foundry:phi-4-mini \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

BenchRig will:
1. **Load only MS Foundry**: Ollama is untouched and idle.
2. **Execute the selected suite**: Runs test assertions through `FoundryClient`.
3. **Merge scorecards**: Combines the cached Ollama metrics with the fresh Foundry metrics.
4. **Render the side-by-side terminal table**.
5. **Generate `results/1TO1_COMPARISON_REPORT.md`**.

### Using a Prism Server as the ONNX Side
The ONNX side of the comparison can also be a [Prism](https://github.com/senssei/prism-local) server, which lets you choose the
execution device explicitly. Start it, then run the same command with `--runtime prism`:

```bash
pip install prism-local
prism serve --device cuda            # or --device cpu; the default `auto` picks CUDA when it is available

benchrig \
  --runtime prism \
  --models prism:Phi-4-mini-instruct-cuda-gpu \
  --suite coding \
  --baseline results/runs/benchmark_20260918_212608.json
```

BenchRig merges the cached Ollama scorecards with the fresh Prism ones and writes the same 1:1 report. Read the `device` field of
the Prism results (in `results/runs/*.json`) instead of trusting the model name: with `--device auto`, models named `generic-cpu`
also run on CUDA. Details and a measurement are in [Runtimes](../runtimes.md#prism).

---

## 📊 Reading the 1:1 Comparative Scorecard

When evaluation concludes, BenchRig outputs a side-by-side comparative table:

```text
   ⚖️ 1:1 Cross-Engine Model Comparison: phi3:mini (Ollama) vs phi-4-mini (MS Foundry)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Evaluation Metric ┃         phi3:mini ┃        phi-4-mini ┃ Delta /          ┃
┃                   ┃           (Ollama ┃ (MS Foundry (ONNX ┃ Advantage        ┃
┃                   ┃      (llama.cpp)) ┃         Runtime)) ┃                  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Decode Speed      │          94.0 t/s │         130.2 t/s │ Foundry 1.4x     │
│ (tok/s)           │                   │                   │ faster           │
│ Prefill Speed     │        2639.8 t/s │        1006.8 t/s │ Ollama 2.6x      │
│ (tok/s)           │                   │                   │ faster           │
│ Avg TTFT          │             0.21s │             0.09s │ Foundry 2.3x     │
│ (Latency)         │                   │                   │ lower            │
│ Coding Pass Rate  │             17.6% │            100.0% │ Foundry +82.4%   │
│ Peak Memory Usage │   11135 MB (VRAM) │          10875 MB │ GPU vs CPU/RAM   │
│                   │                   │        (RAM/VRAM) │                  │
│ Composite Score   │          31.1/100 │          64.0/100 │ Foundry (+32.9)  │
└───────────────────┴───────────────────┴───────────────────┴──────────────────┘
```

### Key Metrics Explained:
1. **Decode Speed (Generation)**: Rate of token emission during text synthesis (`eval_tokens / eval_duration_sec`). Measures interactive responsiveness.
2. **Prompt Prefill Speed (eff.)**: Prompt tokens divided by the time to first token. It is defined the same way for every runtime, so it can be compared across them, but it includes request overhead (HTTP, queueing), so it understates the raw prefill rate of short prompts. Ollama also reports its own prompt evaluation time (`prompt_eval_duration`); that engine-reported value stays in the JSON results as `prompt_tok_per_sec` and is not comparable with Foundry or Prism, which report none.
3. **Time to First Token (TTFT)**: Elapsed wall time from request dispatch to the arrival of the first output token. Slashed by GPU prefill acceleration.
4. **Coding Pass Rate**: Percentage of deterministic unit tests passed in the isolated sandbox (`benchrig/core/sandbox.py`).
5. **Peak VRAM / Memory Fit** and **Model Memory (Δ)**: *Peak* is the maximum of the GPU's total `memory.used` (from `nvidia-smi`) during execution. It is the whole GPU, not just the model: other processes (a desktop, another model server, the WSL2 host) are included. *Model Memory* is that peak minus the GPU memory in use just before the model was loaded, so it isolates the model's own weights and KV cache. Use Δ to compare engines and peak to judge whether the GPU is close to spilling.

### Worked Example and Caveats

The same Phi-4-mini on Ollama (`phi4-mini:latest`, `llama.cpp`) and on a Prism server started with `--device cuda`
(`Phi-4-mini-instruct-cuda-gpu`, ONNX Runtime GenAI): RTX 5070 12 GB, WSL2, all suites, `--runs 3`, 2026-09-20. Values are means over the
three repetitions, with the min-max in brackets:

| | Ollama | Prism (CUDA) |
| :--- | :---: | :---: |
| Coding | 100% | 100% |
| Reasoning (9 scenarios) | 74.1% [66.7-77.8] | 59.3% [55.6-66.7] |
| Decode speed | 162.7 t/s [160.6-163.9] | 130.9 t/s [129.1-133.4] |
| TTFT (scorecard average) | 0.12 s [0.12-0.12] | 0.20 s [0.18-0.25] |
| Prefill (eff.) | 2148 t/s | 1264 t/s |
| Fact found in context (5 sizes) | 100% | 100% |
| Cold start (first request) | 3.4 s | 4.6 s |
| Tokens per joule | 1.08 | 1.10 |
| Peak memory, whole GPU | 4880 MB | 11708 MB |
| Model memory (Δ over baseline) | 3778 MB | 10598 MB |
| Composite score | 92.2 [90.0-93.3] | 81.8 [80.7-84.0] |

- **Speed and latency favour Ollama, in every suite:** about 1.2x the decode speed and a lower TTFT (0.04-0.08 s against 0.13-0.18 s outside the
  context suite). Both find the hidden fact at every context size, and their energy per token is the same within 2%.
- **The ONNX and GGUF builds did not reason equally well in this run.** Prism answered Knights and Knaves correctly in 1 of 3 repetitions
  (Ollama 3 of 3) and Three Mislabeled Boxes in 0 of 3 (Ollama 2 of 3); the ranges barely touch. It was investigated on the twelve-scenario suite with
  three samples per scenario (36 per setting): Ollama 28 of 36 (77.8%), Prism 21 (58.3%) and 22 (61.1%) with its own chat template and with the
  official Phi-4-mini format without newlines, and 22 (61.1%) and 22 (61.1%) for a second int4 variant (`generic-cpu`). So the prompt template is not the cause
  and the two ONNX variants agree with each other; the answers were well formed (Prism spells "strawberry" correctly and then counts 4 "r", where Ollama counts 3).
  What remains is the int4 conversion or the runtime's numerics, which this does not separate. The gap is suggestive but not established
  (p about 0.07 to 0.09 by Fisher's exact test, and samples of one model are not independent); repeat it on your own scenarios before relying on it.
- **The composite gap has two parts:** about 4.4 points from the reasoning difference and 6.0 from the memory warning, which fires when the
  whole-GPU peak crosses the threshold (a 20-point efficiency penalty at 30% weight). Prism's model memory is about 2.8x Ollama's. Measured
  separately, ONNX Runtime GenAI's GPU memory grows by about 1.4 MB per prompt token and is not released (the context suite's long prompts
  are what drives the peak), and Prism's opt-in `PRISM_PREFILL_CHUNK=256` cut the 4500-token peak from 11.7 GB to 6.6 GB
  ([Prism docs](https://senssei.github.io/prism-local/devices/#gpu-memory-and-long-prompts)); this comparison ran without it.
- **Repeat before you quote.** In earlier single runs of these models `deepseek-r1:14b` coding went from 70.6% to 41.2% between two runs, so
  differences of a few points need `--runs 3` and a look at the ranges. Scenarios that no model passes (Letter Counting, Distinct-Digit
  Multiples of Five here) carry no ranking information.
- **Check the engine and device of each result.** Model names such as `generic-cpu` do not say where a model ran
  ([Runtimes](../runtimes.md#prism)). The `device` of every Prism result in this example is `cuda`.
- **Thinking models** (`deepseek-r1`) report a latency `ttft_sec` of the first token of any kind and, separately, `answer_ttft_sec` (seconds); see
  [Benchmark suites](../benchmark-suites.md#2-coding-algorithmic-sandbox-suite-coding).

--- | :---: | :---: |
| Coding / reasoning | 100% / 66.7% | 100% / 66.7% |
| Decode speed | 166.2 t/s | 137.6 t/s |
| TTFT, all suites (scorecard average) | 0.13 s | 0.35 s |
| Prefill (eff.) | 2631 t/s | 1312 t/s |
| Peak memory, whole GPU | 4809 MB | 11432 MB |
| Model memory (Δ over baseline) | 3732 MB | 10424 MB |
| Composite score | 90.0 | 84.0 |

- **Quality is equal.** Ollama decodes about 1.2x faster and answers the first token sooner in every suite.
- **The composite gap is the memory warning.** Both engines reach the full speed score, so the 6.0 points come from Prism's peak
  memory crossing the warning threshold (a 20-point efficiency penalty at 30% weight). Its model memory is about 2.8x Ollama's,
  and grew from about 5.2 GB to 10.4 GB as the context steps grew; that looks like ONNX Runtime's CUDA memory allocation, which
  was not investigated. The warning uses the whole-GPU peak on purpose, because that is what decides whether memory spills.
- **Compare per suite and repeat runs.** One run is an indication: in two consecutive full runs of the same models, `deepseek-r1:14b`
  coding went from 70.6% to 41.2% and `qwen2.5-coder:7b` reasoning from 100% to 66.7% (sampling at temperature 0.1 is not
  deterministic). Use `--runs 3` before drawing conclusions from differences of a few points.
- **Check the engine and device of each result.** Model names such as `generic-cpu` do not say where a model ran
  ([Runtimes](../runtimes.md#prism)). The `device` of every Prism result in this example is `cuda`.
- **Thinking models** (`deepseek-r1`) show TTFT of seconds because it includes the thinking time; see
  [Benchmark suites](../benchmark-suites.md#2-coding-algorithmic-sandbox-suite-coding).

--- | :---: | :---: |
| Composite score | 90.0 | 90.0 |
| Coding / reasoning | 100% / 66.7% | 100% / 66.7% |
| Decode speed | 170.6 t/s | 131.1 t/s |
| TTFT outside the context suite | 0.04 to 0.07 s | 0.13 to 0.59 s |
| TTFT, scorecard average (all suites) | 1.95 s | 0.28 s |

Quality and composite score are equal, and Ollama decodes faster and answers the first token sooner in normal requests. The
scorecard TTFT points the other way only because of the context suite (see
[Benchmark suites](../benchmark-suites.md#4-context-saturation-suite-context)): Ollama's roughly 7 s per step there is model reload caused by the changing `num_ctx`, not prefill.
So:

- Compare TTFT and speed **per suite**, not only from the scorecard average.
- Treat one run as an indication. Repeat with `--runs 3` before drawing conclusions from differences of a few percent.
- Check the `device` of Prism results and the engine of each result; model names such as `generic-cpu` do not say where a model
  ran ([Runtimes](../runtimes.md#prism)).

---

## 🔄 Re-Analyzing Historical Runs (`--compare`)

To re-display terminal tables and re-generate `1TO1_COMPARISON_REPORT.md` without invoking any inference engines:

```bash
benchrig --compare results/latest.json
# Or specify any historical run:
benchrig --compare results/runs/benchmark_20260918_235227.json
```

---

## 🌡 Warm-up and KV Cache Reuse Protocol (`--warmup-runs`)

Benchmarking steady-state inference performance requires separating cold-start latency and JIT pipeline compilation from regular token generation (I14):

- **Warm-up Flag (`--warmup-runs N`, default 1)**: Executes `N` warm-up requests per scenario prior to recording benchmark timings.
- **Strict Average Isolation**: Warm-up records carry `phase="warmup"` and are strictly excluded from scorecard averages (`avg_eval_tok_sec`, `avg_ttft_sec`, and composite scores).
- **Cache Mode Protocol (`cache_mode`)**:
  - `"cold"`: Explicit backend evidence reports initialization during this request, and a prefix miss is verified.
  - `"warm"`: Runtime warm-up succeeded and a prefix miss is verified.
  - `"prefix_cached"`: An explicit boolean cache-hit field in server telemetry verifies prefix reuse. Zero prefill count alone does not prove a hit.
  - `"unverified"`: Prefix reuse or runtime state lacks sufficient evidence. Request order alone cannot establish cold/warm state.



### Methodology feedback — (2026-10-04)

Model/runtime warm-up and prefix reuse fail in different ways and should be
reported separately. The first prepares resident weights and runtime kernels;
the second measures a steady-state KV-cache hit. Compare a second request with
the same prefix while fixing cache type and context length, and retain the
requested settings plus evidence of the actual cache hit. Repetition alone is
not proof that the server reused the prefix.

Record load, prefill and decode durations separately from TTFT. TTFT covers the
wait until the first generated token and may include loading and prefill;
decode continues after that token. When the runtime does not expose a component
or cache-hit evidence, report it as unavailable or unverified rather than infer
it from a faster response. A single label such as "warm" cannot describe both
runtime initialization and verified prefix reuse.

This feedback describes the desired comparison protocol; it does not establish
that every backend currently exposes cache type or engine timing components.


### Controlled paired cache probe

Use `--cache-probe --suite speed --models MODEL --runtime RUNTIME --runs 3`
with a custom `speed.json` selected through `--scenarios-dir`. Each scenario must
include a positive `options.num_ctx`; bundled speed scenarios do not specify it.
BenchRig warms the runtime with a different prompt at those same effective settings,
then issues the scenario prompt twice unchanged for each pair. There is no cache
reset between requests or repetitions, so the first request may already be cached.
`--warmup-runs` does not add scenario iterations to this mode; the probe has its own
single runtime warm-up per scenario.

Records separate `runtime_state` from `prefix_cache_state`. Paired requests retain
prompt hashes, pair IDs, positions, settings, versions where exposed, raw telemetry
and timing provenance. `load_time_sec` is an engine load measurement; startup request
wall time is a separate quantity and may include load, prefill and generation.
Missing load timing is unavailable, including direct ONNX request timing when the
model was loaded before generation. Historical files retain their original labels.

A controlled comparison requires two successful requests and an unchanged observed
cache type and context matching the requested window. Missing settings, failures or
changes produce an unverified comparison with reasons. Explicit cache-hit telemetry
is additionally required to claim prefix reuse. The normalized server fields are
`prefix_cache_hit` (boolean), `cache_type` (string) and `context_tokens` (positive
integer window size), in native final metrics or OpenAI-compatible telemetry.
Backends that do not expose those fields remain unverified; faster responses and
zero prefill tokens are insufficient. No deployed backend support is certified by
fixture tests. BenchRig observes configuration; it does not change daemon settings,
clear caches or restart servers.

The dedicated Markdown report lists load, prefill, decode and client TTFT separately
with provenance, plus pair status and missing evidence. Probe records do not enter
legacy scores or averages. Keep the logging level fixed across comparisons; the
existing INFO/DEBUG logging overhead remains inside parts of the timing window.
Live evidence and conclusions require a separately selected model/runtime run.

A positive backend load/setup duration alone does not prove model initialization;
`runtime_initialized: true` is required for a cold classification. Probe output uses
`CACHE_PROBE_SUMMARY.md`, `cache_probe_latest.json` and `runs/cache_probe_*.json`,
preserving normal benchmark summaries. Startup failures also produce diagnostics.
