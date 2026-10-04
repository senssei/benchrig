# Model selection for a 12 GB GPU

A measured protocol for choosing a local model on a 12 GB card (RTX 5070 class). It started from external advice to prefer
32B Q4 models over a 78B IQ2 model. That advice came without measurements, so each claim below is a **hypothesis** until a
run on your host confirms or rejects it.

## Hypotheses (none verified yet)

- **H1.** A 32B Q4_K_M (about 19-20 GB) does not fit in 12 GB, so it runs with partial CPU offload. The claimed "2-4x more
  tokens/s" over a 78B IQ2 (also offloaded) is more plausibly 1.5-2x and depends on RAM bandwidth and context.
- **H2.** IQ2 degrades code and strict-format output (PowerShell, JSON) enough that a 32B Q4 wins on `coding` and `polish`.
- **H3.** Models that fit entirely in VRAM (Qwen3 14B Q4/Q5), or MoE models with few active parameters (Qwen3-30B-A3B,
  Qwen3-Coder-30B-A3B), beat a dense 32B on tokens/s at comparable quality.
- **H4.** Reasoning-distilled models (DeepSeek-R1 Distill 32B) lose on effective answer latency because of long thinking
  traces; `answer_ttft_sec` and `thinking_chars` should show it.

"glm 32B" and "Kolibri-1 78B" must be confirmed to exist, and in which quantization, before they enter a run.

## Candidates

| Model | Fits 12 GB | Status |
|---|---|---|
| `qwen3:14b`, `qwen2.5-coder:14b`, `deepseek-r1:14b`, `gemma3:12b` | yes (about 8-9 GB) | installed on the reference host |
| Qwen3-30B-A3B, Qwen3-Coder-30B-A3B Q4 | partly | not installed |
| Qwen3 32B Q4_K_M, DeepSeek-R1 Distill 32B Q4 | no, offloaded | not installed |
| GLM 32B Q4 | no, offloaded | name unverified |
| Kolibri-1 78B IQ2 | no, offloaded | existence unverified; benchmark-only data point |

Run the fast, fits-in-VRAM models first. Offloaded 32B and 78B runs decode at single-digit tokens/s, so a full suite with
`--runs 3` can take hours per model.

## Protocol

```bash
benchrig --runtime ollama --models qwen3:14b --suite coding,reasoning,polish,speed --runs 3 --csv out/qwen3-14b.csv
```

Keep fixed across models: `--runs 3`, the default warm-up, an explicit `num_ctx` in the speed scenario options, and the same
logging level. Free the GPU of other processes first; foreign VRAM shows up in `vram_baseline_dirty_mb`.

Record per run: GPU, driver, RAM size and speed, runtime and version, context length, KV-cache type.

## Reading partial offload

`gpu_fit_pct` is the share of the loaded model's bytes resident in GPU memory (`size_vram / size` from the runtime's
running-model list). It is a byte residency ratio, not a measured layer or compute offload fraction. Below 100 part of the
model runs on the CPU. It is available for Ollama only and is empty (not 0) in the CSV when the runtime does not report it.

## LM Studio

BenchRig has no LM Studio client. Benchmark the same GGUF families through Ollama (llama.cpp) as a proxy. Defaults for
context, KV-cache type and offload differ, so results do not transfer one-to-one.

## Limits

Offload results depend on RAM bandwidth; do not generalize them to other hosts. Quantization names (Q4_K_M, IQ2) are
llama.cpp concepts; ONNX int4 is not directly comparable.
