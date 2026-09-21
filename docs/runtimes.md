# Runtimes

BenchRig measures the same scenarios on several inference runtimes and reports them side by side. Pick one with
`--runtime`, or address a model directly with a `runtime:model` prefix in `--models`.

| Runtime | `--runtime` | Prefix | What it is | Config section |
| :--- | :--- | :--- | :--- | :--- |
| Ollama | `ollama` | `ollama:` | `llama.cpp` (GGUF) behind Ollama's REST API | `ollama` |
| Microsoft Foundry Local | `foundry` | `foundry:`, `ms-foundry:` | ONNX Runtime GenAI behind the Foundry daemon | `foundry` |
| Direct ONNX Runtime GenAI | `onnx-gpu` | `onnx-gpu:`, `onnx:` | ONNX Runtime GenAI in the BenchRig process (CUDA) | `onnx` |
| Prism | `prism` | `prism:`, `prism-local:` | One OpenAI-compatible endpoint in front of ONNX Runtime GenAI (CUDA or CPU) and Ollama | `prism` |
| All | `all` | | Every runtime that is reachable | |

```bash
benchrig --check                                              # which runtimes are reachable, and what they serve
benchrig --runtime ollama --models qwen2.5-coder:7b
benchrig --models ollama:qwen2.5-coder:7b,prism:phi-4-mini    # mix runtimes in one run
```

## Ollama

Talks to Ollama's native API (`/api/generate`) and measures time to first token from the stream. The default endpoint is
`http://localhost:11434`; see [`ollama` configuration](configuration.md).

## Microsoft Foundry Local

Talks to the Foundry daemon's OpenAI-compatible API. By default the port is read from `~/.foundry/daemon.json` because Foundry
starts on an ephemeral port (`auto_detect_port: true`); models are loaded and unloaded with the `foundry` CLI. See
[Foundry GPU setup](tutorials/foundry-gpu-setup.md) for making it use an NVIDIA GPU on WSL2.

## Direct ONNX Runtime GenAI

Loads a model folder (one that contains `genai_config.json`) inside the BenchRig process, with the CUDA execution provider when
available. Install the extra and put models in `$BENCHRIG_MODEL_DIRS`, `./models` or `~/.benchrig/models`:

```bash
pip install "benchrig[onnx-gpu]"
benchrig --runtime onnx-gpu --models Phi-4-mini-instruct-cuda-gpu --suite coding
```

## Prism

[Prism](https://github.com/senssei/prism-local) (`prism-local`) is a local server that puts ONNX Runtime GenAI and Ollama behind
one OpenAI-compatible endpoint on a fixed port. It also lets you choose the execution device (`--device auto|cuda|cpu`), which
makes it a convenient way to benchmark ONNX models on WSL2, where the Foundry CLI can fail to detect the GPU
(see Prism's own evaluation notes).

```bash
pip install prism-local && prism serve            # http://127.0.0.1:5272/v1
benchrig --check                                  # lists the models Prism serves and the engine behind each
benchrig --runtime prism --models prism:phi-4-mini --suite coding
```

**How BenchRig talks to it**

- The endpoint comes from `prism.base_url` (default `http://127.0.0.1:5272/v1`) and is never auto-discovered. The `foundry` CLI is
  never used, so a Foundry daemon running on the same machine cannot capture the requests.
- If the server was started with `--api-key`, set `PRISM_API_KEY` (or `prism.api_key`); it is sent as a bearer token.
- Prism loads models on the first request and has nothing to unload, so `load_model` only checks that the server lists the model.
- `--pull-recommended --runtime prism` runs `prism pull <model>` and needs the `prism` CLI on `PATH`.

**Models and engines**

Prism also proxies installed Ollama models as `ollama:<name>`. The engine that serves each model is taken from the server's
`owned_by` field (falling back to the `ollama:` prefix) and shown in reports, so a Prism-proxied Ollama model is not labelled as
ONNX.

- `--models installed` with `--runtime prism` or `--runtime all` skips the `ollama:` proxies: benchmark those natively with the
  `ollama` runtime. To measure the proxy itself, ask for it explicitly: `--models prism:ollama:qwen2.5-coder:3b`.
- A `--pair` may carry a `prism:` model name next to `ollama:` and `foundry:`; with `--baseline` and `--runtime prism` it falls
  back to the pair's `onnx:` and then `foundry:` names.

**Token counts and the model-only memory figure**

- Prism started from a version with `stream_options.include_usage` sends exact token counts in the last streamed chunk, together with its device
  telemetry. From an older Prism, or any server that sends no `usage`, BenchRig estimates the prompt tokens (word count x 1.3) and counts streamed chunks as
  generated tokens; such results are marked `usage_estimated` and the reports say so, because prefill and decode speeds then depend on the guess.
- Prism holds one model at a time and has no unload. When a model is still loaded as the next one is benchmarked, the "model memory" figure is left out
  (`vram_baseline_dirty`), because peak minus a baseline that contains the previous model would be too small. Restart `prism serve` between models for it.

**GPU memory on long prompts**

ONNX Runtime GenAI's GPU memory grows with the prompt length (about 1.4 MB per token for Phi-4-mini) and is not released afterwards, so
the context suite's large steps drive Prism's peak memory up (10.6 GB of model memory in the comparison against 3.8 GB on Ollama), which can
trip the memory warning and its composite-score penalty. Starting Prism with `PRISM_PREFILL_CHUNK=256` bounds it (6.6 GB instead of
11.7 GB at 4500 prompt tokens for Phi-4-mini), but the cost differs by model: on two other models the peak fell by 54% and 5% while time to first
token rose by 110% and 33% (see the [Prism docs](https://senssei.github.io/prism-local/devices/#gpu-memory-and-long-prompts)). It is off by
default; benchmark with and without it if memory matters to your comparison, and note which one you used.

**Tuning the Prism server with environment variables**

BenchRig does not read these — Prism does, when you start it with `prism serve`. Set them before the
server starts and the change takes effect for every model you benchmark against it. Trade-offs here move
with the prism-local release, so always cross-check with the [Prism docs](https://senssei.github.io/prism-local/).

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `PRISM_PREFILL_CHUNK` | `1024` | Prompt tokens processed per step. `0` or `off` processes the whole prompt at once; small values (e.g. `256`) bound peak memory on long prompts at the cost of higher TTFT — see the note above. |
| `PRISM_THREADS` | *(unset, auto)* | Intra-op parallelism for ONNX Runtime GenAI. Set to a positive integer when you want a deterministic thread count; leave unset to let the runtime pick. |
| `PRISM_DEVICE` | `auto` | Execution provider Prism picks when serving ONNX models: `auto` (CUDA when present, else CPU), `cuda`, or `cpu`. Read every result's `device` field rather than trusting the model name. |

**Which device ran it**

Each result record from Prism carries a `device` field (`cuda` or `cpu`): the execution provider Prism reports for that request, from
the response's `telemetry` or, for streamed responses, from `/health`. If Prism reports nothing the field is omitted; treat a
missing value as *unknown*, not as GPU.

!!! warning "Model names do not tell you the device"
    With Prism's default `--device auto`, ONNX models are run on the CUDA execution provider when it is available, **including
    models whose name says `generic-cpu`**. Measured with `qwen3-0.6b-generic-cpu-4:v4` on an RTX GPU under WSL2: `--device auto`
    used the GPU (about 1.4 GB more GPU memory, 33% utilisation) and reported `cuda`; `--device cpu` left GPU memory untouched
    and reported `cpu`. For this 0.6B model the CPU run was faster (about 57 vs 24 tokens/s), so the choice can change a
    comparison. Start `prism serve` with an explicit `--device cuda` or `--device cpu` when the result depends on it, and read the
    `device` field rather than the model name.
