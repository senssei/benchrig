# 🧪 Benchmark runs — analysis

> **Historical manual analysis — 2026-09-23.** This document records the findings,
> hypotheses and recommendations from that review. Code line numbers and release
> decisions refer to the September state and may no longer apply. It is not an
> automatically generated benchmark report. Source paths are relative to the
> repository root; the local result artifacts are not included in the repository
> or documentation site. See the [analysis archive](../index.md).

Follow-up: [adversarial anomaly verification](ANOMALY_VERIFICATION.md).

> **Scope:** Triangulated analysis of every JSON/Markdown artifact under
> `results/` and `results/runs/`. Two end-to-end runs are on disk:
>
> | Run | Timestamp | Scorecards | Source artifact |
> |---|---|---:|---|
> | Main (Ollama + Phi-4-mini-on-Prism) | `2026-09-19 20:44:54` | 10 | `results/LATEST_SUMMARY.md`, `benchmark_20260919_204346.json`, `benchmark_20260919_204454.json`, `1TO1_COMPARISON_REPORT.md` |
> | Prism only (flagged) | `2026-09-22 21:40:37` | 7 | `results/runs/runs/benchmark_20260922_214037.json`, `results/runs/LATEST_SUMMARY.md`, `results/scorecards.csv`, `chart.png` |
> | Smoke (Ollama, single model) | `2026-09-22 14:38` | 1 | `results/smoke/` |
>
> Host: `WSL2 Linux 6.6.87.2`, NVIDIA RTX 5070 (12 227 MB), 16 vCPU, 31.3 GB RAM,
> driver 616.92.

---

## 1. Headline

Only **1 of 7** Prism-target scorecards in the Sept 22 run is a clean, fully-benchmarked model
(`mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32`, composite **70.9**). The other six break into
three clear failure modes, **each of which corresponds to a shipped Phase of BenchRig:**

| Failure mode | How many | Phase that explains / mitigates it |
|---|---:|---|
| Coding `IndentationError`/`SyntaxError` (0 of 4 PASS) on Phi-4-mini variants | 3 | **Phase 9.1 + 9.2** (diagnostic fields + `textwrap.dedent` hardening) — *cause not confirmed* |
| Composite `0.0`, no scenario ever ran (capacity short-circuit) | 2 | **Phase 4.6** (persistent `insufficient_resources` skip + `⚠ Skipped` notice) |
| Generic-cpu slowness on CUDA (4.4 tok/s) | 1 | **Phase 2.1** (one-line warning) + Phi-3.5 actually still completes (`reasoning=50.0`), so the warning is the operator signal, not a hard skip |
| `qwen3-0.6b` Pariah run (`composite=1.9`, all rows `0.0 t/s`) | 1 | Outside any Phase; model didn't generate. Listed under "open anomalies" §6. |

**Net verdict for the release:** the code is doing what it was designed to do on five of the seven
runs. The two diagnostics-rich runs (Phi-4-mini 0% coding and `vram_baseline_dirty_mb=None`) are
the only ones that *don't* line up cleanly with a shipped feature, and they point at concrete
follow-ups rather than release blockers.

---

## 2. The one good run (Sept 22): `mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32`

| Metric | Value |
|---|---|
| Composite | **70.9** |
| Coding pass rate | 64.7% (11/17 assertions across 4 scenarios) |
| Reasoning accuracy | 50.0% (6/12) |
| Avg decode | 101.2 tok/s |
| Prefill (eff.) | 926.1 tok/s |
| TTFT | 0.29 s |
| Peak VRAM | 8 084 MB (over 1 677 MB clean baseline → **6 407 MB** model) |
| Cold start | 25.4 s |
| Tokens / J | 0.67 |

- The only run with both a non-zero coding and a non-zero reasoning score.
- Coding: passed `merge_intervals` (5/5), `valid_brackets` (6/6); failed `flatten_dict`
  (`SyntaxError: invalid syntax`) and `lru_cache` (`IndentationError: unindent does not match…`).
- Reasoning: surprisingly weak. `bat_and_ball` and `set_theory_probability` correct; classic
  "different answer style" misses on `multiples_of_three_or_five` (`33+25+7=65` — counted 3 × 5 +
  7+35 = 23 wrongly), `letters_in_one_word` (`4` for strawberry — same model class as
  the Sept 19 Ollama cohort which scored it correctly: 50/50). This is a model-quality gap, not a
  BenchRig artifact.

---

## 3. The flagged regression: Phi-4-mini-variants **0% coding** (Phase 9 territory)

Three runs that were functional in the Sept 19 Ollama cohort **scored 0 / 4 on every coding
scenario** on Sept 22 through Prism/ONNX:

| Model | Sept 19 (Ollama) | Sept 22 (Prism ONNX) |
|---|---:|---:|
| `Phi-4-mini-instruct-cuda-gpu` | coding PASS 4/4, 3/3, 3/3, 3/3 = **100%** | **0%** (4 × `IndentationError`) |
| `Phi-4-mini-instruct-generic-cpu-5:v5` | (did not run) | **0%** (4 × `IndentationError`) |
| `Phi-3.5-mini-instruct-generic-cpu-2:v2` | (did not run) | **0%** (2 × `IndentationError`, 1 × `SyntaxError`) |

The exact error strings are present-day in the runner (`SyntaxError: invalid syntax`,
`IndentationError: unindent does not match any outer indentation level`,
`IndentationError: unexpected indent`) — they appear 8 / 4 / 3 times across the 27
`sandbox_error` rows of the run.

**Phase 9 status (per `spec.md` Phase 9 and `plan.md`):**
- **9.1 (shipped):** the failing records now carry `extracted_code` and
  `response_excerpt` (first 400 chars, head-truncated — coding prompts put the code **first**, so
  the review-corrected direction is the only one that keeps the diagnostic useful).
- **9.2 (shipped):** `extract_python_code` `textwrap.dedent`s a single matched `` ```python ``
  fence. The review caught the original-version regression on multi-fence class+method patterns
  and scoped the dedent to `len(matches) == 1`.
- **Confirmed-not-confirmed:** the same `(model, scenario)` pairs were replayed live against the
  same Prism server, same prompts, same options, same `extract_python_code` — **15 / 15 passed.**
  Item 9.2 is therefore hardening, not a confirmed root-cause fix for this regression. Do not
  describe it as a confirmed fix in the release notes.

**What this means for the next run:** Phase 9.1 makes the next zero-passed run **debuggable from
the saved JSON alone** (open `benchmark_*.json`, look at `response_excerpt`, see what came back).
Phase 9.2 *might* be the gap, *might* be silent hardening. Re-running the same benchmark with the
release pipeline (Phase 8 progress bar + Phase 9.1 diagnostics captured for every failing record)
is the cleanest next step.

---

## 4. Capacity short-circuit working as designed (Phase 4.6)

| Model | scorecard | how it shows up |
|---|---|---|
| `Phi-4-generic-cpu-2:v2` | `composite=0.0`, `runs=1`, **`cold_start_sec=None`**, all metrics 0.0 | "no scenario ever produced a real number" |
| `qwen2.5-coder-7b-instruct-generic-cpu-4:v4` | same | same |

Both are likely second models in the run that hit `insufficient_resources` (`Insufficient VRAM
… < 9295 MB required, Held by PID 12584`) on the very first scenario, then Phase 4.6's
`BenchmarkRunner._capacity_exhausted_reason` short-circuited the remaining scenarios for the
remaining suites, so the only HTTP call recorded is the failing first one. **This is the desired
behavior** (the alternative would have been dozens of `503` retries per model — the original
operator complaint that triggered Phase 4.6).

The Phase 4.6 CLI-side complement (`⚠ Skipped — model does not fit in available VRAM: …`,
`benchrig/cli.py::_suite_skip_notice`) printed to the terminal for those two models on the live
run (per spec.md Phase 9 status note). **Recommendation:** when these two scorecards appear in a
later run **after Phase 5 ships** (`POST /v1/unload` between models), the residual foreign VRAM
should drop noticeably — keep an eye on whether the `insufficient_resources` *still* triggers
on Phi-4-generic-cpu-2:v2, since the model itself is smaller than what other models needed.

---

## 5. Generic-cpu slowness, confirmed live (Phase 2)

- `Phi-3.5-mini-instruct-generic-cpu-2:v2`: **4.4 tok/s average decode**, prefill 89.4 tok/s, TTFT 5.15 s.
  Total run time ≈ 25 minutes.
- At 4 096 tok context it still finds the hidden fact (✅), but TTFT climbs to **14.7 s**; at
  8 192 tok it times out the fact-retrieval step (❌, peak VRAM 88% — `Near Memory Limit`).
- `Phi-4-mini-instruct-generic-cpu-5:v5`: **128.5 tok/s**, TTFT 0.21 s — same quantization class,
  same `generic-cpu` provider, **30× faster**. Reasoning still 50%, but coding `0/4` (same
  Phase 9 issue as the cuda-gpu variant).

> Phase 2.1's `benchrig.core.runtimes.warning_for_provider(...)` should be firing on both
> variants in this cohort. Verification: re-run `tests/test_warnings.py` and observe the
> live CLI. The warning text was committed precisely so an operator does not have to wait 25
> minutes for a 4.4 tok/s run to discover Phi-3.5-generic-cpu is unusable on this hardware.

**Operator takeaway:** on RTX 5070 + CUDA, `generic-cpu` is unusable for **Phi-3.5-mini**
specifically. Phi-4-mini at `generic-cpu` is fast enough; the cost is in quality (coding 0%),
not speed.

---

## 6. Cross-runtime comparison (Sept 19 Ollama vs Sept 22 Prism, same Phi-4-mini)

`1TO1_COMPARISON_REPORT.md` (Sept 19) pinned the same model on two engines head-to-head:

| Metric | `phi4-mini:latest` [Ollama] | `Phi-4-mini-instruct-cuda-gpu` [Prism ONNX] | Delta |
|---|---:|---:|---|
| Composite | 90.0 | 84.0 | Ollama +6.0 |
| Coding pass rate | 100.0% | 100.0% | tie |
| Decode (tok/s) | 166.2 | 137.6 | Ollama 1.2× |
| Prefill (eff. tok/s) | 2 631.3 | 1 312.4 | Ollama 2.0× |
| TTFT (s) | 0.13 | 0.35 | Ollama 2.7× |
| Peak VRAM | 4 809 MB | 11 432 MB | (Prism retains previous model ⇒ dirty baseline) |
| Model VRAM (Δ) | 3 732 MB | 10 424 MB | (same caveat) |

**Same model, same prompt, same `seed: 42`, same `--runs N`, same host, ~3 days apart — the
Ollama cohort ran Sept 19 with 100% coding PASS, the Prism cohort on Sept 22 ran the
**identical ONNX-side scenario set** to 0% coding PASS. The composite gap came from there:
Prism-side composite dropped from 84 (Sept 19) to 45 (Sept 22) entirely on the coding leg.

- Decode and TTFT numbers are within run-to-run noise (Sept 22 Phi-4-mini-cuda-gpu posted
  126.2 tok/s vs 137.6 on Sept 19 — `−8%`, comparable to a prefill-vs-decode mix shift).
- The *behavior* change is binary: 100 % → 0 % on coding, all four scenarios, same
  `IndentationError` text → **dedent-extraction gap surfaced**, OR non-determinism surfaced
  (intent.md Constraint 4 says seeding should make this deterministic; spec.md Phase 9's
  "alternative hypothesis" — `seed: 42` doesn't actually pin ONNX/CUDA generation determinism
  under these conditions). Phase 9.1 now captures enough diagnostics to disambiguate
  **on the next run** instead of forcing live repro.

---

## 7. New field `vram_baseline_dirty_mb` returns `None` for all 7 scorecards

The Phase 2.2 invariant is `vram_baseline_dirty_mb: float | None` populated from
`BaseHardwareProvider.read_gpu_foreign_memory_mb(own_pids)` (nvidia-smi
`--query-compute-apps=pid,used_memory`). On this **WSL2 + NVIDIA** run, every scorecard has
`vram_baseline_dirty_mb=None`:

```
mistral-7b-instruct-v0.2-cuda-int4-rtn-block-32   baseline=1677   dirty_mb=None   peak=8084
Phi-4-mini-instruct-cuda-gpu                     baseline=8084   dirty_mb=None   peak=9114
Phi-4-mini-instruct-generic-cpu-5:v5             baseline=9114   dirty_mb=None   peak=9131
Phi-4-generic-cpu-2:v2                           baseline=9124   dirty_mb=None   peak=3945
qwen2.5-coder-7b-instruct-generic-cpu-4:v4       baseline=3944   dirty_mb=None   peak=3982
Phi-3.5-mini-instruct-generic-cpu-2:v2           baseline=3941   dirty_mb=None   peak=10793
qwen3-0.6b-generic-cpu-4:v4                      baseline=6481   dirty_mb=None   peak=7927
```

**Suspected causes (in order of likelihood):**
1. `read_gpu_foreign_memory_mb(...)` is never called for **ONNX Runtime GenAI** runtimes
   (it lives in the Ollama/Foundry provider paths; Prism doesn't inherit it). Spec.md Phase 2
   documents nvidia-smi only; cross-engine parity for this field is an undocumented gap.
2. The provider returns `None` on `nvidia-smi` failure (driver/cache mismatch), and the field
   silently maps to JSON `null`. The test in `tests/test_hardware_dirty_baseline.py` exercises
   the helper in isolation, not the wiring into `BenchmarkRunner.measure_vram_baseline` for
   every runtime.
3. The phase's intent ("record the foreign-process GPU memory on the same GPU the model
   will use") is moot for Prism because Prism can't unload between models anyway — so the
   field would always read 0.0 for the *next* model even with perfect nvidia-smi support.

**Recommendation:** before tagging `v0.2.0`, add an assertion (or a one-line log) at
`BenchmarkRunner.measure_vram_baseline` that the field is set to a number, not `None`, for any
runtime on an NVIDIA host — even `None` is technically allowed by the I-anything spec, but
shipping a release where every foreign-VRAM cell is `None` undermines the field's intent.

---

## 8. Context suite shows a 3 000 tok/s sentinel (measurement anomaly)

For `Phi-4-mini-instruct-cuda-gpu` and `Phi-4-mini-instruct-generic-cpu-5:v5` at ctx ∈ {512,
1024, 2048, 4096} the per-step context suite reports `eval_tok_per_sec = 3000.0` while
`prompt_eval_count = 307` is real and `eval_count = None`:

```
Phi-4-mini-instruct-cuda-gpu   context_512   prefill=1872   eval_tps=3000.0   prompt=307   eval=None
```

This is **not real** — the model emits one or two opening tokens and the timer starts, the
context-suite scenario ends in milliseconds. The same `(model, ctx=8192)` row reads
`eval=125.5 tok/s` (real) and `prefill=4784 tok/s` (also real), so the issue is bounded to the
short-context cases where `eval_count` is tiny. The Markdown report's "Decode Speed" column
shows `3000.0 t/s` for those rows, which **contradicts intent.md Constraint 5 ("Numbers
reported are honest")** — the field is being shown without the `usage_estimated` flag that
covers this kind of measurement gap. Not a release blocker, but a flag for a Phase 10 follow-up:
clamp display values to `tok/s_real = eval_count / max(time, floor_time)`, or carry
`eval_estimated: true` so the table can show a `~` prefix.

By contrast, `qwen3-0.6b-generic-cpu-4:v4` shows the correctly reported `0.0 t/s` (no model
output) for every context step — that one is honest, just catastrophic.

---

## 9. Per-suite scoreboard (Sept 22 run)

| Suite | Working models | Failing / skipped | Notes |
|---|---|---|---|
| **speed** | 5/7 | 2 (qwen3-0.6b and qwen2.5-coder-7b; both composite=0) | Two composite=0 scorecards clearly started, hit `insufficient_resources` mid-suite, were skipped |
| **coding** | 1/7 fully passing (mistral: 64.7%), rest 0–64% | All Phi-4 / Phi-3.5 variants `0/4` with `IndentationError` | Phase 9 fix-the-diagnostics shipped; root cause unproven |
| **reasoning** | 4/5 scorecards report `50%` (correct on 6/12 scenarios) | mistral 50%; Phi-4-mini variants 50%; Phi-3.5 50%; qwen3 0% (model didn't emit) | Same weak-spots as Sept 19: classic `multiples_of_three_or_five` and `letters_in_one_word` reg-failed |
| **polish** | 4/5 | qwen3 0% | All non-qwen scorecards 100% Polish |
| **context** | 4/7 fully found-the-fact in long prompts | 3 (Phi-3.5 at 8 k lost the fact; both qwen3 and qwen2.5-coder-7b got 0/5; Phi-4-mini variants missed only at 8 k) | §8 anomaly on short-context eval speeds |

---

## 10. Recommendation matrix for the release

| # | Action | Owner | Blocking? |
|---|---|---|---|
| A | Commit the working-tree Phase 8 / Phase 9 changes | operator | **Yes** (release gate) |
| B | Move `[Unreleased]` → `[0.2.0] - 2026-09-23` in `CHANGELOG.md` | operator | **Yes** (tag) |
| C | Tag `v0.2.0` | operator | **Yes** (cut) |
| D | PyPI Trusted Publisher publish to `pypi` env | operator | optional (later) |
| E | GitHub Pages for `benchrig` and `prism-local` | operator | optional (later) |
| F | Create `senssei/local-coders` | operator | optional (later) |
| G | Re-run the Sept 22 Prism benchmark **with this release installed** to: (a) verify the Phi-4-mini coding regression either disappears (Phase 9.2 was the gap) or is now diagnosable from saved JSON (Phase 9.1); (b) confirm Phase 5's `POST /v1/unload` actually drops the foreign VRAM between models; (c) capture real `vram_baseline_dirty_mb` values now that each model starts from clean VRAM | operator | strongly recommended |
| H | Phase 10 backlog item: cross-runtime parity for `vram_baseline_dirty_mb` (currently `None` on Prism); see §7 | spec → new phase | follow-up |
| I | Phase 10 backlog item: clamp / annotate `eval_tok_per_sec` when `eval_count` is tiny or missing in the context suite; see §8 | spec → new phase | follow-up |
| J | Phase 10 backlog item: Phase 9 **alternative hypothesis** audit — does `seed: 42 + temperature: 0.1` make coding/reasoning truly deterministic on ONNX Runtime GenAI + CUDA under host load? Compare 3 fresh `--runs 3` runs after the release. | spec → new phase | follow-up (still open per spec.md Phase 9) |

---

## 11. Quick links

- Run JSON (Sept 22, Prism): `results/runs/runs/benchmark_20260922_214037.json`
- Run Markdown (Sept 22, Prism): `results/runs/LATEST_SUMMARY.md`
- Run chart (Sept 22, Prism): `results/chart.png`
- Scorecards CSV (Sept 22): `results/scorecards.csv`
- Run Markdown (Sept 19, Ollama+Prism): `results/LATEST_SUMMARY.md`
- 1:1 comparison (Sept 19, Ollama vs Prism on Phi-4-mini): `results/1TO1_COMPARISON_REPORT.md`
- Phase 9 investigation: `spec.md` §"Phase 9"
- Phase 4.6 short-circuit contract: `plan.md` Phase 4 item 4.6
- Phase 5 unload contract: `spec.md` I7; `plan.md` Phase 5

*Generated as part of the v0.2.0 release evaluation, 2026-09-23.*
