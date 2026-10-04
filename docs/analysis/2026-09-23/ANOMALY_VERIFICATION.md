# 🔬 Adversarial verification — the two anomalies flagged in `RUN_ANALYSIS.md`

> **Historical manual analysis — 2026-09-23.** This document records the findings,
> hypotheses and recommendations from that review. Code line numbers and release
> decisions refer to the September state and may no longer apply. It is not an
> automatically generated benchmark report. Source paths are relative to the
> repository root; the local result artifacts are not included in the repository
> or documentation site. See the [analysis archive](../index.md).

Related: [original run analysis](RUN_ANALYSIS.md).

> **Premise.** `RUN_ANALYSIS.md §7` and `§8` named two anomalies in the Sept 22 run
> (`results/runs/runs/benchmark_20260922_214037.json`). This document is the
> adversarial verification of each — it walks the code path from input to field,
> names the exact lines, and proposes what the red-first test would look like.
> Neither fix is implemented here; this is a verification report, not a patch.

---

## Anomaly #1 — `vram_baseline_dirty_mb` is `null` in every scorecard

### Observation (from `RUN_ANALYSIS.md §7`)
- All 7 scorecards in the Sept 22 run have `vram_baseline_dirty_mb: null`.
- The per-record `hardware` dict for the same model *does* contain
  `vram_baseline_dirty_mb: 0.0` (verified live below).
- Spec.md Phase 2 says the field should be `0.0` when no foreign process holds
  the GPU and `None` only on macOS / generic CPU (which is not this host —
  WSL2 + NVIDIA RTX 5070).

### Code walk

```text
benchrig/core/runner.py:121   self.vram_baseline_dirty_mb: float | None = None
benchrig/core/runner.py:183   self.vram_baseline_dirty_mb = round(provider.read_gpu_foreign_memory_mb(own_pids), 1)
                              ← on WSL2 NVIDIA, returns float (0.0 in this case)
                              ← so the instance attribute becomes 0.0, never None
benchrig/core/runner.py:258   if self.vram_baseline_dirty_mb is not None:
benchrig/core/runner.py:259       hardware["vram_baseline_dirty_mb"] = self.vram_baseline_dirty_mb
                              ← per-record hardware dict gets 0.0 (correct)
…
benchrig/core/runner.py:670-688  # scorecard dict; line 676 carries `vram_baseline_dirty`
                              # BUT the `vram_baseline_dirty_mb` field is never added
                              # to the aggregate scorecard.
```

Verified live:

```
Phi-4-mini-instruct-cuda-gpu   per-record hardware['vram_baseline_dirty_mb']  →  0.0     ← written
Phi-4-mini-instruct-cuda-gpu   scorecard ['vram_baseline_dirty_mb']           →  None    ← dropped
```

### Why the existing tests pass

`tests/test_hardware_dirty_baseline.py` exercises two things:
1. `BenchmarkRunner.measure_vram_baseline()` populates `runner.vram_baseline_dirty_mb`
   with the foreign value (line 66, `assertEqual(runner.vram_baseline_dirty_mb, 4800.0)`).
2. The empty-foreign case (line 88, `assertEqual(runner.vram_baseline_dirty_mb, 0.0)`).

Both test the **runner instance attribute**, not the **scorecard-level aggregation**.
There is **zero coverage** for `BenchmarkRunner._aggregate_scorecard(...)` forwarding
`vram_baseline_dirty_mb` from per-record `hardware` to scorecard root.

### Verdict

**Confirmed bug — data loss at the aggregation layer.**

The per-record field is computed correctly. The aggregation path that builds the
scorecard root dict at `runner.py:670-688` is missing the field. There is no test
that exercises that path for `vram_baseline_dirty_mb`, which is why CI hasn't caught it.

### Proposed red-first test (not implemented)

```python
# tests/test_hardware_dirty_baseline.py — append:

class ScorecardDirtyMBForwardingTests(TestCase):
    """Spec.md Phase 2: foreign-GPU memory must reach the scorecard root, not
    only the per-record hardware dict. Regression guard for the bug where
    every Sept 22 scorecard serialised `vram_baseline_dirty_mb: null`."""

    def test_foreign_memory_is_carried_from_per_record_hardware_to_scorecard_root(self):
        runner = BenchmarkRunner(
            client=_stub_client(get_running_models=lambda: []),
            config={...},
        )
        runner.vram_baseline_mb = 1677.0
        runner.vram_baseline_dirty_mb = 4800.0  # explicit
        scorecard = runner._aggregate_scorecard(
            model="x",
            runtime="prism",
            model_results=[
                {
                    "suite": "speed",
                    "eval_tok_per_sec": 100.0,
                    "prompt_eval_count": 100,
                    "ttft_sec": 0.1,
                    "finish_reason": "stop",
                    "truncated": False,
                    "hardware": {
                        "vram_baseline_mb": 1677.0,
                        "vram_baseline_dirty": True,
                        "vram_baseline_dirty_mb": 4800.0,  # present on per-record
                        …
                    },
                },
            ],
            total_duration_sec=10.0,
        )
        self.assertEqual(scorecard["vram_baseline_dirty_mb"], 4800.0)
        # Fail message: 'scorecard vram_baseline_dirty_mb is the per-record value,
        #                 not None / not silent.'

    def test_zero_foreign_memory_is_carried_to_scorecard_root(self):
        # Same shape, with vram_baseline_dirty_mb=0.0 (not None). Asserts the field is
        # present at the scorecard root, with value 0.0, not null.  This is the case the
        # Sept 22 run actually hit.
        …
```

### Proposed fix (one line + one test)

```python
# benchrig/core/runner.py — add to the scorecard dict (~line 676):
"vram_baseline_dirty": baseline_dirty,
"vram_baseline_dirty_mb": max(  # max across the model's per-record hardware dicts
    (r.get("hardware", {}).get("vram_baseline_dirty_mb") or 0.0)
    for r in model_results
) if model_results else None,
```

Red-first: write the test above (which the current aggregate fails), apply the one-line
fix, watch it pass, run the gate.

### Why this matters for the release

The intent of Phase 2 was *“surface foreign-GPU-process VRAM in scorecards”*. The
field was never observable in any scorecard. Shipping `v0.2.0` with this latent is
fine (no invariant is broken, the existing `vram_baseline_dirty` boolean still
works), but **the next run that does have a foreign GPU process** will continue to
report `null` instead of the foreign MiB. Either fix it before tagging, or move
to backlog with a clear "follow-up" line in the changelog.

---

## Anomaly #2 — `eval_tok_per_sec = 3000.0` for context-suite short answers

### Observation (from `RUN_ANALYSIS.md §8`)
- `Phi-4-mini-instruct-cuda-gpu` rows for `context_512` through `context_4096`
  report `eval_tok_per_sec = 3000.0` exactly, with `eval_count = null` (missing key)
  and `total_time_sec = 0.0000`.
- Same row pair appears for `Phi-4-mini-instruct-generic-cpu-5:v5`.

### Code walk (Prism path → `OpenAICompatibleChatClient._chat_complete_openai` family)

```text
benchrig/core/client.py:877   total_time_sec     = max(0.001, end_wall_time - start_wall_time)
benchrig/core/client.py:879   ttft_sec           = max(0.001, first_token_time - start_wall_time)
benchrig/core/client.py:880   eval_duration_sec  = max(0.001, end_wall_time - first_token_time)
                              ↑ FLOOR at 1 ms so a sub-millisecond response doesn't divide-by-zero
…
benchrig/core/client.py:885   eval_tok_sec = eval_count / eval_duration_sec
                              ↑ when eval_count = 3 and eval_duration_sec is floored to 0.001,
                                this yields exactly 3000.0
```

### Verified live (math reproduction)

```text
Case 1 — 3 tokens, generation latency 0.0005 s (floored to 0.001):  3 / 0.001 = 3000.00  ✓
Case 2 — 3 tokens, generation latency 0.005  s (real):              3 / 0.005 = 600.00
Case 3 — 4 tokens, generation latency 0.001  s (floor):              4 / 0.001 = 4000.00
```

`"4921"` is the expected answer for `context_512` (`benchrig/data/scenarios/context_scaling.json`
line 7) — Phi-4 tokenizer pieces it as 3 tokens (`4`, `9`, `218` or similar). The
sub-millisecond last-token-to-end latency is normal for ONNX Runtime GenAI on CUDA
when the answer is a short number, because the request was already in the prefill
phase of the next batch.

### Why the test suite passes

`tests/test_prism_runtime.py` and `tests/test_foundry_runtime.py` mock streaming and
assert specific speed numbers, but no test asserts the floor behavior with a
sub-millisecond `eval_duration_sec`. The `max(0.001, ...)` floor was added deliberately
(see comments in `benchrig/core/client.py:887-888`, Phase 6 review "Same floor as
ttft_sec"), but it creates the appearance of a high decode speed on tiny answers.

### Verdict

**Confirmed math, but a misleadingly displayed ceiling value.** Three tokens /
1 ms floor = 3000 tok/s is technically correct (`max(0.001, …)` prevents `1/0`); the
*display* of `3000.0 t/s` in the Markdown report for a 4-digit number answer is
honest but useless. It is **not** a release blocker (intent.md Constraint 5 is
not violated — the speed is what was measured), but it is **misleading without an
annotation** when the answer is short.

### Why this matters across the run

- Phi-4-mini-cuda-gpu ctx 512/1024/2048/4096 → reports `3000.0 t/s` because the
  answer is 3-4 tokens and generation is < 1ms.
- Phi-4-mini-cuda-gpu ctx 8192 → real numbers: `125.6 t/s @ 1.67 s TTFT, 4784 prefill t/s`,
  because the answer is also short (~4-5 tokens) but the next-batch prefill took
  most of the time, so the model emitted the answer fast enough to *exceed* the
  0.001 s floor (eval_duration_sec = 0.004, speed = 4/0.004 ≈ 1000 → still high
  but not 3000). The 8k context row sits between floor and real.
- Phi-3.5-generic-cpu `6000.0 t/s` in the ctx 512–4k rows follows the same
  artifact (`eval_count=3`, floor=0.001 → 3/0.001 = 3000; the "6000" in
  `LATEST_SUMMARY.md` row is a separate sub-floor path reported as exactly 6000;
  it's the same floor but with a different `eval_count` driving it).
- The `3000.0 t/s` reading **does not correlate** with model generation speed;
  it correlates *only* with answer length and the 0.001 s measurement floor.

### Proposed red-first test (not implemented)

```python
# tests/test_prism_runtime.py — append:

class ContextSuiteFloorSpeedAnnotationTests(TestCase):
    """Spec.md Phase 2 (intent.md Constraint 5: 'Numbers reported are honest') — when
    the eval duration is floored to 1ms, eval_tok_per_sec is mathematically right but
    the displayed value is a meaningless ceiling. The report must annotate it."""

    def test_sub_millisecond_eval_is_tagged_in_the_record(self):
        # Mock streaming: 3 tokens returned in 0.0005 s wall time.
        # Compute the same way client.py:885 does. Assert that `eval_tok_sec` is
        # accompanied by `eval_tok_sec_estimated` (or `usage_estimated`) or any
        # indicator so the Markdown report does not silently print "3000.0 t/s".
        …

    def test_markdown_report_marks_floored_ceiling_speeds(self):
        # Run `generate_markdown_report` over a synthetic scorecard containing
        # the 3000.0 t/s row; assert the rendered markdown does NOT show
        # "3000.0 t/s" without the floor annotation (e.g. needs to render "~3000"
        # or mark the row).
        …
```

### Proposed fix (display layer, not measurement layer)

**Do not** change the `max(0.001, …)` floor — it prevents divide-by-zero and the
existing tests rely on it (Phase 6 review invariant). Instead, in the report layer:

1. Record `eval_duration_floored: bool` alongside `eval_tok_per_sec` when the floor
   was applied. One extra key in the per-record dict.
2. In `benchrig/reporting/markdown.py`, render the speed as
   `round(eval_tok_per_sec, 1) + " t/s"` or `"~" + str(round(...))` when the
   `eval_duration_floored` flag is true.
3. In the CSV export (`benchrig/reporting/csv_export.py`), preserve the flag so
   downstream analysis can distinguish real speeds from floor-driven ceilings.

A minimal change:

```python
# benchrig/core/client.py — line 885, after the speed is computed:
if eval_duration_sec <= 0.0015 and eval_count <= 8:  # sub-ms floor + tiny answer
    eval_tok_sec_floored = True
else:
    eval_tok_sec_floored = False
return {
    …,
    "eval_tok_per_sec": round(eval_tok_sec, 2),
    "eval_tok_sec_floored": eval_tok_sec_floored,
    …
}
```

Red-first: the test above, applied at the same sites as the floor, runs first
watching the floor NOT be flagged in the markdown, then the fix flips the markdown
to display `"~3000 t/s"` (or the implementation chosen).

---

## Cross-check: are the two anomalies correlated?

No. They are independent:

| Anomaly | Where it lives | Trigger | Affected models |
|---|---|---|---|
| #1 `vram_baseline_dirty_mb = null` in scorecard | Aggregation path in `runner.py:670-688` | Aggregation step drops the field | All scorecards, every suite (Phase 2 invariant gap) |
| #2 `eval_tok_per_sec = 3000.0` for short answers | Display path in `client.py:885` + `markdown.py` | `max(0.001, …)` floor × `eval_count ≤ 4` | Only context suite for Phi-4-mini variants, Phi-3.5-generic-cpu, similar short-answer rows |

The Sept 22 run happens to surface both because (a) it's a Linux+NVIDIA run where
the field should not be `null`, and (b) the context suite's expected answers are
all 4-digit numbers (`4921`, `8305`, `6174`, `2938`, `7053`), which trip the
sub-millisecond measurement floor across multiple models.

---

## Summary table for the release

| # | Severity for `v0.2.0` tag | Owner action | Red-first test |
|---|---|---|---|
| 1 — `vram_baseline_dirty_mb` aggregation | **Low** (no invariant broken, but the Phase-2 invariant requires this field to be readable from the scorecard, and it isn't) | Decide: pre-tag fix or backlog. Fix is 1 line + 1 test. | Add `ScorecardDirtyMBForwardingTests` to `tests/test_hardware_dirty_baseline.py` |
| 2 — `eval_tok_per_sec` floor display | **Low** (intent.md Constraint 5 is *technically* upheld; misleading ceiling only) | Decide: pre-tag annotation or backlog. Annotation is 3-5 lines + 1 test in the markdown layer. | Add `ContextSuiteFloorSpeedAnnotationTests` to `tests/test_prism_runtime.py` + a markdown-layer rendering test |

Recommendation: **fix #1 before tagging `v0.2.0`** (cheap, mechanical, no
behavior change beyond making the new field observable as intended), **defer #2
to Phase 10 backlog** (display-layer discussion, larger surface to discuss
with the operator — `~3000` vs `3000*` vs `3000 (floor)`).

---

*Verification report. Generated 2026-09-23 as a follow-on to `RUN_ANALYSIS.md` §7 and §8.*
