# Historical analyses

These manually written analyses preserve investigations of specific benchmark runs.
They are separate from the reports BenchRig generates in `results/` after a run.
Findings, hypotheses, code line numbers and release recommendations describe the
state at the date of analysis; they are not current validation results.

## 2026-09-23

The analyses cover local runs from September 19 and 22 on WSL2 with an NVIDIA
RTX 5070 and record the subsequent release investigation:

- [Benchmark runs — analysis](2026-09-23/RUN_ANALYSIS.md)
- [Adversarial verification of the two flagged anomalies](2026-09-23/ANOMALY_VERIFICATION.md)

The documents were moved from `results/RUN_ANALYSIS.md` and
`results/ANOMALY_VERIFICATION.md`. Their source JSON, CSV, PNG and generated
Markdown artifacts remain local, Git-ignored files and are not available in a
fresh checkout or on this documentation site. Paths quoted in the analyses are
relative to the repository root and identify the original evidence.

For current benchmarking procedures, see [cross-engine benchmarking](../tutorials/cross-engine-benchmarking.md)
and [hardware telemetry](../hardware-telemetry.md).
