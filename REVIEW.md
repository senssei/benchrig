# Review policy (`REVIEW.md`)

## 1. Who decides

Agents propose; the **operator decides** what merges and what ships (the operator gates are in `AGENTS.md`). A change that passes every gate is not done until it has been
reviewed, and the review is never done by the session that wrote the change.

```text
Plan approved -> test red -> code green -> gate exit 0 -> independent review -> operator decision -> commit / release
```

## 2. Independent review

Review with fresh context: a separate subagent or a new session that is given only `intent.md`, `spec.md`, the `plan.md` items
under review, this file, and the diff (`git diff $(git merge-base <base> HEAD)` plus untracked files, `<base>` being `base` in
`sdlc.toml`). It reports findings ranked by
severity, each with `file:line` and a concrete failing input, and no praise. The author fixes findings test-first; the reviewer
does not edit code.

## 3. Checklist

### A. Evidence
- [ ] `python3 scripts/sdlc_check.py` exited `0` in this session.
- [ ] Every new or changed behavior has a test that was seen failing first (`sdlc_check.py --red`).
- [ ] No test was weakened, skipped or deleted to get green.

### B. Spec
- [ ] The behavior is written in `spec.md` (or the normative doc it points to) and the code matches it.
- [ ] No invariant is broken. Touching one needs operator approval.

### C. Patch
- [ ] The diff contains only what the plan item covers; no drive-by refactors.
- [ ] No dependency the project rules forbid.

### D. Security and cleanliness
- [ ] No injection into shell commands, queries or paths; no path traversal; no secrets in code or logs.
- [ ] Input is validated at the boundary; errors keep their documented shape.

### E. Project checks
- [ ] `ruff check .` and `ruff format --check .` both pass (see `AGENTS.md` §5).
- [ ] `python3 -m build` succeeds (the package builds a wheel and sdist).
- [ ] Docs references in `README.md` and `mkdocs.yml` stay consistent with the public CLI (`benchrig --help`).

## 4. Commands

```bash
python3 scripts/sdlc_check.py                        # the gate
python3 scripts/sdlc_check.py --red <test-id>        # red-first
git diff $(git merge-base <base> HEAD)               # what the reviewer sees (<base> from sdlc.toml)
git config core.hooksPath .githooks                  # opt in to the pre-commit gate
```

## Shared SDLC review — Codex

Read `AGENTS.md`, `intent.md`, `spec.md` and the current phase in `plan.md`. Review the actual working diff, including untracked files, with independent context. Findings must name severity, file/line and a reproducible scenario. Record the exact gate command, exit status and any sandbox limitations; a blocked check is not a pass. Existing operator authorization covers the requested implementation; commits, pushes and releases need their own authorization.

Project checks: Verify benchmark isolation and result provenance; runtime changes update CHANGELOG.md.


## Item 12.4 review — 2026-10-04

Independent reviewer: fresh-context Codex subagent `/root/review_cli`, read-only, given the
artifacts and working diff including untracked CLI modules. Three P2 findings: source-command
patch dispatch, moved-command package helper lookups, and report/progress console lookups.
All fixed with red-first regressions using `python3 scripts/sdlc_check.py --red`; final
independent re-review found no actionable findings and passed all 23 dispatch tests.

Author verification: `python3 scripts/sdlc_check.py` exit 0 (460 passed, 4 skipped,
4 subtests passed; lint, format, changelog PASS). `.venv/bin/python -m build` exit 0,
wheel and sdist built. Initial build dependency installation failed under sandbox DNS
restrictions; the final build used approved network escalation. Operator shipping decision
remains pending; no commit or push performed. Previously reviewed reasoning changes preserved.


## Item 12.7 review — 2026-10-04

Fresh-context read-only reviewer `/root/review_analysis`: no actionable findings.
Independently verified 7 documentation tests, 62 relative Markdown link targets,
and MkDocs YAML/archive navigation. Historical scope/provenance notices reviewed.
Original files were Git-ignored and moved locally; no independent byte comparison
with the original bodies was possible. Author used rename and inserted notices.

Author gate: `python3 scripts/sdlc_check.py` exit 0 (460 passed, 4 skipped,
4 subtests passed; lint, format, changelog PASS). `git diff --check` clean.
MkDocs build remains unverified: MkDocs is not installed in `.venv`.
No fresh runtime benchmarks, commit, push, or site publication performed.
