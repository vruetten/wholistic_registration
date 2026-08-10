# Full-package review — instance plan

**Started:** 2026-08-10 · **Owner:** Virginia + Claude
**Process:** `/full-package-review` skill (`~/.claude/skills/full-package-review/`, user-level)
— passes, evidence protocol, ledger format, and subagent briefs all live
there. This file holds only what is specific to *this* run.

## Scope decisions (2026-08-10)

1. **`journal/`, `plan/`, `audit/` committed to git.**
2. **Execution: parallel fan-out + adversarial verification.**

## Prior audits (settled ground — append there, not here)

`AUDIT.md` + `REMEDIATION_PLAN.md` (2026-05-27) cover packaging, repo
hygiene, layout, tooling, CI, docs. This review covers correctness, math,
performance, architecture. Pass 4 merges into `REMEDIATION_PLAN.md` Phase 4
rather than forking it.

## Orient results (2026-08-10)

~22k LOC live code. Churn × size ranking — top targets:

| File | LOC | Last touched |
|---|---|---|
| `utils/motion_correlation_pattern.py` | 7,941 | 2026-08-03 (cyf, active) |
| `utils/calFlowCrossResolution.py` | 2,610 | 2026-08-03 (active) |
| `core/main_function.py` | 1,675 | 2026-05 |
| `utils/IO.py` | 1,020 | 2026-05-28 |
| `utils/calFlow3d_Wei_v1.py` | 990 | 2026-05-28 |

Runnability: reviewing on macOS; cupy/GPU paths cannot execute here →
`[gpu-unverified]`, confirm later on the Janelia server. Local import of the
package still untested (todo).

## Pass status

| Pass | Ledger | Status |
|---|---|---|
| 0 — liveness | `audit/pass-0-inventory.md` | **done 2026-08-10** |
| 1 — bugs | `audit/pass-1-bugs.md` | **done 2026-08-10** — 114 findings, 96 confirmed (13 🟥) / 18 refuted; open debt: regression tests, GPU re-verification on Janelia |
| 2 — math | `audit/pass-2-math.md` | **done 2026-08-10** — core numerics sound; 8 findings (1 🟨 live: reference off-by-one M-C1; getMask robustness M-C4); GPU re-verification complete |
| 3 — performance | `audit/pass-3-performance.md` | not started |
| 4 — architecture | `audit/pass-4-architecture.md` | not started |

## Pass 1 batch order

| Batch | Files | LOC |
|---|---|---|
| 1 | `utils/motion_correlation_pattern.py` (chunked ~800-line, + cross-chunk reviewer) | 7,941 |
| 2 | `utils/calFlowCrossResolution.py` | 2,610 |
| 3 | `core/main_function.py`, `utils/registration.py`, `utils/reference.py` | 2,156 |
| 4 | `utils/calFlow3d_Wei_v1.py`, `utils/interp.py`, `utils/imresize.py`, `utils/calculate.py` | 1,488 |
| 5 | `utils/IO.py`, `utils/motion_stage_cache.py`, `utils/converters.py` | 1,668 |
| 6 | remaining live utils (`preprocess`, `mask`, `reliableAnalysis`, `visualization`, `simulation`, `ImmuneCell`, `generate_demo_data`, `__init__`) | ~2,600 |
| 8 | `pipeline.py`, `pipeline_vmsr.py` | 279 |

Batch membership may shrink after Pass 0 removes dead files.

## Run-specific notes

- Each session starts with `git fetch --all --prune`; check `origin/cyf` —
  cyf pushes large commits, and batches 1–2 are his active files. Flag
  findings there for coordination before fixing.
- Cross-check `motion_correlation_pattern.py` math (Pass 2) against
  `pipeline/Motion_Extraction_and_Recognization.md`.
