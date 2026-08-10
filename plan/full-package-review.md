# Full-package code review — master plan

**Started:** 2026-08-10 · **Owner:** Virginia + Claude
**Goal:** systematic, evidence-based review of the entire package, coarse → fine:
dead files → bugs → math errors → performance → architecture. **Bugs first.**

## Relationship to existing audits

`AUDIT.md` (2026-05-27) + `REMEDIATION_PLAN.md` already cover packaging, repo
hygiene, layout, tooling, CI, and docs. **This review does not repeat those
axes.** New findings that belong to those axes get appended to `AUDIT.md`;
everything else lives in `audit/` here. The May audit excluded `v2/`; this
review **includes v2** (decision 2026-08-10).

## Decisions (2026-08-10)

1. **v2/ fully in scope** — all passes.
2. **`journal/`, `plan/`, `audit/` are committed to git** — the audit trail is
   a deliverable and must survive the Mac ↔ Janelia-server split.
3. **Execution: parallel subagent fan-out + independent adversarial
   verification** before any finding is accepted.

## Evidence protocol (applies to every pass)

Every finding carries:

- **ID** — `B-###` (bug), `M-###` (math), `P-###` (perf), `A-###`
  (architecture), `D-###` (dead code). Sequential, never reused.
- **Location** — `file:line` on a stated commit SHA.
- **Claim** — one sentence, falsifiable.
- **Failure scenario** — concrete inputs/state → wrong output/crash.
- **Verification status:**
  - `CONFIRMED-run` — reproduced by a minimal executed snippet (gold standard).
  - `CONFIRMED-read` — full surrounding context read; an adversarial
    "why might this actually be fine?" check was performed and written down.
  - `PLAUSIBLE` — reported but not yet verified. Not actionable.
  - `REFUTED` — kept in the ledger with the refutation, so it isn't re-found.
- **Severity** — 🟥 wrong results / crash on main path · 🟧 wrong results on
  plausible path · 🟨 latent / edge case · 🟦 cosmetic-but-real.

Rules:

- A subagent's finding is **always** `PLAUSIBLE` until an *independent*
  verification agent (or Virginia/Claude directly) tries to refute it.
- Every `CONFIRMED-run` finding becomes a failing pytest test in top-level
  `tests/` — the bug hunt seeds the missing test suite.
- **Environment limit:** we review on macOS; GPU (cupy) paths cannot execute
  here. GPU-only findings cap at `CONFIRMED-read` and are flagged `[gpu-unverified]`
  for later confirmation on the Janelia server.

## Passes

### Pass 0 — Inventory & liveness (dead-file level) — NEXT UP
Build the import/reachability graph from real entry points (`pipeline.py`,
`pipeline_vmsr.py`, `v2/pipeline/runner.py`, `demos/`, tracked notebooks,
`configs/*.toml` references). Classify all 65 `.py` files + notebooks as
**live / demo-only / legacy / dead**. Cross-check with git-log recency.
- Output: `audit/pass-0-inventory.md` — the scope table every later pass uses.
- Exit: every file classified, with evidence (who imports it / nothing does).

### Pass 1 — Bugs & errors (the priority)
Module-by-module over **live** files, priority = recency × size × centrality:

| Batch | Files | LOC | Notes |
|---|---|---|---|
| 1 | `utils/motion_correlation_pattern.py` | 7,941 | active (2026-08-03); chunk by top-level function/class groups, ~800-line chunks, one subagent each, cross-chunk pass at the end |
| 2 | `utils/calFlowCrossResolution.py` | 2,610 | active (2026-08-03); math-heavy |
| 3 | `core/main_function.py` + `utils/registration.py` + `utils/reference.py` | 2,156 | orchestrator |
| 4 | `utils/calFlow3d_Wei_v1.py` + `utils/interp.py` + `utils/imresize.py` + `utils/calculate.py` | 1,488 | flow + numerics primitives |
| 5 | `utils/IO.py` + `utils/motion_stage_cache.py` + `utils/converters.py` | 1,668 | I/O + caching |
| 6 | remaining live utils (`preprocess`, `mask`, `reliableAnalysis`, `visualization`, `simulation`, `ImmuneCell`, `generate_demo_data`, `__init__`) | ~2,600 | small files, wide fan-out |
| 7 | `v2/` (27 files) | 4,539 | typed/newer style; review against its own tests |
| 8 | `pipeline.py`, `pipeline_vmsr.py` | 279 | entry points |

**Shared bug-class checklist** (every review subagent gets this verbatim):
axis-order/indexing errors (zyx vs xyz, transposes) · off-by-one in
pyramid/patch/window loops · dtype overflow & silent truncation (uint16 math,
float32 accumulation) · aliasing & unintended in-place mutation · shadowed or
wrong-variable typos (copy-paste rows: x where y intended) · cupy↔numpy shim
divergence (`from wholistic_registration.utils import cp`) · swallowed
exceptions hiding failures · broken/stale imports · boundary handling
(padding, edges, NaN) · unit/scale mismatches (pixel vs micron, downsample
factors) · mutable default args · resource leaks (unclosed files/handles).

- Mechanics per batch: N review subagents in parallel → findings pooled →
  independent verification agents attempt to **refute** each finding →
  survivors get repro snippets run where executable → ledger + tests.
- Output: `audit/pass-1-bugs.md` (+ tests in `tests/regressions/`).
- Exit: all live files reviewed; every finding CONFIRMED or REFUTED, none
  left PLAUSIBLE.

### Pass 2 — Math & numerics
Deep verification of the algorithmic core against analytic ground truth:
identity flow ⇒ identity warp; `imresize` vs `scipy.ndimage.zoom` on known
arrays; flow composition across pyramid levels; interpolation at boundaries;
normalization/ZNCC formulas vs textbook definitions; motion-mode decomposition
(SVD/eigen conventions, sign/ordering) in `motion_correlation_pattern.py`
cross-checked against `pipeline/Motion_Extraction_and_Recognization.md`.
- Output: `audit/pass-2-math.md` + property-style tests.

### Pass 3 — Performance
GPU↔host transfer churn, redundant array copies, unvectorized hot loops,
memory footprint of full-volume ops, dask/zarr chunking sanity. Profile-first
where runnable on CPU; otherwise read-level with `[gpu-unverified]`.
- Output: `audit/pass-3-performance.md`.

### Pass 4 — Architecture
Merges into `REMEDIATION_PLAN.md` Phase 4 rather than forking it. Uses the
Fowler smell baseline from `mattpocock-skills:code-review` (duplicated code,
divergent change, shotgun surgery, speculative generality, …) applied
package-wide, plus: module boundaries for the v1→v2 convergence question.
- Output: `audit/pass-4-architecture.md` + updates to `REMEDIATION_PLAN.md`.

## Working rhythm

- Each session: read `journal/` latest page → `git fetch --all --prune` and
  check `origin/main` / `origin/cyf` (cyf pushes big commits) → pick up todos.
- Each session ends: journal page updated (done / found / next), findings
  committed.
- Findings on cyf's active files (`motion_correlation_pattern.py`,
  `calFlowCrossResolution.py`): flag for coordination before fixing — his
  branch may have moved.
- Fixes are **not** applied during a review pass — review and remediation stay
  separate commits/PRs, per the existing plan's sequencing rules.
