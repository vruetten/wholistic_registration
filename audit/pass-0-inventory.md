# Pass 0 — inventory & liveness

**Date:** 2026-08-10 · **Commit:** `b975648` · **Method:** AST import graph
(script: session scratchpad `pass0_graph.py`) over all 65 package `.py` files,
14 notebooks, and top-level `tests/`; resolves proper absolute, relative, and
the repo's broken bare (`from utils import …`) import styles. Roots: pipeline
scripts, v2 runner, demo/test scripts, notebooks, pytest trees.

## Classification

### Live — v1 registration chain (reachable from `pipeline.py` / `pipeline_vmsr.py`)
`core/main_function.py` · `core/__init__.py` · `utils/__init__.py` ·
`utils/IO.py` · `utils/reference.py` · `utils/registration.py` ·
`utils/reliableAnalysis.py` · `utils/calFlow3d_Wei_v1.py` ·
`utils/calculate.py` · `utils/imresize.py` · `utils/interp.py` ·
`utils/mask.py` · `utils/preprocess.py` · plus entry scripts `pipeline.py`,
`pipeline_vmsr.py`.

### Live — v2 chain (reachable from `v2/pipeline/runner.py`)
All of `v2/core/`, `v2/config/`, `v2/io/`, `v2/utils/` (incl.
`array_ops.py`, `logging.py`, `validation.py`), `v2/pipeline/`,
`v2/__init__.py`. `v2/tests/*` are pytest roots; `v2/examples/*` are script
roots — both in scope as test/demo roots.

### Live-standalone (zero in-repo callers; adjudicated manually)
- `utils/motion_correlation_pattern.py` (7,941 LOC) — no in-repo importer, no
  `__main__` guard; referenced only by the methods write-up
  (`pipeline/Motion_Extraction_and_Recognization.md`). Actively developed
  (2026-08-03, cyf) — driven from cyf's environment. **In scope, batch 1.**
- `utils/motion_stage_cache.py` (576 LOC) — same situation; notably *not*
  imported by `motion_correlation_pattern.py` either.
- `utils/calFlowCrossResolution.py` (2,610 LOC) — the HR cross-resolution
  algorithm; reached only from `src/…/tests/` scripts and notebooks
  (`test_F260517*`, `run_F260517_0625.py`), not from `pipeline.py`. **In
  scope, batch 2.** The v1 pipeline's core algorithm living outside the
  pipeline's reach is a Pass 4 architecture observation (noted for A-###).

### Demo-only (reached solely from demos/notebooks/test scripts — in scope, low priority)
`utils/visualization.py` · `utils/simulation.py` · `utils/converters.py` ·
`utils/generate_demo_data.py` · top-level `__init__.py` (only demos and the
smoke test import the package root).

### Demo/test script roots (liveness-checked only; line-level review deferred)
13 files in `demos/` and `src/…/tests/` — hardcoded-path scripts and ports,
already characterized by `AUDIT.md` (#5, #6) and excluded from tooling by
repo convention. Deferred by dated scope note below.

### Legacy
`archive/demo_toy.py` — imports a nonexistent top-level `registration`
module; cannot run. Out of scope (existing `AUDIT.md` F2 territory).

### Dead
- **D-001 🟨 `utils/ImmuneCell.py` (257 LOC) — dead code.**
  - **Commit:** b975648
  - **Claim:** nothing in the repo imports or executes this module.
  - **Evidence:** zero importers in the AST graph; `grep -rn "ImmuneCell"`
    over `*.py`/`*.ipynb`/`*.md` finds no reference outside the file itself;
    no `__main__` guard.
  - **Status:** CONFIRMED-read. Adversarial check: could be consumed from
    outside the repo like the motion files — but unlike them it is untouched
    since the 2026-05-28 format pass and referenced by no doc. Removal still
    needs cyf/Virginia sign-off.
  - **Recommendation:** move to `archive/` or delete.

## Scope notes (dated)

- 2026-08-10: `demos/` and `src/wholistic_registration/tests/` script roots
  excluded from Pass 1–3 line review (CLAUDE.md tooling-excluded dirs;
  brokenness already documented in `AUDIT.md`). Revisit in Pass 4 only as
  evidence of API usage patterns.
- 2026-08-10: `ImmuneCell.py` (D-001) and `archive/` leave the scope of
  passes 1–4.

## Looks wrong but is fine

- `v2/config/settings.py` shows as reachable only via `v2/config/__init__` —
  fine: it is the module's implementation file, re-exported by the package.
- `pipeline.py` imported by nothing — fine: it is a root (script entry), and
  `v2/__init__` importing `pipeline` was a resolver artifact, fixed before
  this ledger was written.
- 13 `v2/tests`+`examples` files with zero importers — fine: pytest/script
  roots, reached by convention not import.

## Coverage

65/65 package `.py` files classified · 14/14 notebooks parsed as roots ·
top-level `tests/` (6 files) parsed as roots. Import-graph caveat: dynamic
imports (`importlib`, `%run` magics) would be invisible; grep sweep for
`importlib` in the package found none.
