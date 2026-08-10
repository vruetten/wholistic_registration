---
name: full-package-review
description: Multi-pass evidence-based review of an entire package — liveness → bugs → math → performance → architecture — with fan-out reviewers, adversarial verification, and a persistent audit trail.
disable-model-invocation: true
---

# Full-package review

Reviews a whole package (not a diff) in ordered passes, coarse → fine. Every
finding is evidence-backed and adversarially verified before it counts; every
confirmed-by-execution bug becomes a failing regression test. All state lives
in three committed folders, so the review survives sessions and machines:

- `plan/` — instance plan: scope decisions (dated), batch table, pass status.
- `audit/` — one findings **ledger** per pass.
- `journal/` — one page per working day: done / found / todos / handoff.

## Process

### 1. Orient
- Read the manifest (`pyproject.toml` etc.), map the tree, rank source files
  by **churn × size** (`git log -1 --format=%as` per file, `wc -l`). Big,
  recently-touched files are the highest-yield targets and get reviewed first.
- Read existing audits/plans in the repo. Their axes are settled ground —
  append new findings of those kinds there; this review covers what they left.
- Note what linters/CI already enforce; those finding classes are theirs.
- Test **runnability**: can the package import and execute in this
  environment? Code that can only run elsewhere (GPU, cluster data) caps at
  CONFIRMED-read and carries an explicit flag like `[gpu-unverified]`.
- Done when: churn×size table and runnability statement are in `plan/`.

### 2. Scope contract
Settle with the user and record dated in `plan/`: directories in/out of
scope; which passes run; execution mode (parallel fan-out vs sequential
deep-read). Changing a decision later means a new dated entry, beneath the old.

### 3. Pass 0 — liveness
Build the reachability graph from the real entry points (scripts, configs,
notebooks, CI, `__init__` exports). Classify every source file **live /
demo-only / legacy / dead**, each with its evidence (who imports it, or
nothing does). Ledger: `audit/pass-0-inventory.md`.
Done when: every file in the tree is classified. Dead and legacy files leave
the scope of all later passes.

### 4. Passes 1–4 — bugs, math, performance, architecture
Run in order — bugs first; a pass starts only when the previous one's ledger
is closed. Per-pass checklists: [reference/checklists.md](reference/checklists.md).
For each pass:

1. Batch live files by churn × size × centrality. Files over ~1,500 lines are
   chunked along top-level function/class boundaries, one reviewer per chunk
   plus one cross-chunk reviewer for interactions.
2. **Fan out** one reviewer subagent per batch/chunk, briefed from
   [reference/reviewer-prompt.md](reference/reviewer-prompt.md). A reviewer
   reads files whole, reports only on code it read, returns at most 12
   findings (few well-evidenced findings beat a padded list) plus a mandatory
   **"looks wrong but is fine"** list — surfacing the calls it considered and
   rejected is what catches shallow analysis.
3. Pool and dedupe; every pooled finding enters the ledger as PLAUSIBLE.
4. **Verify adversarially**: an independent verifier per finding tries to
   *refute* it, briefed from [reference/verifier-prompt.md](reference/verifier-prompt.md).
   Survivors get a minimal repro executed wherever runnable.
5. Close the ledger: final statuses recorded; each CONFIRMED-run finding gets
   a failing regression test under `tests/`.

Done when: every scoped file reviewed, every finding CONFIRMED or REFUTED —
none left PLAUSIBLE — and the ledger's "looks wrong but is fine" section is
non-empty.

### 5. Session wrap-up
Journal page updated (reviewed / found / next / blockers), ledgers and tests
committed. Review commits carry findings and tests only; fixes land in
separate commits/PRs after the pass closes. On re-runs, fixed findings are
marked RESOLVED in place, never deleted.

## Evidence protocol

Every finding carries:

- **ID** — `D-###` dead code · `B-###` bug · `M-###` math · `P-###` perf ·
  `A-###` architecture. Sequential, never reused.
- **Location** — `file:line` at a stated commit SHA.
- **Claim** — one falsifiable sentence.
- **Failure scenario** — concrete inputs/state → wrong output/crash.
- **Severity** — 🟥 wrong results or crash on the main path · 🟧 wrong results
  on a plausible path · 🟨 latent / edge case · 🟦 cosmetic-but-real.
- **Status** —
  - `CONFIRMED-run`: reproduced by an executed minimal snippet (gold standard).
  - `CONFIRMED-read`: full surrounding context read; the written adversarial
    "why might this be fine?" check failed to clear it.
  - `PLAUSIBLE`: reported, unverified. Never actionable, never final.
  - `REFUTED`: kept in the ledger with its refutation, so it isn't re-found.
  - `RESOLVED`: confirmed previously, fixed since; points at the fixing commit.

A finding without `file:line` and a failure scenario stays out of the ledger —
unfalsifiable findings don't get fixed.

## Ledger format

```markdown
### B-014 🟧 `utils/interp.py:88` — claim in one sentence
- **Commit:** <sha reviewed>
- **Failure scenario:** concrete inputs/state → wrong output/crash
- **Status:** CONFIRMED-run
- **Evidence:** repro snippet / adversarial-check notes / refutation
- **Test:** tests/regressions/test_b014_edge_interp.py
```

Each ledger ends with two sections: **Looks wrong but is fine** (with the
reason each is fine) and **Coverage** (files reviewed vs scoped, so gaps are
visible rather than silent).
