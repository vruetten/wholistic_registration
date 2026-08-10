# CLAUDE.md — Working conventions for this repo

How Virginia and Claude collaborate on **WHOLISTIC Registration**. Auto-loaded as
context every session. A new agent should read this + `AUDIT.md` +
`REMEDIATION_PLAN.md` and be caught up.

> **Python env:** always use the conda env **`wholistic-registration`**. Activate it
> at the start of every shell session before running any Python:
> ```bash
> source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh && conda activate wholistic-registration
> ```
> See the [Environment](#environment) section for details. Don't use the base env
> or a system `python`.

---

## What this project is

WHOLISTIC Registration is a fast, non-rigid image-registration method for
whole-body cellular-activity imaging (e.g. larval zebrafish whole-body imaging).
It corrects motion from skeletal/smooth-muscle contraction so cellular activity
can be analysed cleanly. Core algorithm: patch-wise iterative modified optical
flow over an image pyramid, GPU-accelerated. A second, newer line of work
(contributed on the `cyf` branch) extracts and clusters *motion patterns* from
the registration displacement fields.

It is a **library + pipeline**, not a CLI app. There is no console script: you
drive it from `pipeline.py` / notebooks with a `.toml` config.

---

## Where things live

| Concern | Location |
|---|---|
| Repo-level audit (issues, severities) | `AUDIT.md` |
| Phased plan to fix audit findings | `REMEDIATION_PLAN.md` |
| Working conventions (this file) | `CLAUDE.md` |
| Stable repo docs | `README.md` |
| Pipeline entry point | `src/wholistic_registration/pipeline.py` |
| Registration core | `src/wholistic_registration/core/`, `utils/` |
| Motion-pattern analysis (cyf) | `utils/motion_correlation_pattern.py`, `utils/motion_stage_cache.py`, methods write-up in `pipeline/Motion_Extraction_and_Recognization.md` |
| Run configs | `src/wholistic_registration/configs/*.toml` |
| Tests run by CI/pytest | top-level `tests/` |
| Local-only outputs (gitignored) | `registrated_data/`, `results/`, `src/wholistic_registration/registrated_data/`, `HR_exp/`, `*.txt` |

> Heads-up: `src/wholistic_registration/tests/` is mostly notebooks/scripts, not
> pytest tests. Real pytest tests live in the **top-level** `tests/` (that's
> what `pyproject.toml` `testpaths` and CI point at).

---

## Environment

- **Host:** Janelia Linux server. Repo lives at
  `/groups/ahrens/home/ruttenv/python_packages/wholistic_registration`. Ignore
  any `/home/cyf/...`, `/nrs/...`, or macOS paths baked into configs/notebooks —
  those are a collaborator's machine, not ours (see "Hardcoded paths" below).
- **Python env:** conda env `wholistic-registration`. Activate with:
  ```bash
  source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh && conda activate wholistic-registration
  ```
  `conda activate` alone may not work inside Claude tool calls because the conda
  shell function isn't loaded by default — always source the profile.d hook
  first.
- **Direct python path** (fallback if activation is unreliable):
  `/groups/ahrens/home/ruttenv/miniforge3/envs/wholistic-registration/bin/python`.
- **Install the repo as a package** (one-time, after activating the env):
  ```bash
  pip install -e .            # core
  pip install -e ".[gpu]"     # adds cupy for GPU acceleration
  pip install -e ".[dev]"     # pytest, ruff, black, mypy, pre-commit, nbstripout
  ```
  This makes `import wholistic_registration` work from any cwd / IDE cell.
  `pyproject.toml` (src layout, `package-dir = {"" = "src"}`) drives the install;
  deps come from its `dependencies` list.
- **GPU:** acceleration is via `cupy`, imported through a shim:
  `from wholistic_registration.utils import cp` (falls back to numpy if cupy is
  unavailable). Pipeline code selects a device with `cp.cuda.Device(N).use()`.
- **A clean `pip install -e .` succeeding is the canary** that the package is
  still installable — it's the exit check for every remediation phase.

---

## How to run

There is no `wholistic_registration` console script. Runs are config-driven:

- Edit / pick a config in `src/wholistic_registration/configs/*.toml`.
- The current entry point is the `__main__` block of
  `src/wholistic_registration/pipeline.py`, which calls
  `main_function.DefineParams(configFile=..., inputFile=..., outputFile=...)`
  then `main_function.Registration_v3(configFile, parallel=...)`.
- That `__main__` block currently hardcodes a collaborator's paths and GPU index
  — **don't commit edits to those defaults**; override locally instead (see
  below).

---

## Tooling (lint / format / types / tests)

- **Formatter + linter:** `ruff` (lint + format) and `black`, both `line-length = 100`.
  Run via pre-commit:
  ```bash
  pre-commit run --all-files
  ```
- **Type checker:** `mypy`, advisory only for now (loose settings; ratchet up over time).
- **Excluded from lint/format/mypy** (legacy / WIP, deliberately untouched):
  `archive/`, `code/`, `demos/`, `tests/` (the src-level ones), `macros/`,
  `simulations/`. Don't burn time reformatting these.
- **`.git-blame-ignore-revs`** records the big format-pass SHA so `git blame`
  skips it. If you land another bulk-format commit, add its SHA there.
- **Tests:** `pytest` (testpaths = top-level `tests/`). CI (`.github/`) runs
  pytest on every PR.

---

## Git workflow

- **Remote:** `origin` = `https://github.com/vruetten/wholistic_registration.git`
  (Virginia's fork). No upstream is configured right now.
- **Branches:**
  - `main` — integration branch.
  - `cyf` — Yunfeng Chi's (`chiyf21`, Tsinghua) motion-analysis work; gets
    merged into `main`. **Check `origin/main` and `origin/cyf` at the start of
    each session** — cyf pushes large feature commits.
  - `vmsr` — Virginia's branch.
  - `temp-branch`, `fixup_main`, `backup_with_bad_commit` — local cleanup/backup
    branches; leave them alone unless asked.
- **Start each session:** `git fetch --all --prune`, then review what's new:
  `git log --oneline main..origin/main` and `git diff --stat main..origin/main`.
  Pull with `git pull --ff-only` when it's a clean fast-forward.
- Commits: small, focused, descriptive. Use HEREDOC for multi-line messages.
- Keep the formatting/bulk-mechanical changes in their own commits, separate
  from logic changes (see `.git-blame-ignore-revs`).
- **Never commit:** data/outputs (`registrated_data/`, `results/`, `*.npy`,
  large TIFFs), `.DS_Store`, `*.egg-info/`, notebook outputs (use `nbstripout`),
  or machine-specific paths.
- Don't push to anything other than your working branch (or a new feature branch
  off it) without confirming first. No `--no-verify`, `--force`, or amending
  pushed commits without asking.

### Hardcoded paths (a real, known hazard)
Configs, `pipeline.py`, and many notebooks contain absolute paths like
`/home/cyf/...` and `/nrs/ahrens/...`. These belong to a collaborator's machine.
When you change input/output paths for a local run, **keep that edit out of
shared commits** (or parametrize it) so it never leaks upstream — this mirrors
the audit's "hardcoded user paths" finding (`AUDIT.md` #5).

---

## How we work

### Before non-trivial changes
- Propose the approach in 2-3 sentences and confirm before implementing.
  Especially for:
  - Anything that changes a module's public API or import surface.
  - Anything that touches `pyproject.toml` (deps, packaging, tool config).
  - Anything that renames files/functions (`IO.py`, `DefineParams`, etc.) — these
    are deferred in `REMEDIATION_PLAN.md` until a tagged release.
  - Adding files outside the area you're working on.
- For lookups / small fixes / file edits: just do it.

### Follow the remediation plan
`REMEDIATION_PLAN.md` sequences fixes into phases with ordering rules (e.g.
untrack files before extending `.gitignore`; fix imports before moving files;
write a test before turning on CI). Respect that sequencing — it exists so early
work doesn't block later work. Update the relevant audit/plan entry when an item
lands.

### Verify before claiming done
- If a change is supposed to make something work, run it.
- If something isn't testable in this environment (GPU kernels, real microscopy
  data, large registration runs), say so explicitly rather than claiming success.
- Don't mark a task `completed` if any step failed.

### Don't
- Add CLI flags, error handling, or abstractions "for the future." Add them when needed.
- Add comments that re-state what the code does. Only comment the *why* when non-obvious.
- Inline-reformat the excluded legacy dirs (`archive/`, `code/`, …).
- Re-read a file Claude just edited; the harness tracks state.

### Communication
- Short responses by default. Match the question's depth.
- When proposing options, list the recommended one first with `(Recommended)`.
- Surface risks before acting on them (e.g. "this rewrites tracked files",
  "this overwrites local changes", "this pushes to a shared branch").

---

## When the session ends

1. All in-session tasks are either `completed` or written down as concrete
   handoff items (file paths, function names, exact commands).
2. If we made commits, note the SHA range so the next session can pick up.
3. If `origin` advanced and we didn't integrate it, say so explicitly.

---

## Debugging discipline

When something fails and the cause isn't immediately obvious, follow this discipline. The goal is to find the true cause, not to defend the first plausible one.

### Form hypotheses as a set, not a singleton
On the first failure, write down 2–4 candidate explanations, not one. Explicitly note which is most likely AND what the leading alternatives are. A single hypothesis is an anchor; a set keeps you honest. State your confidence in each as a rough probability.

### Every diagnostic step must be able to DISCONFIRM, not just confirm
Before running a command to investigate, ask: "What outcome would prove my leading hypothesis WRONG?" Prefer tests that can falsify over tests that merely re-observe the symptom. If a command can only ever confirm what I already believe, it's low value. The single most useful experiment is usually the one that distinguishes between two competing hypotheses in one shot — design that experiment explicitly and run it early.

### Treat surprising results as evidence, not noise to explain away
If a result contradicts my current theory (e.g. a plain `cp` fails when my theory says only one specific app should be blocked), that contradiction is a SIGNAL the theory is wrong. Do not invent a new sub-mechanism to rescue the hypothesis. Inventing a plausible-sounding rule to patch a failing theory ("X is OS-protected so only the originating app can touch it") is confabulation — flag it to myself as such and downgrade the hypothesis instead.

### Know which mechanisms are real before invoking them
Before asserting that some system behaves a certain way ("this attribute gates file access by originating process"), check: do I actually know this is how it works, or am I pattern-matching to something that sounds right? If I'm not sure, say so explicitly and verify (docs, web search, or a direct test) rather than building a chain of reasoning on an assumed mechanism. State the assumption out loud so it's auditable.

### Re-read my own evidence before concluding
Before committing to an explanation, scan back over everything observed this session. Often the disconfirming fact is already on the screen (e.g. I correctly identified the sandbox earlier, then ignored it). My own earlier output is data; don't let a later narrative overwrite an earlier correct observation.

### Separate "what changed" from "what I started doing differently"
If access/behavior seems to "suddenly" break mid-session, distinguish between (a) the environment actually changing and (b) me starting to do a different kind of operation (e.g. moving from session-created files to pre-existing ones). The second is far more common and points to a structural cause, not a transient one.

### Escape the loop after 2 failed fixes
If two attempted fixes for the same hypothesis both fail, STOP iterating on that hypothesis. Explicitly reconsider the premise. State: "My working theory is X; two fixes have failed; here are the alternative theories I deprioritized and the one cheap test that would distinguish them." Then run that test. Do not propose a third variation of the same fix.

### Prefer the test that resolves the question over the fix that assumes the answer
When blocked, the instinct is to keep trying fixes. Resist it. One well-chosen diagnostic that tells you WHICH world you're in is worth more than three speculative fixes. Cheap, decisive, falsifying — in that order.

### Don't outsource the diagnosis to the user prematurely
Before asking the user to run privileged commands or change system settings, exhaust the tests I can run myself, especially falsifying ones. If I must ask, ask for the output of a test that distinguishes hypotheses, not for a fix that presumes one.
