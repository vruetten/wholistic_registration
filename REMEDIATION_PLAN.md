# `wholistic_registration` — Remediation Plan

Companion to [`AUDIT.md`](./AUDIT.md). Every audit finding has a concrete plan
entry below. Items are sequenced into phases so that earlier work doesn't
block later work (e.g. fix imports before moving files, write tests before
adding CI that runs them).

**Conventions**

- **Effort:** XS = <30 min, S = ~1 h, M = a few hours, L = 1–2 days, XL = >2 days.
- **Risk:** how likely the change is to break callers / downstream users.
- **Branch:** suggested branch / PR name (use one PR per ✓ item ideally; small
  related ones can be combined).

---

## 0. Sequencing principles

These ordering rules drive the phase grouping below. Violate them at your own
risk.

1. **Untrack files before extending `.gitignore`.** A `.gitignore` rule never
   removes already-tracked files; you must `git rm --cached` first.
2. **Fix imports before moving files.** Refactoring file locations on top of
   already-broken `from utils import …` will create a churn explosion.
3. **Add the public `__init__.py` API surface after imports work.** Otherwise
   the re-exports themselves fail to import.
4. **Write at least one real test before turning on CI.** An empty CI that
   only lints is fine; CI that runs `pytest` against zero tests just causes
   noise.
5. **Land the formatter pass in its own commit, the same day pre-commit is
   added.** Future `git blame` will hate you if the formatting pass is mixed
   with logic changes.
6. **Defer breaking renames** (`IO.py` → `io.py`, `DefineParams` →
   `define_params`) until a tagged release exists so you can ship them in a
   clean `v0.2.0` with a one-release deprecation shim.
7. **Don't split `main_function.py` until tests exist.** Splitting 1.5 kLOC
   without a safety net is how regressions enter.
8. **Each phase ends with `pip install -e .` succeeding from a clean venv.**
   That's the canary that the package is still installable.

---

## Phase 1 — Make it installable & honest (≈ ½ day)

Goal: a fresh-venv `pip install -e .` succeeds, `import wholistic_registration`
works, and the working tree contains no obvious crud.

### F1 — `pyproject.toml` declares `"json"` as a dep (Critical)
- **Plan:** delete `"json"` from `dependencies`.
- **Verify:** `pip install -e .` succeeds from a clean venv.
- **Files:** `pyproject.toml`.
- **Effort:** XS · **Risk:** none · **Branch:** `fix/pyproject-json-dep`.

### F2 — Broken intra-package imports (Critical)
- **Plan:** Rewrite all `from utils import …` → `from wholistic_registration.utils import …` and `from core import …` → `from wholistic_registration.core import …`. For files inside the package, prefer relative imports (`from ..utils import IO`).
- **Files (non-exhaustive):** `core/main_function.py`, `pipeline.py`, `pipeline_vmsr.py`, `tests/test.py`, `tests/test_crossresolution_registration.py`, `demos/demo2d.py`, `demos/demo_338.py`, `archive/demo_toy.py`.
- **Also fix the typo:** `from utils import … visulization …` → `visualization` in `demos/demo_338.py`.
- **Tooling shortcut:** once ruff is added (F12), `ruff --select TID --fix` will catch most. Until then, a `sed -i 's/^from utils /from wholistic_registration.utils /'` pass over `*.py` plus manual review.
- **Verify:** `python -c "import wholistic_registration.core.main_function; import wholistic_registration.utils.IO"`.
- **Effort:** S · **Risk:** low (we already know these are broken) · **Branch:** `fix/imports-absolute`.

### F3 — Empty top-level `__init__.py` (Critical)
- **Plan:** populate `src/wholistic_registration/__init__.py` with the minimal public surface:
  ```python
  """wholistic_registration — whole-body cellular activity image registration."""
  from importlib.metadata import version, PackageNotFoundError
  try:
      __version__ = version("wholistic_registration")
  except PackageNotFoundError:  # editable install before metadata exists
      __version__ = "0.0.0+unknown"
  from .core.main_function import DefineParams, Registration_v3, ReliableAnalysis
  __all__ = ["__version__", "DefineParams", "Registration_v3", "ReliableAnalysis"]
  ```
- **Verify:** `python -c "import wholistic_registration as w; print(w.__version__, w.Registration_v3)"`.
- **Effort:** XS · **Risk:** low · **Depends on:** F2.

### F4 — Undeclared runtime dependencies (Critical)
- **Plan:** add the missing imports to `pyproject.toml`:
  ```toml
  dependencies = [
      "numpy>=1.23",
      "scipy>=1.10",
      "matplotlib>=3.7",
      "tifffile>=2023.7",
      "nd2>=0.8",
      "toml>=0.10",
      "zarr>=2.16,<3",
      "dask[array]>=2023.10",
      "h5py>=3.9",
      "scikit-image>=0.21",
  ]
  ```
- **Verification approach:** spin up a clean venv, `pip install -e .`, then run `python -c` for each imported name from `grep -rh '^import\|^from' src/wholistic_registration/{core,utils,pipeline.py,pipeline_vmsr.py}`.
- **Note on cupy:** keep `gpu` extra but also document fallback path; `utils/__init__.py` already handles `ImportError` gracefully.
- **Effort:** XS · **Risk:** none.

### F5 — Hardcoded `/home/cyf/...`, `/nrs/ahrens/...` paths (Critical)
- **Plan (this phase):** *survey only*. Don't try to plumb argparse through
  every demo today. Mark scripts as broken with a top-of-file comment, and
  open a tracking issue per script. Real fix happens in **Phase 4 (F22)**
  alongside the scripts move.
- **Effort (survey):** XS · **Risk:** none.

### F6 — Tracked `egg-info`, `.DS_Store`, large PNGs (High)
- **Plan:**
  ```bash
  git rm --cached src/.DS_Store
  git rm -r --cached src/wholistic_registration.egg-info
  git rm --cached src/wholistic_registration/pipeline/images/*.png
  ```
- **Note:** the PNGs are referenced from `pipeline/HighResolution.md`. Before
  deleting from history, either (a) leave them on disk (they're now
  gitignored, just untracked), or (b) move them to a `docs/assets/`
  directory that *is* exempt from the `**/*.png` ignore (use `!docs/assets/**/*.png`).
- **Verify:** `git status -s` shows no untracked egg-info / DS_Store.
- **Effort:** XS · **Risk:** low.

### F7 — `.gitignore` missing standards (High)
- **Plan:** replace with the GitHub Python template + project specifics
  (already drafted in AUDIT.md §3). Order matters: the `!docs/assets/…` allow-rule
  must come *after* the `**/*.png` block.
- **Verify:** `git status --ignored | head` shows `__pycache__/`, `*.egg-info/`,
  `.pytest_cache/`, `registrated_data/` all ignored.
- **Effort:** XS · **Risk:** none · **Depends on:** F6 (untrack first).

### F8 — 672 MB `registrated_data/` inside `src/` (High)
- **Plan:**
  1. `du -sh src/wholistic_registration/registrated_data/` to confirm size.
  2. Move it OUT of the repo entirely:
     `mv src/wholistic_registration/registrated_data ~/data/wbr_registrated_data`
     (or wherever your scratch lives).
  3. Confirm nothing in the package references this path (`grep -rn 'registrated_data' src/`).
- **Note:** the test data location should be in a config file or env var
  (`WHOLISTIC_TEST_DATA_DIR`).
- **Verify:** `du -sh .` drops to a few MB.
- **Effort:** XS · **Risk:** none (file is untracked; move is local).

### F10 — Duplicate `create_downsample_dataset_v4` definition (High)
- **Plan:** open `core/main_function.py`, compare line 1163 vs line 1232
  definitions, keep the later one (it's what Python actually uses), delete
  the earlier dead one. If they differ semantically, port the wanted bits
  into the surviving definition first.
- **Verify:** `grep -n 'def create_downsample_dataset_v4' src/wholistic_registration/core/main_function.py` returns one line.
- **Effort:** S · **Risk:** low (dead code by definition).

### F11 — `utils/__init__.py` prints on import (High)
- **Plan:** replace every top-level `print(...)` with module-level logger calls:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ...
  logger.info("CuPy is available with CUDA — using GPU acceleration")
  ```
  Do *not* call `logging.basicConfig` here.
- **Verify:** `python -c "import wholistic_registration.utils" 2>&1 | wc -l` prints 0.
- **Effort:** XS · **Risk:** low (the logger is silent by default; users who
  relied on the print to confirm GPU init will need to opt in to
  `logging.basicConfig(level=logging.INFO)`. Document this in CHANGELOG.).

### F14 — Stray nested `src/wholistic_registration/code/` (Medium)
- **Plan:** `rm -rf src/wholistic_registration/code` after `diff -r` against
  `src/wholistic_registration/configs/` confirms `config_0120.toml` is
  already present (it is — same filename in both locations).
- **Effort:** XS · **Risk:** none.

### ✅ Phase 1 acceptance criteria

- [ ] `pip install -e .` succeeds in a fresh venv.
- [ ] `python -c "import wholistic_registration; print(wholistic_registration.__version__)"` works.
- [ ] `git status --ignored` shows no `*.egg-info/`, `__pycache__/`, `registrated_data/`.
- [ ] `du -sh src/` < 20 MB.
- [ ] `grep -rn '^from utils \|^from core ' src/wholistic_registration` returns nothing.

---

## Phase 2 — Tooling & dev-loop (≈ 1 day)

Goal: every commit is auto-formatted/linted; CI is green; contributors have a
one-command dev setup.

### F12 — No formatter / linter / type checker / pytest config
- **Plan:** add to `pyproject.toml` (config-as-data, no extra files):

  ```toml
  [tool.ruff]
  line-length = 100
  target-version = "py310"

  [tool.ruff.lint]
  select = ["E","F","W","I","B","UP","N","RUF","SIM","PIE"]
  ignore = ["E501","N802","N803","N806"]   # tolerate existing PascalCase for now (Phase 4 fixes)

  [tool.ruff.format]
  quote-style = "double"

  [tool.black]
  line-length = 100
  target-version = ["py310"]

  [tool.mypy]
  python_version = "3.10"
  ignore_missing_imports = true
  files = ["src/wholistic_registration"]
  # Start loose; ratchet up over time.
  check_untyped_defs = false

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "-ra --strict-markers --import-mode=importlib"
  ```

- **Effort:** XS · **Risk:** none.

### F25b — `.editorconfig` and `.gitattributes`
- **Plan:** add `.editorconfig` (4-space indent for `.py`, 2-space for `.toml`/`.yaml`, LF line endings, final newline). Add `.gitattributes` (text=auto eol=lf, mark binary file types, `.ipynb -text merge=jupyter`).
- **Effort:** XS · **Risk:** none.

### Pre-commit hooks
- **Plan:** add `.pre-commit-config.yaml`:
  ```yaml
  repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.6.0
      hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
        - id: check-toml
        - id: check-merge-conflict
        - id: check-added-large-files
          args: ["--maxkb=500"]
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.6.9
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
    - repo: https://github.com/kynan/nbstripout
      rev: 0.7.1
      hooks:
        - id: nbstripout
  ```
- **Verify:** `pre-commit install && pre-commit run --all-files`.
- **One-time pain:** the first run will produce a huge formatting diff. Land that diff in its own commit (`style: ruff format pass`) so future blame is clean. Add the commit SHA to `.git-blame-ignore-revs` and configure GitHub to honor it.
- **Effort:** S · **Risk:** low.

### F18 — ~220 `print()` calls
- **Plan:** *do not* mass-replace in this phase. The risk is too high and
  the value is incremental. Instead:
  1. In Phase 2, only fix the *library import-time* prints (F11 covers this).
  2. In Phase 4, when each module is touched anyway, swap its `print()` for `logger.info()`.
  3. Add a `B021`-equivalent ruff rule? No — but add a custom check via `ruff --select T20` (`flake8-print`) and *allow-list* `T20` for now; flip the gate later.
- **Effort:** *deferred* (Phase 4).

### F19 — Bare / over-broad `except`
- **Plan:** ruff's `BLE001` and `E722` will flag them. Triage each: replace with
  `except SpecificError:` or `except Exception as exc: logger.exception(...)`.
  Five sites only — do them now while ruff is being introduced.
- **Effort:** S · **Risk:** low.

### F26 — No CI
- **Plan:** add `.github/workflows/ci.yml`:
  ```yaml
  name: CI
  on:
    push: { branches: [main] }
    pull_request:
  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.11", cache: pip }
        - run: pip install ruff black mypy
        - run: ruff check .
        - run: ruff format --check .
        - run: mypy src || true   # advisory until F23 strict-types pass
    test:
      runs-on: ubuntu-latest
      strategy:
        matrix: { python: ["3.10", "3.11", "3.12"] }
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: ${{ matrix.python }}, cache: pip }
        - run: pip install -e ".[dev]"
        - run: pytest -q
  ```
  Note: `pytest` will exit-code 5 (no tests collected) until Phase 3. Either
  add `--exitfirst` plus an allow-empty trick, or just merge the `test` job
  in the same PR as F14/F20 below.
- **Effort:** S · **Risk:** low.

### F25a — Dependabot
- **Plan:** `.github/dependabot.yml` for `pip` and `github-actions`, weekly,
  open PRs against `main`.
- **Effort:** XS · **Risk:** none.

### ✅ Phase 2 acceptance criteria

- [ ] `ruff check .` returns 0.
- [ ] `ruff format --check .` returns 0.
- [ ] `pre-commit run --all-files` passes locally.
- [ ] GitHub Actions CI runs lint + tests (tests may be empty stub) and is green.
- [ ] At least one Dependabot PR has shown up.

---

## Phase 3 — Tests & docs (≈ 2–3 days)

Goal: a real pytest suite covers the critical primitives, a CONTRIBUTING.md
tells anyone how to set up dev, README has a working quickstart.

### F20 — No real tests
- **Plan:** create a top-level `tests/` directory with structure:
  ```
  tests/
  ├── conftest.py             # synthetic data fixtures
  ├── unit/
  │   ├── test_io_roundtrip.py        # IO.readMeta / saveTiff_new on tiny tif
  │   ├── test_preprocess.py          # auto_contrast, normalize_to_255, robust_mean_std
  │   ├── test_calculate.py           # getDet2, getDet3, zncc, hann2d
  │   └── test_imresize.py            # known-output 2x2 → 4x4 imresize
  └── integration/
      └── test_registration_smoke.py  # tiny 16x16x3 synthetic volume, 3 frames
  ```
- **Synthetic data:** generate tiny in-memory volumes. Never depend on
  ND2 paths, NFS, or `/home/cyf/...`.
- **Coverage target:** start with ≥ 30 %, gate at >= current value in CI to
  prevent regression.
- **Add pytest deps to `dev` extra**: `pytest`, `pytest-cov`, `hypothesis`
  (optional).
- **Effort:** L · **Risk:** medium (writing tests for legacy code always
  surfaces bugs — that's the point).

### Move tracked `tests/` out of the package
- **Plan:** `git mv src/wholistic_registration/tests/* tests/legacy/` then
  decide per-file: keep as integration test (port to pytest), move to
  `examples/notebooks/`, or delete (the benchmark `test_vectorization.py`
  belongs in `bench/` or just `examples/`).
- **Effort:** M · **Risk:** low.

### F13 — README has no install/usage
- **Plan:** rewrite README structure:
  ```
  # wholistic_registration
  [badges]
  ## Overview (keep current paragraph + videos)
  ## Installation
    pip install -e ".[gpu]"             # users
    pip install -e ".[dev]"             # contributors
  ## Quickstart (10–15 lines of real code, runs against a synthetic fixture)
  ## Documentation  → docs.* link / docs/ dir
  ## Citing  → CITATION.cff one-liner
  ## License  → "BSD-3-Clause, see LICENSE" (delete the embedded license body)
  ## Acknowledgments (keep)
  ```
- **Effort:** S · **Risk:** none.

### F22 — `pipeline/pipeline.md` is a TODO journal
- **Plan:**
  1. Triage each bullet: completed → CHANGELOG.md entry, open → GitHub issue, dead → delete.
  2. Delete `pipeline/pipeline.md`.
- **Effort:** S · **Risk:** none — but be sure to capture the open TODOs in
  issues first (the file has institutional knowledge).

### Move `pipeline/HighResolution.md` to `docs/`
- **Plan:** `git mv src/wholistic_registration/pipeline/HighResolution.md docs/algorithm.md`. Set up minimal MkDocs (or just leave as Markdown for now and add MkDocs later).
- **Effort:** S · **Risk:** none.

### F20 (docs side) — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `CITATION.cff`
- **Plan:** stubs from standard templates:
  - `CONTRIBUTING.md`: dev setup (`pip install -e ".[dev]" && pre-commit install`), how to run tests, code style policy, PR checklist.
  - `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1.
  - `SECURITY.md`: "Email <addr> for vulnerability reports; please do not open public issues."
  - `CHANGELOG.md`: Keep-a-Changelog format; `## [Unreleased]` section.
  - `CITATION.cff`: render "Cite this repository" button.
- **Effort:** S · **Risk:** none.

### F21 — `pyproject.toml` missing standard metadata
- **Plan:** add `authors`, `readme = "README.md"`, `urls`, `classifiers`.
  Also adopt `setuptools-scm` for dynamic version (see F24, but can be
  folded in here):
  ```toml
  [build-system]
  requires = ["setuptools>=61", "setuptools-scm>=8"]

  [project]
  dynamic = ["version"]
  readme = "README.md"
  authors = [{ name="Virginia Ruetten", email="ruettenv@hhmi.org" }, …]
  classifiers = [
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
  ]

  [project.urls]
  Homepage = "https://github.com/vruetten/wholistic_registration"
  Issues   = "https://github.com/vruetten/wholistic_registration/issues"

  [tool.setuptools_scm]
  version_file = "src/wholistic_registration/_version.py"
  ```
  Then update `__init__.py` to `from ._version import __version__` and remove
  the `importlib.metadata` fallback (or keep it as belt-and-suspenders).
- **Effort:** S · **Risk:** low — first tag will set the version.

### ✅ Phase 3 acceptance criteria

- [ ] `pytest` runs and ≥ 5 tests pass (unit + 1 integration).
- [ ] Coverage report uploaded to CI artifact (or Codecov).
- [ ] README has a runnable quickstart against the synthetic fixture.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `CITATION.cff` exist.
- [ ] `pyproject.toml` has full metadata + dynamic versioning.

---

## Phase 4 — Structural refactor (≈ 1–2 weeks, can ship as v0.2.0)

Goal: codebase reflects modern Python conventions; large modules split;
hardcoded paths gone; scripts live where scripts live. This phase contains
breaking changes — ship them in a single tagged release with a deprecation
shim for at least one minor version.

### F17 — Split `core/main_function.py` (1586 LOC)
- **Plan:** create the structure:
  ```
  core/
  ├── __init__.py            # re-exports for back-compat
  ├── config.py              # DefineParams (becomes load_config + dataclass)
  ├── reference.py           # mid-window reference building
  ├── registration.py        # Registration_v3 orchestrator
  ├── reliable_analysis.py   # ReliableAnalysis
  └── downsample.py          # create_downsample_dataset_v3/v4
  ```
  Keep the old names re-exported from `core/main_function.py` with a
  `DeprecationWarning` so external scripts don't immediately break:
  ```python
  # core/main_function.py
  import warnings
  from .registration import Registration_v3 as _Registration_v3
  def Registration_v3(*a, **kw):
      warnings.warn(
          "core.main_function.Registration_v3 is deprecated; "
          "use core.registration.Registration_v3.",
          DeprecationWarning, stacklevel=2)
      return _Registration_v3(*a, **kw)
  ```
- **Verify:** integration test from Phase 3 still passes against the new layout.
- **Effort:** L · **Risk:** medium — must have tests first (Phase 3 gate).

### F16 — PEP 8 module / function renames
- **Plan:** create new lowercase names, keep PascalCase aliases as deprecated
  shims for one release:
  - `utils/IO.py` → `utils/io.py` (Python stdlib name shadowing is fine — the
    package context disambiguates)
  - `utils/ImmuneCell.py` → `utils/immune_cell.py`
  - `DefineParams` → `define_params`
  - `Registration_v3` → `register` (or `run_registration`) — `v3` doesn't
    belong in a public API.
  - `ReliableAnalysis` → `reliable_analysis`
- **Implementation:** add the new name; old name calls the new and emits a
  `DeprecationWarning`. Document in CHANGELOG.
- **Verify:** add a test that imports both names and asserts the alias works.
- **Effort:** M · **Risk:** medium (external scripts will warn; document well).

### F18 — Replace `print()` with `logging`
- **Plan:** module-by-module while you're already touching files for F17/F16.
  Pattern:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  # logger.info(...) / logger.debug(...) / logger.warning(...) / logger.exception(...)
  ```
  In CLI entry points (Phase 4 / F22), configure handlers once:
  `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")`.
- **Effort:** M (split across F17 PRs).

### F22 (real fix) — Scripts and CLI
- **Plan:**
  1. Move out of package: `git mv src/wholistic_registration/pipeline.py scripts/run_pipeline.py` and same for `pipeline_vmsr.py`.
  2. Replace hardcoded paths with `argparse`/`click`:
     ```python
     # scripts/run_pipeline.py
     import argparse, logging
     from wholistic_registration.core.registration import register
     def main():
         p = argparse.ArgumentParser()
         p.add_argument("--config", required=True)
         p.add_argument("--gpu", type=int, default=0)
         p.add_argument("--no-parallel", action="store_true")
         args = p.parse_args()
         logging.basicConfig(level=logging.INFO)
         register(args.config, parallel=not args.no_parallel, gpu_id=args.gpu)
     if __name__ == "__main__":
         main()
     ```
  3. Optional: register as a console entry point:
     ```toml
     [project.scripts]
     wholistic-register = "wholistic_registration.cli:main"
     ```
     (Move `main` into `wholistic_registration/cli.py` for that.)
  4. Same treatment for every `demos/*.py` and `tests/test*.py` that has
     hardcoded paths (F5 real fix).
- **Effort:** L · **Risk:** medium (users currently launching `python pipeline.py` will need to relearn).

### F15 — Sibling dumping-ground directories
- **Plan:**
  ```bash
  mkdir -p tools/imagej tools/matlab examples/notebooks examples/scripts
  git mv src/wholistic_registration/macros/*.ijm     tools/imagej/
  git mv src/wholistic_registration/simulations/*.m  tools/matlab/
  git mv src/wholistic_registration/demos/*.ipynb    examples/notebooks/
  git mv src/wholistic_registration/demos/*.py       examples/scripts/
  git mv src/wholistic_registration/archive          archive/   # at repo root, mark "not maintained"
  ```
  Delete `src/wholistic_registration/results/` (move its useful PNG to `docs/assets/`).
- **Effort:** S · **Risk:** low — but every internal reference (notebooks,
  README links) needs updating.

### F23 — Duplicate IO functions
- **Plan:**
  1. `utils/io.py` audit: pick the canonical implementation for each
     duplicate pair (`readMeta` vs `readMeta_new`, `saveTiff` vs
     `saveTiff_new`).
  2. Add `DeprecationWarning` to the loser, point to the winner.
  3. After one release cycle, delete the loser.
- **Effort:** M · **Risk:** medium (need tests to be confident).

### F24 — `.m` and `.ijm` files
- Already handled by F15 (move to `tools/`).

### F25 — Missing `__init__.py`
- **Plan:** after Phase 4 moves, only directories that are actual Python
  subpackages need `__init__.py`. The MATLAB/ImageJ/notebook directories
  must *not* have one (they're not Python). After F15, the remaining
  subpackages that need `__init__.py` are: `core/`, `utils/`, `cli/` (if
  created). Confirm each is present.
- **Effort:** XS · **Risk:** none.

### Type hints (start)
- **Plan:** annotate the public API only (`core/__init__.py`'s re-exports
  and their signatures). Flip `mypy` from advisory to gating in CI.
- **Effort:** M · **Risk:** low.

### ✅ Phase 4 acceptance criteria

- [ ] No file inside `src/` is > 600 LOC.
- [ ] No hardcoded user paths anywhere in `src/`.
- [ ] `wholistic-register --config path/to/config.toml` works (console script).
- [ ] All public-API functions have type hints; mypy passes in gating mode.
- [ ] `grep -rn 'def DefineParams\|def Registration_v3' src/` finds only the
      deprecated shims, plus the new lowercase implementations.

---

## Phase 5 — Release engineering (≈ ½ day)

Goal: tag, publish, and don't have to think about versions manually again.

### F21b / F24 — Dynamic versioning + first release
- **Plan:**
  1. Confirm `setuptools-scm` writes `_version.py` (from Phase 3 / F21).
  2. Tag: `git tag -a v0.1.0 -m "First releasable version"`.
  3. Push tag: `git push origin v0.1.0`.

### Release workflow
- **Plan:** add `.github/workflows/release.yml`:
  ```yaml
  name: Release
  on:
    push: { tags: ["v*"] }
  jobs:
    pypi:
      runs-on: ubuntu-latest
      permissions: { id-token: write, contents: read }
      steps:
        - uses: actions/checkout@v4
          with: { fetch-depth: 0 }   # setuptools-scm needs tags
        - uses: actions/setup-python@v5
          with: { python-version: "3.12" }
        - run: pip install build
        - run: python -m build
        - uses: pypa/gh-action-pypi-publish@release/v1
  ```
- **Prereq:** register the project on (Test)PyPI as a trusted publisher tied
  to this repo + workflow filename.
- **Effort:** S · **Risk:** low — test against TestPyPI first.

### Branch & PR strategy throughout
- **Plan:** small, focused PRs against `main`. Use these branch prefixes for
  organization (already used above):
  - `fix/*` — bug fixes
  - `refactor/*` — non-behaviour changes
  - `chore/*` — repo hygiene
  - `docs/*` — docs only
  - `feat/*` — additions
  - `style/*` — formatting only (the big ruff-format pass)
- Add `.git-blame-ignore-revs` and commit the style-pass SHA there.
- Enable branch protection on `main`: require CI green, require ≥ 1 review.

### ✅ Phase 5 acceptance criteria

- [ ] `v0.1.0` tag exists on `origin`.
- [ ] `pip install wholistic_registration` from TestPyPI works in a clean venv.
- [ ] Release workflow has run successfully end-to-end at least once.

---

## Risks & rollback strategy

| Risk | Mitigation |
|---|---|
| Phase 1 import rewrite breaks an unknown user's script | Land as one focused PR; tag a `pre-cleanup` git tag before merging so any user can `pip install git+https://…@pre-cleanup`. |
| Phase 2 mass formatting churn destroys `git blame` | Land formatting in its own commit; add SHA to `.git-blame-ignore-revs`; configure GitHub blame to honor it. |
| Phase 3 tests reveal real bugs in v1 | That's the point — file as issues, don't block the test PR on fixing them. Use `pytest.mark.xfail` for known-broken behaviour and ratchet up. |
| Phase 4 renames break downstream notebooks | One-release deprecation shim; document loudly in CHANGELOG and README; bump only minor (0.x.y → 0.x+1.0). |
| Phase 4 splitting `main_function.py` introduces regressions | Gated by Phase 3 tests; can roll back by `git revert` of the split PR. |
| Release workflow misconfigured, ships broken wheel | First release goes to TestPyPI only; only after a smoke install do you trigger a real PyPI release. |

---

## Effort summary

| Phase | Effort | Calendar (1 dev, half-time) |
|---|---|---|
| 1 — Installable & honest | ~½ day | 1 day |
| 2 — Tooling & dev-loop | ~1 day | 2 days |
| 3 — Tests & docs | ~2–3 days | 1 week |
| 4 — Structural refactor | ~1–2 weeks | 3–4 weeks |
| 5 — Release engineering | ~½ day | 1 day |
| **Total** | **~3 weeks of focused work** | **5–6 weeks part-time** |

---

## Recommended commit message convention

Adopt **Conventional Commits** (`fix:`, `feat:`, `chore:`, `refactor:`,
`docs:`, `test:`, `style:`, `ci:`). This pairs nicely with `release-please`
or `commitizen` for auto-generated CHANGELOGs later, if you want to skip
maintaining `CHANGELOG.md` by hand.

---

## What I'd do *first* if I had only 2 hours

1. F1: drop `"json"`.
2. F4: add the missing deps to `pyproject.toml`.
3. F2: bulk rewrite of `from utils …` / `from core …` imports.
4. F3: populate the public `__init__.py`.
5. F6 + F7: untrack egg-info / DS_Store / PNGs, replace `.gitignore`.
6. F8: move 672 MB `registrated_data/` out of `src/`.
7. F10: delete duplicate `create_downsample_dataset_v4`.
8. F11: replace import-time prints with logger calls.
9. F14: delete stray `code/` directory.

That's ~90 % of the *external-user-blocking* problems solved before lunch.
