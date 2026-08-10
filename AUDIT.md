# `wholistic_registration` — Repository Audit

**Scope:** Whole repo.
**Date:** 2026-05-27
**Branch audited:** `main` @ `84637ab` (1 ahead of `origin/main`).
**Tracked files:** 104. **Working-tree size:** ~683 MB (672 MB of which is an untracked `registrated_data/` blob).

---

## TL;DR — Severity Summary

| # | Severity | Issue | Effort |
|---|----------|-------|--------|
| 1 | 🟥 Critical | `pyproject.toml` declares `"json"` as a dep (stdlib module — `pip install` breaks) | trivial |
| 2 | 🟥 Critical | Broken intra-package imports (`from core import …`, `from utils import …`) make the installed package unusable from outside the source tree | small |
| 3 | 🟥 Critical | Top-level `__init__.py` is **empty** — no public API surface; users can't `from wholistic_registration import …` | small |
| 4 | 🟥 Critical | Heavy runtime deps imported but undeclared: `nd2`, `h5py`, `dask`, `zarr`, `scikit-image`, `cupy` (declared only as optional) | small |
| 5 | 🟥 Critical | Hardcoded absolute user paths (`/home/cyf/...`, `/nrs/ahrens/...`) in 30+ files | medium |
| 6 | 🟥 Critical | `tests/` contains scripts & notebooks, **no real pytest tests**; no CI | medium |
| 7 | 🟧 High | Build/runtime artifacts tracked in git: `*.egg-info/`, `src/.DS_Store`, 5 large PNGs (1–1.6 MB each) | trivial |
| 8 | 🟧 High | `.gitignore` missing standard Python entries (`__pycache__/`, `*.egg-info/`, `.ipynb_checkpoints/`, venvs, `.env`) | trivial |
| 9 | 🟧 High | 672 MB `src/wholistic_registration/registrated_data/` (output data) sits inside the package directory | medium |
|10 | 🟧 High | `core/main_function.py` defines `create_downsample_dataset_v4` twice (dead first definition) | trivial |
|11 | 🟧 High | `utils/__init__.py` `print`s to stdout on import (library anti-pattern) | trivial |
|12 | 🟧 High | No CI, no pre-commit, no formatter (black/ruff), no linter, no type checker | medium |
|13 | 🟨 Medium | README has zero installation/usage instructions | small |
|14 | 🟨 Medium | Mysterious nested package copy `src/wholistic_registration/code/wholistic_registration/configs/` | trivial |
|15 | 🟨 Medium | Sibling dumping-ground dirs: `archive/`, `code/`, `demos/`, `macros/`, `simulations/`, `results/` | medium |
|16 | 🟨 Medium | Module/function names violate PEP 8 (`IO.py`, `ImmuneCell.py`, `DefineParams`, `Registration_v3`) | medium |
|17 | 🟨 Medium | Files >1 kLOC: `calFlowCrossResolution.py` (1974), `main_function.py` (1586), `calFlow3d_Wei_v1.py` (1090), `IO.py` (982) | large |
|18 | 🟨 Medium | ~220 raw `print()` calls — no `logging` module | medium |
|19 | 🟨 Medium | Bare `except:` / `except Exception:` blocks in 5 files | small |
|20 | 🟨 Medium | No `CONTRIBUTING.md`, `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `docs/` | small |
|21 | 🟨 Medium | `pyproject.toml` missing standard project metadata (`authors`, `readme`, `urls`, `classifiers`, dynamic version) | trivial |
|22 | 🟨 Medium | `pipeline/pipeline.md` is a private TODO journal mixed with real docs | small |
|23 | 🟦 Low  | Duplicate IO functions (`readMeta` / `readMeta_new`, etc.) suggest stale lineage | medium |
|24 | 🟦 Low  | MATLAB (`.m`) and ImageJ (`.ijm`) files mixed inside a Python package | small |
|25 | 🟦 Low  | No `__init__.py` in `tests/`, `demos/`, `configs/`, `archive/`, `macros/`, `simulations/`, `pipeline/`, `code/` | small |

Counts: **6 Critical**, **6 High**, **10 Medium**, **3 Low**.

---

## 1. Packaging & Dependency Management

### Findings

`pyproject.toml` (only 26 lines):

```toml
dependencies = [
    "numpy", "scipy", "matplotlib", "tifffile", "nd2", "toml", "zarr",
    "json",          # ← stdlib module; this will break pip install
]

[project.optional-dependencies]
gpu = ["cupy"]
```

- **🟥 `json` is a stdlib module.** A package by that name *does* exist on PyPI (an abandoned shim), but listing it here is wrong and at best installs noise.
- **🟥 Missing declared deps** (all imported by tracked code):
  `dask`, `h5py`, `nd2`, `zarr`, `scikit-image` (`from skimage…`), `cupy` (only optional, but pipeline crashes at import-time without it on machines without it because of `from utils import cp` chain logic that *prints* but still runs… still needs declaring as a hard runtime dep or guarded better).
- **🟥 No version pins** anywhere → unreproducible builds.
- Missing standard metadata: `authors`, `readme = "README.md"`, `urls = {…}`, `classifiers = [...]`, `keywords`, dynamic version from VCS.
- No `[tool.setuptools.packages.find]` — relies on implicit discovery; risky given the *nested* `src/wholistic_registration/code/wholistic_registration/` directory which `setuptools` could plausibly treat as a second package.
- `wholistic_registration.egg-info/` (5 files) is **tracked** in git. Egg-info is a build artifact; it must not be in version control.
- No lockfile (`uv.lock`, `requirements.lock`, `poetry.lock`).

### Recommendations

```toml
[build-system]
requires = ["setuptools>=61", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "wholistic_registration"
dynamic = ["version"]
description = "Whole-body cellular activity image registration"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "Virginia Ruetten", email = "..." }, ...]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Operating System :: POSIX :: Linux",
]
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

[project.optional-dependencies]
gpu  = ["cupy-cuda12x>=12"]    # be explicit about CUDA major version
dev  = ["pytest>=7", "pytest-cov", "ruff", "black", "mypy", "pre-commit"]
docs = ["sphinx", "myst-parser", "furo"]

[project.urls]
Homepage = "https://github.com/vruetten/wholistic_registration"
Issues   = "https://github.com/vruetten/wholistic_registration/issues"

[tool.setuptools.packages.find]
where = ["src"]
include = ["wholistic_registration*"]

[tool.setuptools_scm]
version_file = "src/wholistic_registration/_version.py"
```

Then:

```bash
git rm -r --cached src/wholistic_registration.egg-info
echo "*.egg-info/" >> .gitignore
```

---

## 2. Source Layout & Imports

### Findings

**Package skeleton:**

```
src/wholistic_registration/
├── __init__.py                  (0 bytes — empty)
├── pipeline.py                  (script-with-hardcoded-paths)
├── pipeline_vmsr.py             (another script-with-hardcoded-paths)
├── archive/                     (no __init__.py)
├── code/wholistic_registration/configs/   ← duplicated/orphaned tree
├── configs/                     (7 .toml files, no __init__.py)
├── core/
│   ├── __init__.py              (1 byte — whitespace)
│   └── main_function.py         (1586 LOC)
├── demos/                       (mix of .py + .ipynb, no __init__.py)
├── macros/                      (3 ImageJ .ijm files)
├── pipeline/
│   ├── HighResolution.md        (30 KB)
│   ├── pipeline.md              (developer TODO list)
│   └── images/                  (5 PNGs, 1–1.6 MB each — tracked)
├── registrated_data/            (672 MB, untracked, in working tree)
├── results/                     (4 binary outputs)
├── simulations/                 (MATLAB .m files)
├── tests/                       (1 .py + 3 notebooks + scripts)
└── utils/                       (15 .py files, ~3700 LOC)
```

**Critical import issues** (`grep ^from`):

| File | Bad import | Why broken |
|---|---|---|
| `pipeline.py` | `from core import main_function` | Only works if `cwd == src/wholistic_registration/`; fails for installed package |
| `pipeline.py` | `from utils import cp` | Same |
| `pipeline_vmsr.py` | `from core import main_function` | Same |
| `core/main_function.py` | `from utils import IO,reference,registration` | Same — and this is the *core* of the package |
| `tests/test_crossresolution_registration.py` | `from utils import …` | Same |
| `demos/demo2d.py` | `from utils import preprocess, …` | Same |
| `demos/demo_338.py` | `from utils import preprocess, calFlow3d_Wei_v1, **visulization**,mask,option` | Also **typo** `visulization` → import fails |
| `archive/demo_toy.py` | `from registration import motion_estimation` | Imports a non-existent `registration` top-level pkg |

All of these need to be `from wholistic_registration.core import …` and `from wholistic_registration.utils import …`.

**Other layout issues:**

- `src/wholistic_registration/code/wholistic_registration/configs/config_0120.toml` — a stray nested duplicate of the main config dir. Almost certainly an accidental copy or merge-leftover. Delete.
- Top-level `pipeline.py` and `pipeline_vmsr.py` are scripts (not modules) — both contain hardcoded paths, are mostly commented-out, and duplicate work that belongs in `core/`. Either:
  - move to `scripts/` or `bin/` outside the package, or
  - turn into a real `pipeline` subpackage with a `__main__.py` entry point.
- `__init__.py` files missing in 8 directories that contain Python or are referenced as packages.
- Mixed-language artifacts (`.m`, `.ijm`) should live in a top-level `tools/` or `extras/` directory outside the Python package, not inside it.

### Recommendations

```bash
# remove stray nested copy
rm -rf src/wholistic_registration/code

# move scripts out of the package
mkdir -p scripts
git mv src/wholistic_registration/pipeline.py scripts/run_pipeline.py
git mv src/wholistic_registration/pipeline_vmsr.py scripts/run_pipeline_vmsr.py

# move non-python tooling out
mkdir -p tools/imagej tools/matlab
git mv src/wholistic_registration/macros/*.ijm tools/imagej/
git mv src/wholistic_registration/simulations/*.m tools/matlab/

# move outputs/data out of source tree
git mv src/wholistic_registration/results examples/results          # if you want to keep tiny demo outputs
rm -rf src/wholistic_registration/registrated_data                  # actually 672 MB of personal data; should never be near the repo
```

Populate `src/wholistic_registration/__init__.py` with a thin public API:

```python
from ._version import __version__
from .core.main_function import (
    DefineParams,
    Registration_v3,
    ReliableAnalysis,
)
__all__ = ["__version__", "DefineParams", "Registration_v3", "ReliableAnalysis"]
```

Fix every `from utils import …` → `from wholistic_registration.utils import …` (or `from ..utils import …` inside the package). A one-shot `ruff --fix --select I` plus a couple of hand edits will do most of this.

---

## 3. Repo Hygiene

### Findings

`.gitignore` (the entire file):

```
results/*
*.DS_Store
*.png
*.tif
*.jpg
*.jpeg
*.gif
*.bmp
*.tiff
*.pyc / *.pyo / *.pyd ...
.DS_Store
src/.DS_Store
```

**Missing standard Python ignores:**
`__pycache__/`, `*.egg-info/`, `build/`, `dist/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `.ipynb_checkpoints/`, `venv/`, `.venv/`, `env/`, `.env`, `*.zarr/`, `*.h5`, `*.hdf5`, `*.nd2`.

**Files tracked in git that should not be:**
- `src/.DS_Store` (despite the ignore rule — added before the rule)
- All of `src/wholistic_registration.egg-info/` (5 files)
- 5 PNGs in `src/wholistic_registration/pipeline/images/` totalling **3.7 MB** (despite `**/*.png`)
- 2 large notebooks committed with outputs:
  - `demos/generateSimulation.ipynb` (1.78 MB)
  - `demos/demo_0912.ipynb` (453 KB)
  - `demos/demo_0822.ipynb` (454 KB)

**Working-tree pollution:**
- `src/wholistic_registration/registrated_data/` — **672 MB** of registration output. Not tracked, but should not be inside `src/`.
- `__pycache__/` directories in `core/` and `utils/`.
- `.pytest_cache/`.

### Recommendations

Replace `.gitignore` with the GitHub Python template plus project specifics:

```gitignore
# Byte-compiled / optimized
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
build/
dist/
*.egg-info/
*.egg
.eggs/

# Tooling caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/

# Virtual envs
.venv/
venv/
env/
.env

# Editors / OS
.DS_Store
*.swp
.idea/
.vscode/

# Jupyter
.ipynb_checkpoints/

# Large binary / scientific data — must not enter the repo
*.nd2
*.h5
*.hdf5
*.zarr/
*.tif
*.tiff
*.png
*.jpg
*.jpeg

# Project specific
results/
registrated_data/
src/wholistic_registration/registrated_data/
```

Then untrack the offenders:

```bash
git rm --cached src/.DS_Store
git rm -r --cached src/wholistic_registration.egg-info
git rm --cached src/wholistic_registration/pipeline/images/*.png
# Replace any docs that referenced these PNGs with links to GitHub-hosted assets,
# or commit small SVGs / store the PNGs in a docs branch or LFS.
```

Strip notebook outputs before committing — install pre-commit + `nbstripout`:

```bash
pre-commit install
# in .pre-commit-config.yaml: include nbstripout
```

---

## 4. Code Quality

### Findings

**Style / config:**
- **No formatter, no linter, no type checker, no editor config.** No `ruff.toml`, no `pyproject.toml [tool.ruff]`, no `[tool.black]`, no `.editorconfig`, no `mypy.ini`.
- Naming: `IO.py`, `ImmuneCell.py` (modules must be `lower_snake_case`); functions `DefineParams`, `Registration_v3`, `ReliableAnalysis` (functions must be `snake_case` — these read like classes).
- Mixed quote styles, mixed indent widths, occasional trailing whitespace.

**Bad patterns observed (counts, v1 only):**
- ~220 `print()` calls — should be `logging.getLogger(__name__).info(...)` (esp. in `core/main_function.py: 101`, `utils/IO.py: 48`).
- `utils/__init__.py` prints 3 status messages on every `import wholistic_registration.utils`. Library code must never `print` at import time.
- `5 bare/over-broad except blocks` (silently swallow errors).
- Duplicate functions: `readMeta`/`readMeta_new`, `saveTiff`/`saveTiff_new` in `utils/IO.py` (982 LOC) — stale lineage, no deprecation path.
- **Duplicate definition** of `create_downsample_dataset_v4` at lines 1163 and 1232 of `core/main_function.py`. The first is dead (shadowed). Also called recursively from line 1276 inside the second definition.
- Re-imports of `numpy` and `scipy.ndimage` 2× back-to-back in `utils/preprocess.py:20–24`.

**Hardcoded paths (in 30+ files):** `/home/cyf/wbi/...`, `/nrs/ahrens/Virginia_nrs/...`. These will leak somebody's filesystem layout to anyone who clones the repo and break every demo for new users. All paths should be CLI args, config-driven, or env vars.

**File sizes (worth splitting):**

| File | LOC |
|---|---|
| `utils/calFlowCrossResolution.py` | 1974 |
| `core/main_function.py` | 1586 |
| `utils/calFlow3d_Wei_v1.py` | 1090 |
| `utils/IO.py` | 982 |
| `utils/reliableAnalysis.py` | 707 |

Files >1 kLOC are red flags. `main_function.py` mixes config validation, IO, pipeline orchestration, downsampling, OME-TIFF writing, and reliable analysis — clear cohesion problem.

**No type hints** anywhere.

### Recommendations

Add a minimal but opinionated tooling stack (`pyproject.toml` additions):

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "RUF"]
ignore = ["E501"]   # line length governed by formatter

[tool.ruff.format]
quote-style = "double"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
warn_unused_ignores = true
strict_optional = true
files = ["src/wholistic_registration"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

Add `.editorconfig`, `.pre-commit-config.yaml` (black + ruff + nbstripout + check-yaml + end-of-file-fixer + trailing-whitespace), and run a one-time formatting pass.

Replace all `print()` with module-level loggers; default to `logging.basicConfig(level=logging.INFO)` only inside CLI entry points, never in library modules.

Split `main_function.py` into:
- `config.py` (loading + validation; `DefineParams` should arguably be a `dataclass`/`pydantic.BaseModel` with `from_toml(...)`)
- `pipeline/registration.py` (`Registration_v3` flow)
- `pipeline/reliable_analysis.py`
- `pipeline/downsample.py`
- `io/ometiff.py`

Delete the second definition of `create_downsample_dataset_v4` (and the surrounding dead code), or merge differences into a single canonical function.

---

## 5. Tests

### Findings

`src/wholistic_registration/tests/`:

```
test.py                            ← interactive script (#%% cells, hardcoded /nrs/ahrens/ paths)
test_HR.ipynb                      ← notebook
test_F260517.ipynb                 ← notebook
test_mask.ipynb                    ← notebook (untracked? — verify)
test_vectorization.py              ← educational benchmark, not a test
test_crossresolution_registration.py ← script with broken `from utils import …` imports
zarr_to_tiffseries.py              ← utility script
```

- **No pytest-style tests** (no `test_*` functions with `assert`s).
- **No CI run** would catch anything because there's no `tests/` at the project root, no `.github/workflows/`, no `tox.ini`, no `noxfile.py`.
- Notebooks are committed with outputs (some 450 KB+).
- `tests/` lives **inside the package**, which is unusual and makes `pytest --pyargs wholistic_registration` find scripts that aren't tests.

### Recommendations

```
tests/                       # at repo root, NOT inside the package
├── conftest.py
├── unit/
│   ├── test_io.py
│   ├── test_preprocess.py
│   ├── test_reference.py
│   ├── test_calculate.py
│   └── test_calflow.py
├── integration/
│   └── test_registration_smoke.py    # uses tiny synthetic data fixture
└── fixtures/
    └── synthetic_5x5x3.tif
```

- Generate tiny synthetic volumes in-memory; never depend on real ND2 paths.
- Add `pytest-cov`, target ≥40 % to start, ratchet up.
- Move notebooks into `examples/` (executable but not test artefacts) and strip outputs via `nbstripout`.

---

## 6. CI/CD & Automation

### Findings

- **No `.github/`** directory at all. No Actions, no issue templates, no PR template, no Dependabot, no CodeQL.
- **No pre-commit config.**
- **No release automation** (no `release-please`, no tags, no `setuptools-scm`).
- No changelog. No version-bump workflow.

### Recommendations

Add `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push: { branches: [main] }
  pull_request:

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest]
        python: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy src
      - run: pytest --cov=src/wholistic_registration --cov-report=xml
      - uses: codecov/codecov-action@v4
        with: { file: coverage.xml }
```

Add a release workflow that:
1. Triggers on tag `v*`.
2. Builds wheel + sdist with `python -m build`.
3. Publishes to (Test)PyPI via OIDC trusted publisher.

Add `dependabot.yml` for `pip` and `github-actions`.

Add `.pre-commit-config.yaml` covering ruff, black, nbstripout, end-of-file-fixer, trailing-whitespace, check-toml.

---

## 7. Documentation

### Findings

- `README.md` (76 lines): high-level marketing only. **No installation instructions, no quickstart, no Python API example, no link to docs.** Has two embedded video links and a license blurb.
- `pipeline/HighResolution.md` (30 KB) — actually useful pipeline description, but lives in an obscure path inside the package.
- `pipeline/pipeline.md` — a personal TODO journal mixed with deployment notes ("Things to check (top priority)…", "Ginny needs to check this"). This should not be public-facing.
- No `docs/` directory, no Sphinx/MkDocs config.
- No `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.
- No docstrings on `__init__.py` (it's empty); module docstrings exist on most utils but are mixed-style and sometimes contain non-English comments untranslated.

### Recommendations

1. **README.md** — expand with at minimum: install (`pip install wholistic_registration[gpu]`), 10-line quickstart, link to docs, link to citation. Move the "BSD 3-Clause License" body block out (it's in LICENSE; README just needs a one-liner).
2. **Move `pipeline/HighResolution.md` → `docs/algorithm.md`** and set up minimal MkDocs with `mkdocs-material`.
3. **Convert `pipeline/pipeline.md` into**:
   - `CHANGELOG.md` for resolved/done items
   - GitHub Issues for outstanding TODOs
   - Delete the rest.
4. **Add boilerplate**:
   - `CONTRIBUTING.md` (how to set up dev env, run tests, code style)
   - `CODE_OF_CONDUCT.md` (Contributor Covenant)
   - `SECURITY.md` (private vulnerability reporting)
   - `CHANGELOG.md` (Keep-a-Changelog format)
5. **Citation file**: add `CITATION.cff` so GitHub renders a "Cite this repository" button.

---

## 8. Licensing & Legal

### Findings

- `LICENSE` (BSD-3-Clause) ✅ present, correctly formatted.
- `pyproject.toml` license field uses `{text = "BSD-3-Clause"}` — the modern PEP 639 form is `license = "BSD-3-Clause"` (string) or `license = { file = "LICENSE" }`.
- No SPDX headers on source files (low priority but nice for clarity).
- `README.md` re-paraphrases the BSD text; ok, but watch for drift vs `LICENSE`.

No copyright violations spotted in cursory scan; nothing vendored from third parties without attribution.

---

## 9. Security & Secrets

### Findings

- No `.env`, no API keys spotted.
- Hardcoded **internal NFS paths** (`/nrs/ahrens/...`) and **home directories** (`/home/cyf/...`) — minor information disclosure once the repo goes public.
- No SECURITY.md / private vuln reporting channel.
- No dependency pinning → supply-chain risk.

### Recommendations

- Replace every hardcoded path with `argparse`/`click` CLI args or environment variables. Use `pathlib.Path` throughout.
- Add `SECURITY.md`.
- Pin runtime deps with a lockfile (`uv pip compile` or `pip-tools` produces `requirements.lock`).
- Enable Dependabot / Renovate.
- Consider `bandit` in CI.

---

## 10. Git Hygiene

### Findings

- Branches: 5 local, several stale (`backup_with_bad_commit`, `fixup_main`, `temp-branch` 52 commits behind, `vmsr`).
- **17 stashes** dating back to "WIP on main" of pre-2025 commits. High risk of permanently losing work to GC.
- No `.gitattributes` — line endings and binary handling unconfigured.
- No GPG-signed commits.

### Recommendations

- Triage and drop dead stashes (`git stash list`, then `git stash drop stash@{N}` for each obsolete one). For valuable ones, branch them off (`git stash branch wip/...`) so they're addressable.
- Delete merged/orphan branches.
- Add `.gitattributes`:
  ```gitattributes
  * text=auto eol=lf
  *.png binary
  *.tif binary
  *.ipynb -text merge=jupyter
  ```

---

## 11. Notebooks

### Findings

- 8 tracked notebooks, several committed with outputs.
- Notebooks contain hardcoded `/nrs/ahrens/...` paths and rely on `importlib.reload` patterns (interactive only — should not be committed as canonical examples).

### Recommendations

- Adopt `nbstripout` via pre-commit hook.
- Move user-facing notebooks to `examples/` and write a `README.md` per example explaining what data is needed.
- Convert the smallest demo into a `tests/integration/test_demo.py` so the example is verified by CI.

---

## Prioritized Remediation Plan

### Phase 1 — Make it installable & usable (one afternoon)
1. Fix `pyproject.toml`: drop `"json"`, add the missing deps with floors, fill in metadata, add `[tool.setuptools.packages.find]`.
2. Fix every broken `from utils …` / `from core …` import in v1 code.
3. Populate `src/wholistic_registration/__init__.py` with the public API.
4. Untrack: `*.egg-info/`, `src/.DS_Store`, big PNGs in `pipeline/images/`.
5. Replace `.gitignore` with the standard Python template.
6. Delete `src/wholistic_registration/code/` (stray nested duplicate).
7. Delete dead duplicate `create_downsample_dataset_v4` definition.
8. Remove or refactor the import-time `print`s in `utils/__init__.py`.

### Phase 2 — Tooling & CI (one day)
9. Add `[tool.ruff]`, `[tool.black]`, `[tool.mypy]`, `[tool.pytest.ini_options]`.
10. Add `.pre-commit-config.yaml` and run `pre-commit run --all-files` (commit the formatting pass separately).
11. Add `.editorconfig`, `.gitattributes`.
12. Add `.github/workflows/ci.yml` (lint + type-check + tests on 3.10–3.12).
13. Add `dependabot.yml`.

### Phase 3 — Tests & docs (a few days)
14. Move `tests/` to repo root; write real pytest tests using synthetic data.
15. Expand README with install/quickstart/usage; move `HighResolution.md` to `docs/`.
16. Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, `CITATION.cff`.

### Phase 4 — Structural refactor (ongoing)
17. Split `core/main_function.py` (1586 LOC) into cohesive submodules.
18. Audit and de-duplicate `utils/IO.py` (`readMeta` vs `readMeta_new`, etc.).
19. Rename modules/functions to PEP 8 (`IO.py` → `io.py`, `DefineParams` → `define_params`, etc.) — use a deprecation shim for one minor release.
20. Replace all `print()` with `logging`.
21. Move scripts (`pipeline.py`, `pipeline_vmsr.py`) out of the package; convert to `console_scripts` entry points or `python -m wholistic_registration ...`.
22. Move `macros/` (`.ijm`) and `simulations/` (`.m`) out of the Python package into top-level `tools/`.
23. Type-annotate the public API; ratchet up mypy strictness.

### Phase 5 — Release engineering
24. Adopt `setuptools-scm` for dynamic versioning from git tags.
25. Add a release workflow to PyPI via OIDC trusted publisher.
26. Tag `v0.1.0` after Phase 1+2 land.

---

## Appendix A — Quick wins (copy-paste)

```bash
# 1. Stop tracking the obvious offenders
git rm --cached src/.DS_Store
git rm -r --cached src/wholistic_registration.egg-info
git rm --cached src/wholistic_registration/pipeline/images/*.png

# 2. Remove orphan trees
rm -rf src/wholistic_registration/code
rm -rf src/wholistic_registration/registrated_data    # 672 MB; back up first if needed

# 3. Untrack pycache (if any sneaks in)
find . -type d -name __pycache__ -exec git rm -rf --cached {} + 2>/dev/null || true
```

## Appendix B — Tooling-config seed files

(Provided inline above for `ruff`, `black`, `mypy`, `pytest`, GitHub Actions CI.)
