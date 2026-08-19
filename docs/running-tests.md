# Running the tests before you push

**Audience:** anyone pushing to `wholistic-registration`, written for cyf after the
2026-08-12 sync. **Owner:** Virginia. **Last updated:** 2026-08-19.

The suite takes **under 4 seconds** on CPU and needs no data, no GPU, and no
`/home/cyf` or `/groups` paths. Running it before `git push` is the whole protocol.

---

## Why this file exists

On 2026-08-12 two commits (`8c52fbe`, `2d64301`) were pushed straight to `main`.
GitHub Actions ran and reported, in 50 seconds, on Python 3.10, 3.11 and 3.12:

```
============= 12 failed, 54 passed, 4 xfailed, 2 warnings in 3.85s =============
```

Twelve tests that passed on the parent commit failed. Eleven of them guard fixes
that had been reviewed and merged earlier the same day; the gap-closing set (PR #32)
was merged 47 minutes before the push that removed it. The twelfth follows from
deleting `Yunfeng_edge_map` while its regression test still calls it. `main` has been
red ever since.

What the twelve failures said (quoted from CI run `31600233151`):

```
test_b002_safe_corr_constant_traces          - assert 0.0 == 1.0 ± 1.0e-05
test_b009_split_mode_survives_single_nan_pixel - assert 0 == 1
test_b012_activationless_pair_...            - UnboundLocalError: local variable 'D_h' referenced before assignment
test_b018_artifact_filter_...                - assert 1 == 0
test_b041_unfitted_episode_...               - KeyError: 'B'
test_b043_tab20_colormap_lookup_works        - assert 'cm.get_cmap' not in '...'
test_b044_k0_episode_source_viz_...          - ValueError: Number of rows must be a positive integer, not 0
test_b019_closes_exactly_gaps_up_to_...      - AssertionError: gap 1 should close at n=1
test_b019_border_runs_are_not_eroded         - AssertionError: n=1: [(0, 3), (5, 6), (19, 22)]
test_b017_gpu_branch_also_closes_gaps        - AssertionError: expected one closing call per backend, got []
test_b020_explicit_use_gpu_without_cupy_...  - AttributeError: 'NoneType' object has no attribute 'asarray'
test_b055_yunfeng_edge_map_runs              - AttributeError: module '...preprocess' has no attribute 'Yunfeng_edge_map'
```

Each line names the test and the value it got. The same lines appear locally from
the command in the next section.

---

## One-time setup

Do this once, in the conda env you already use for this package (yours is
`wbi_cuda124`, Python 3.10.19, going by the notebook kernels you commit).

```bash
conda activate wbi_cuda124
cd /path/to/wholistic_registration      # the repo root, where pyproject.toml is
pip install -e ".[dev]"
```

`[dev]` is declared in `pyproject.toml` and pulls in `pytest`, `pytest-cov`,
`imagecodecs`, `ruff`, `black`, `mypy`. `-e` installs the package in editable mode,
so the tests import the source tree you are editing rather than a stale copy.

If you would rather not add anything else to your working env, `pip install pytest`
on its own runs the suite: no test file under `tests/` imports `imagecodecs`, `ruff`,
`black` or `mypy` (checked by grep). The rest of `[dev]` is for linting and for
`utils/converters.py`, which has no test yet.

Check the install:

```bash
python -c "import wholistic_registration, pytest; print(wholistic_registration.__file__)"
```

The path printed must be your working copy's `src/wholistic_registration/__init__.py`.
If it points anywhere else, an older copy of the package is shadowing the repo and
every test result you get is about that other copy.

---

## The command

From the repo root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

That is the whole thing. Notes on each piece:

- `python -m pytest` rather than bare `pytest` runs the pytest belonging to the
  active env, not whichever one is first on `$PATH`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` stops unrelated installed plugins (napari,
  numba) from loading and clashing with numpy. CI sets the same variable. Leave it
  out and you may see import errors that have nothing to do with your change.
- Paths, options and markers come from `[tool.pytest.ini_options]` in
  `pyproject.toml`: `testpaths = ["tests"]`, `addopts = "-ra --strict-markers
  --import-mode=importlib"`. Do not pass a directory; running plain `pytest` from
  the root already collects the right tests.

**Only the top-level `tests/` directory holds pytest tests.**
`src/wholistic_registration/tests/` contains notebooks and machine-specific debug
scripts, some named `test_*.py`. They are excluded by `testpaths` and are not run by
CI. Running `pytest` from inside that directory collects them by accident and the
result means nothing.

### Faster loops while you work

```bash
# one file
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/regressions/test_motion_correlation_pattern.py

# one test, with full output
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/regressions/test_motion_correlation_pattern.py::test_b018_artifact_filter_discards_pure_whole_body_motion -vv

# stop at the first failure
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -x

# re-run only what failed last time
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --lf
```

---

## Reading the result

A clean run ends like this. Quoted from CI run `31593179106`, the last green commit
on `main` (`3168417`), which is the state `main` is supposed to be in:

```
======================== 66 passed, 4 xfailed in 2.80s =========================
```

Three outcomes and what each one means:

| Outcome | Meaning | What to do |
|---|---|---|
| `passed` | the test ran and its assertions held | nothing |
| `xfailed` | a *known, documented, unfixed* bug, listed in `tests/regressions/test_deferred_xfail.py` and waiting on a decision | nothing — expected |
| `failed` | an assertion that held before does not hold now | **do not push** |

`xpassed` is worth a look too: it means a bug listed as unfixed now behaves
correctly. Usually good news, but it means the ledger is out of date.

The count of `xfailed` should stay at 4 and `passed` at 66. If `xfailed` rises or
`passed` falls without a matching `failed`, something that used to be tested is no
longer being tested.

### Python version matters for the count

CI runs Python 3.10, 3.11 and 3.12. On **Python 3.9** two test files fail to
*collect*, because `src/wholistic_registration/utils/generate_demo_data.py:43` uses
`tuple[float, float] | None`, which needs Python ≥ 3.10:

```
TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'NoneType'
```

That error is pre-existing and unrelated to any current work, but it hides two test
files, so a 3.9 run reports fewer failures than are really there. Your `wbi_cuda124`
is 3.10.19, so you will see the full picture. If you ever run on 3.9, know that the
number you get is a lower bound.

---

## The push protocol

```bash
# 1. start from what is on the server, before you edit anything
git checkout main
git pull --rebase

# 2. work on a branch, not on main
git checkout -b cyf/<short-description>

# 3. before every push
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

# 4. only if the failure count is 0
git push -u origin cyf/<short-description>
gh pr create        # or open the PR in the web UI
```

What each step does:

1. **`git pull --rebase` before you start editing.** On 2026-08-12,
   `motion_correlation_pattern.py` was written on top of a copy taken before eleven
   fixes landed, and pushing it replaced the fixed file wholesale. The new work in
   that commit is untouched by this — the file simply carried an older base. Rebasing
   first turns the stale regions into conflicts you resolve, instead of an overwrite
   git cannot warn about, because a whole-file replacement has nothing to conflict
   with.
2. **A branch and a PR instead of pushing to `main`.** CI runs on pull requests too,
   so a red result blocks the PR rather than breaking `main` for everyone.
3. **Running the suite before pushing.** Four seconds. All twelve failures appear on
   your own machine, with the same assertion text CI printed — `wbi_cuda124` is
   Python 3.10.19, so nothing is hidden from you.
4. **A commit per logical change.** `8c52fbe` bundles a z-penalty fix, a new pattern
   feature, two file deletions and a safety check into one commit touching 1044 lines.
   Separate commits can be reviewed and reverted independently; a bundled one can
   only be taken or dropped whole.

### If you cannot run the suite

Push to a branch and open a draft PR anyway. CI runs the same command on three
Python versions and posts the result in about 50 seconds. Do not push to `main` and
read CI afterwards — by then everyone else has pulled the red commit.

---

## When a test fails

The failure line names the function and the value. Work out **which of the two is
wrong — the test or the code** before touching either.

- **The code is wrong.** Usual case, and usually means a merged fix was overwritten.
  `git log -p --follow <file>` or `git log --oneline <parent>..origin/main` shows
  what landed while you were working. Rebase and keep both changes.
- **The test is wrong.** Possible, and then the test must be changed — but derive
  the new expected value from the specification or from first principles, in a
  calculation that never calls the function under test. Never adjust an expected
  value to match what the code currently prints: that turns the test into an
  assertion that the code does what the code does.
- **The behaviour changed on purpose.** Then the test is telling you the change is
  results-changing. Say so in the PR, and let the reviewer decide, rather than
  editing the test in the same commit.

`CLAUDE.md` §"Test discipline" is the full rule set, including the falsification
gate every new regression test has to pass. Read it before adding a test.

---

## Deleting a function

`8c52fbe` deleted `canny_edge_map` and `Yunfeng_edge_map` from `preprocess.py`.
Neither has a caller in `src/`. Both have callers in `tests/`, and three open GitHub
items describe them: PR #24, issue #25 and issue #33.

Before deleting anything, check for all of these:

```bash
grep -rn "<function_name>" src tests          # callers, including tests
gh issue list --search "<function_name>"      # open issues about it
gh pr list --search "<function_name>"         # open PRs touching it
```

If any of the three turns up a hit, the deletion is a decision rather than
housekeeping. Put the reason in the commit message, delete the tests in the same
commit, and close the issues and PRs as moot. What must not happen is the function
disappearing while its tests and its issues stay open, because then the tracker
describes code that does not exist.

---

## Reference

| Thing | Where |
|---|---|
| Test config (`testpaths`, `addopts`) | `pyproject.toml`, `[tool.pytest.ini_options]` |
| Dev dependencies | `pyproject.toml`, `[project.optional-dependencies] dev` |
| CI definition | `.github/workflows/ci.yml` |
| CI results | `gh run list` · https://github.com/vruetten/wholistic-registration/actions |
| Test discipline rules | `CLAUDE.md`, §"Test discipline" |
| Known-unfixed bugs (the 4 xfails) | `tests/regressions/test_deferred_xfail.py` |
| Open findings and their status | `plan/fix-queue.md` |
| The 2026-08-12 audit | `audit/cyf-2026-08-12-regression-audit.md` |
