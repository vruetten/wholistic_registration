# Audit — cyf commits `8c52fbe` + `2d64301` (2026-08-12)

**Reviewed:** `8c52fbe` "fix the penalty on z direction in calFlowCrossResolution.py,
update some new functions in motion_correlation_pattern.py, delete no longer required
functions in preprocess.py, add safety check in reliableAnalysis.py" and `2d64301`
"add a demo to debug the projection functions", both by Yunfeng Chi
(`chiyf21@mails.tsinghua.edu.cn`), fetched into `main` on 2026-08-19.

**Parent:** `3168417` (merge of PR #32), which is the tip of the 2026-08-10/11 fix batch.

**Method:** commit-range diff read in full, plus the pytest suite executed in the
`wholistic-registration` conda env on `login1.int.janelia.org` from a clean clone at
each of the two commits. CuPy 13.4.1 is importable in that env, so the GPU-path
findings below are executed, not read.

## Status as of 2026-08-19

Every finding below has a PR. None is merged.

| Finding | PR | State |
|---|---|---|
| R-1 B-001 | [#42](https://github.com/vruetten/wholistic-registration/pull/42) | restores the `cupyx.scipy.ndimage` import |
| R-1 B-002 | [#37](https://github.com/vruetten/wholistic-registration/pull/37) | |
| R-1 B-009 | [#38](https://github.com/vruetten/wholistic-registration/pull/38) | |
| R-1/R-3 B-012 | [#45](https://github.com/vruetten/wholistic-registration/pull/45) | whitespace-only re-indent; `git diff -w` is empty |
| R-1 B-017/019/020 | [#41](https://github.com/vruetten/wholistic-registration/pull/41) | reopened issue [#31](https://github.com/vruetten/wholistic-registration/issues/31) |
| R-1 B-018 | [#44](https://github.com/vruetten/wholistic-registration/pull/44) | **results-changing, needs cyf sign-off**; reopened issue [#12](https://github.com/vruetten/wholistic-registration/issues/12) |
| R-1 B-041 | [#46](https://github.com/vruetten/wholistic-registration/pull/46) | |
| R-1 B-043 | [#39](https://github.com/vruetten/wholistic-registration/pull/39) | |
| R-1 B-044 | [#40](https://github.com/vruetten/wholistic-registration/pull/40) | |
| R-4 B-055 | [#43](https://github.com/vruetten/wholistic-registration/pull/43) | **proposal, needs cyf sign-off** — restore the function or delete its test |
| R-5 | [#35](https://github.com/vruetten/wholistic-registration/pull/35) | filed as B-118, issue [#34](https://github.com/vruetten/wholistic-registration/issues/34) |
| R-2, R-6, R-7 | none | no fix prepared; R-2 needs cyf's call on deleting the two call sites |

The ten restore branches were merged together onto `2d64301` and the suite run on
that integrated tree: **55 passed, 1 skipped, 4 xfailed, 0 failed** (measured,
Janelia conda env `wholistic-registration`, Python 3.9.22, `PYTHONPATH=<repo>/src`,
with the two files that cannot collect on 3.9 ignored). All ten merge without
conflict. Symbol counts for `compute_pattern_unified_mode` (8), `compute_unified_mode`
(5), `spatial_rule` (16), `centroid_dist_thresh` (13), `unified_mask` (23) and
`unified_activation` (7) are identical on `origin/main` and on the integrated tree,
and the integration diff removes no line mentioning any of them: the new work in
`8c52fbe` survives the restoration intact.

One caveat on B-001 ([#42](https://github.com/vruetten/wholistic-registration/pull/42)):
it does not change the CI count. `.github/workflows/ci.yml` installs without GPU
extras, so `import cupy` fails on the runner and `HAS_CUPY` is `False` however the
next line is spelled (verified on CI run `32257616775`). On a GPU host the count does
drop, but the affected test goes from failed to *skipped*, not to passed.

## Tally

**GitHub Actions already reported this.** Both cyf pushes turned CI red and `main`
has been red since 2026-08-12T12:31:49Z: run `31596879566` (`8c52fbe`) and run
`31600233151` (`2d64301`), each failing in under 60 s on all three of Python 3.10,
3.11 and 3.12. CI result, quoted from run `31600233151`:

```
============= 12 failed, 54 passed, 4 xfailed, 2 warnings in 3.85s =============
```

Reproduced on `login1.int.janelia.org` in the `wholistic-registration` conda env
(Python 3.9.22, CuPy 13.4.1), from a clean clone at each commit, with
`python -m pytest -q --ignore=tests/regressions/test_small_utils.py
--ignore=tests/unit/test_generate_demo_data.py`:

| Commit | Result |
|---|---|
| `3168417` (parent) | 55 passed · 1 skipped · 4 xfailed · **0 failed** |
| `2d64301` (cyf HEAD) | 45 passed · 4 xfailed · **11 failed** |

The local count is 11 and the CI count is 12. The difference is `test_small_utils.py`,
which this ledger had to `--ignore` because it fails to *collect* on Python 3.9 for a
pre-existing reason (`generate_demo_data.py:43` uses `tuple[float, float] | None`,
which needs Python ≥ 3.10). CI runs 3.10+, collects the file, and reports the twelfth
failure: `test_b055_yunfeng_edge_map_runs`. **CI's 12 is the correct number**; use it.

Eleven of the twelve are regression tests guarding fixes landed on 2026-08-10/11.
`8c52fbe` rewrites `motion_correlation_pattern.py` from a pre-fix copy of the file,
so eleven merged fixes are gone from `main`. The twelfth is the edge-map deletion
covered in R-4.

## R-1 🟥 · Eleven merged fixes are reverted by `8c52fbe`

The diff is not a set of edits on top of the fixed file; it is a wholesale replacement
whose baseline predates the fix batch. Each row below is a bug that was fixed, tested,
merged, and is now back:

| ID | Fix commit | What the revert restores | Test that now fails |
|---|---|---|---|
| B-001 | `c81e861` | `from cupyx.scipy import ndi as cupy_ndi` — `cupyx.scipy.ndi` does not exist, the `ImportError` is swallowed by `except Exception`, so `HAS_CUPY` is `False` on GPU machines | (surfaces via B-017) |
| B-002 | `5b3c725` | `_safe_corr` constant-trace fallback | `test_b002_safe_corr_constant_traces` |
| B-009 | `c33e6ce` | `np.nanmax` → `np.max` in `split_mode_to_regions` | `test_b009_split_mode_survives_single_nan_pixel` |
| B-012 | `83fd55b` | stale `D_h`/`D_b`/`sign_j` reuse in `compute_region_distance_matrix_simple` | `test_b012_activationless_pair_gets_invalid_activation_not_stale_distance` |
| B-017 | `ab9d47e` | `_close_temporal_gaps` deleted; CPU path loses gap closing entirely while the GPU path keeps an inline version | `test_b017_gpu_branch_also_closes_gaps` |
| B-018 | `4535d36` | `ep.motion_abs` → `ep.motion_delta` in `filter_episodes_artifacts` | `test_b018_artifact_filter_discards_pure_whole_body_motion` |
| B-019 | `ab9d47e` | inline structure length `close_gap_frames + 2` (bridges gaps of ≤ n+1, contract says ≤ n) and no edge padding, so runs touching t=0 / t=T−1 are eroded | `test_b019_closes_exactly_gaps_up_to_close_gap_frames`, `test_b019_border_runs_are_not_eroded` |
| B-020 | `ab9d47e` | `_resolve_use_gpu` deleted; `use_gpu=True` without CuPy now raises `AttributeError: 'NoneType' object has no attribute 'asarray'` instead of a named `RuntimeError` | `test_b020_explicit_use_gpu_without_cupy_raises_clearly` |
| B-041 | `867e82c` | `_get_BH_from_episode` guard weakened to `mode_model is None`, so the default `mode_model={}` passes the guard and `KeyError`s on `["B"]` | `test_b041_unfitted_episode_skipped_and_valueerror` |
| B-043 | `8bfd0e2` | `plt.get_cmap` → `cm.get_cmap`, deprecated in Matplotlib 3.7, removal now scheduled for 3.11 (see correction below) | `test_b043_tab20_colormap_lookup_works` |
| B-044 | `603282b` | `n == 0` guards dropped from `visualize_episode_sources_overview` and `compare_sources_to_observed_frames` | `test_b044_k0_episode_source_viz_returns_cleanly` |

Two of these are worth calling out for downstream impact rather than test colour:

- **B-001 kills the whole GPU path of the module.** Measured on the server:
  `cupyx.scipy.ndimage` imports; the code's own `from cupyx.scipy import ndi` raises
  `ImportError: cannot import name 'ndi' from 'cupyx.scipy'` (the module-import form
  `import cupyx.scipy.ndi` raises `ModuleNotFoundError` instead — the `from`-form is
  what the file uses); and `motion_correlation_pattern.HAS_CUPY` is `False` while a
  standalone `import cupy` succeeds (13.4.1). Every other module in the package spells it `cupyx.scipy.ndimage`.
  With B-020 also reverted, an explicit `use_gpu=True` no longer errors clearly — it
  dereferences `cp = None`.
- **B-018 correlates a velocity against a cumulative displacement.**
  `ep.global_motion` is built from `motion_full_abs` (`_compute_global_motion_series`
  call at line 1275), so pairing it with `ep.motion_delta` compares series in different
  units and the whole-body-motion artifact filter never fires.

### Correction to the B-043 row (2026-08-19)

`8bfd0e2`'s commit message said `cm.get_cmap` was "removed in matplotlib 3.9", and
the first version of this ledger repeated that. The real sequence: deprecated in 3.7,
removed in 3.9.0, restored in 3.9.1 as deprecated, removal rescheduled for 3.11. The
`AttributeError` recorded in `8bfd0e2` was measured on Matplotlib 3.11.1.

On the Janelia conda env, which has **Matplotlib 3.9.4** (measured), `cm.get_cmap("tab20")`
does not raise — it returns a `ListedColormap` and emits `MatplotlibDeprecationWarning`.
So on that environment `test_b043_tab20_colormap_lookup_works` is failing on its
source-text assertion (`assert "cm.get_cmap" not in inspect.getsource(mcp)`), which is
the test's first statement, and not on a runtime failure. The test's second half, which
renders and checks scatter offsets and face colours, would pass there on the unfixed
code. The declared floor `matplotlib>=3.7` (`pyproject.toml:14`) admits both raising and
non-raising versions, so the fix is still correct — the reach of the bug is narrower
than the original message claimed.

## R-2 🟥 · Two functions deleted while still called

`8c52fbe` deletes `compute_lagged_ca_correlation_map` and
`get_top_ca_sites_from_corr_map`, but `analyze_roi_motion_activation_ca` still calls
both — `motion_correlation_pattern.py:6268` and `:6277`. The call site is reached
whenever `ca_patch_stack is not None` and at least one class activation trace has
`nanstd >= 1e-8`, so that branch raises `NameError` on the first live run. No test
covers the branch, so the suite does not show it; the calls are confirmed present by
reading the file and by an AST scan for called-but-undefined names.

## R-3 🟧 · `compute_region_distance_matrix_simple` reuses the previous pair's distance

This is the concrete shape of the reverted B-012. The `if not np.isfinite(D_h) ...`
block was dedented out of the `else:` that defines `D_h`, `D_b`, `sign_j`
(lines 3271–3288). When `h_i is None or h_j is None`, the branch sets
`info = {... "reason": "invalid_activation" ...}` and then falls through to the
dedented block, which reads `D_h` and `D_b` from the **previous loop iteration** and
overwrites `info` with a `"reason": "ok"` entry carrying that stale distance. On the
first such pair in the loop, `D_h` is unbound and the function raises
`UnboundLocalError` instead. Either way the pair is scored wrong or the call dies.

## R-4 🟧 · The edge-map deletions break one live test and hollow out two xfails

`8c52fbe` also deletes `canny_edge_map` and `Yunfeng_edge_map` from `preprocess.py`.
Neither has a caller in `src/`, but both have callers in `tests/`:

- **`test_b055_yunfeng_edge_map_runs`** (`tests/regressions/test_small_utils.py:76`)
  is an ordinary, non-xfail test. It now fails in CI with
  `AttributeError: module 'wholistic_registration.utils.preprocess' has no attribute 'Yunfeng_edge_map'`.
  This is the twelfth CI failure, invisible to a Python 3.9 run.
- **`test_b058_canny_edge_map_matches_mod180_reference`** and
  **`test_b046_lagged_ca_correlation_accepts_lag_mode_both`** still report XFAIL, but
  for the wrong reason. Run with `--runxfail` they fail with
  `AttributeError: module ... has no attribute 'canny_edge_map'` /
  `... 'compute_lagged_ca_correlation_map'`, not with the defect each `reason=` string
  documents. Both are declared `strict=True` (`test_deferred_xfail.py:85` and `:103`),
  which does not help here: `strict` only converts an unexpected *pass* into a failure
  and says nothing about *why* a test failed, so an `AttributeError` from a deleted
  function counts as the expected failure. Both tests now certify nothing and hide the
  deletions.

The deletions are also not free of open work. Three GitHub items are about functions
that no longer exist: PR [#24](https://github.com/vruetten/wholistic-registration/pull/24)
(B-058 canny fold), issue [#25](https://github.com/vruetten/wholistic-registration/issues/25)
(B-115 canny diagonal pairs), and issue [#33](https://github.com/vruetten/wholistic-registration/issues/33)
(B-117 `Yunfeng_edge_map` defaults inert). If the deletions stand, all three close as
moot and the three tests get deleted with them. Deleting a function that a regression
test and two open issues describe is a decision, not housekeeping, so it needs cyf's
stated reason on the record before `main` is cleaned up either way.

## R-5 🟧 · `getMotionPattern` returns `patterns` filtered but `groups`/`labels` unfiltered

The new pre-filter (`min_pattern_members`) and post-filter (`min_unified_area`,
`min_h_snr`, `max_h_cv`) drop entries from `patterns`. `kept_units`, `groups`,
`labels`, and `info["labels"]` are returned unchanged. `build_motion_patterns_from_groups`
establishes `patterns[i] ↔ groups[i]`; after filtering, that index correspondence is
broken and any caller zipping the two reads the wrong group for every pattern past the
first drop.

## R-6 🟨 · Three quality-filter parameters are silently inert

The post-hoc quality-filter block is nested inside `if compute_unified and len(patterns) > 0:`.
With `compute_unified=False`, `min_unified_area`, `min_h_snr`, and `max_h_cv` are
accepted, echoed back in `info["params"]`, and never applied. `min_pattern_members`
is applied unconditionally, so the grouping is inconsistent within one function.

## R-7 🟨 · `unified_h` is renormalised but `unified_B` is not

In `compute_pattern_unified_mode`, `unified_h` is divided by its L2 norm ("V2
convention") after averaging, while `unified_B` is a plain per-patch mean of the member
`B`s. The outer product `unified_B ⊗ unified_h` therefore no longer carries the members'
motion amplitude — the reconstruction is off by the discarded `‖mean h‖` factor. The
filled entries add a second scale question: existing entries come from the mode fit,
filled entries from `b_new = M[patch]ᵀ h / (h·h)` computed with the member's own,
un-renormalised `h`. Both are defensible if `h` is unit-norm in the fit, which the code
comment assumes ("should be ≈1 for V2") but does not check.

## Accepted without change

- **`neiDiff[:,:,2] *= zRatio_hr` → `neiDiff[...,2]`** in
  `calFlowCrossResolution.py:2111`. Correct, and identical to the pending B-089 fix on
  `fix/b089-neidiff-z-component`. That branch can be dropped in favour of this commit.
- **`_estimate_structural_difference_scale`** in `reliableAnalysis.py`. On the normal
  path (≥100 pixels with `R > r_threshold`) it reproduces the old
  `np.percentile(diff[R > r_threshold], 99)`. Below that count it no longer falls back
  to the whole-volume distribution but lowers the reliability cutoff to admit the ~100
  most reliable positive-`R` pixels, using the whole volume only when no pixel has
  `R > 0`. It adds non-finite filtering, an `eps` floor against division by zero, and a
  diagnostics dict. One behaviour change to note: the old guard was `len(valid_vals) > 100`,
  the new one `n_above_threshold >= min_reliable_pixels`, so exactly 100 pixels now takes
  the fixed-threshold path instead of the fallback. The change does not touch the
  `maximum`/`minimum` reliability line that B-063 is about.
- **`_regions_spatially_compatible` gating** in `compute_region_distance_matrix_simple`.
  The default `spatial_rule="iou"` with `iou_thresh=min_iou` reproduces the previous
  `iou < min_iou` gate exactly, and the three other rules are opt-in.
- **`2d64301` / `test_projection_single_frame.py`.** A machine-specific debug script
  under `src/wholistic_registration/tests/`, which `testpaths = ["tests"]` excludes from
  collection, so it cannot break CI. It hardcodes `/home/cyf/...` paths and
  `sys.path.insert`s a foreign checkout, consistent with the rest of that directory. The
  `test_` prefix is a collection hazard only for someone running pytest from inside that
  directory.

## Recommended remediation

1. Re-apply the eleven reverted fixes onto the current `main` (R-1). The pre-fix
   regions are small and localised; `git show` of each fix commit against the current
   file is the cheapest route. Re-run the suite until it is back to 0 failed.
2. Restore `compute_lagged_ca_correlation_map` and `get_top_ca_sites_from_corr_map`
   from `3168417`, or delete their call site in `analyze_roi_motion_activation_ca` (R-2).
   Deleting the call site is a behaviour change and needs cyf's sign-off.
3. Delete `test_b058_...` and `test_b046_...` from `tests/regressions/test_deferred_xfail.py`,
   and close the B-058 PR as moot (R-4).
4. Filter `groups`/`labels`/`kept_units` alongside `patterns`, or return the surviving
   indices so callers can re-key (R-5); move the quality-filter block out of the
   `compute_unified` branch, or document the three parameters as requiring it (R-6).
5. Ask cyf whether `unified_B` should be scaled by the pre-normalisation `‖mean h‖` (R-7).
6. Agree a sync protocol with cyf before the next batch: pull `main` and rebase rather
   than committing a whole-file copy. A `pytest` run before pushing would have caught
   ten of the eleven reverts.
