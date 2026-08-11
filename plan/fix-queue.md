# Fix queue — remaining bug fixes, explained

**Updated:** 2026-08-11. One entry per open finding, in review order, each with
a plain-language explanation, expected impact, and who needs to decide.
Evidence for every item: `audit/pass-1-bugs.md` / `audit/pass-2-math.md` and
the verification log. Fixed items live in the ledger's Fix log — not here.

Review protocol (agreed 2026-08-10): nothing lands without Virginia's per-item
sign-off; fixes are applied to the working tree, verified with a before/after
repro, and committed one bug per commit only on her word.

---

## 1 · Awaiting Virginia's verdict now

### ~~M-C1 — reference scoring off-by-one~~ **RESOLVED** (committed, see ledger)
`pick_initial_reference` averages the top correlations excluding self, but the
slice `1:ncorr` drops one column: it averages ncorr−1 values, and for tiny
reference blocks it degenerates — a 2-frame block produces all-NaN scores and
silently picks frame 0; a 1-frame block yields an all-NaN reference (B-077).
The applied fix uses `1:ncorr+1` and adds a guard that averages everything for
degenerate blocks. **Impact:** on healthy windows the selected frame set was
unchanged in 20/20 trials — effectively only the broken small-window paths
change. Verified: T=1/2/3 now finite, planted-cluster selection still exact,
suite green. **Committed 2026-08-11; also resolves B-077's T=1 NaN path.**

### B-076 — backward reference seed under-filled (main_function.py:780)
While registering the middle block in batches, the pipeline saves the first
`reference_chunk` registered frames as the seed reference for the backward
pass (`head_mem`) and the last ones for the forward pass (`tail_mem`).
`tail_mem` is topped up every batch; `head_mem` only during batch 0 — so it
can never exceed `batch_size` frames. With `config_f2013`
(`reference_chunk`=400 frames, `batch_size`=100) the backward pass's first
reference averages 100 frames where you configured 400 — a noisier reference,
silently. **Fix:** remove the batch-0 gate so the buffer fills until it holds
`reference_chunk` frames (the first ones — correct contents). **Impact:**
no-op whenever `reference_chunk ≤ batch_size` (most configs); f2013-style
configs get the reference window they asked for.

### B-018 — the whole-body-motion artifact filter never fires (motion_correlation_pattern.py:1407)
The filter's documented job is to discard episodes that are "just the whole
sample moving together". It should compare each episode's mean patch motion
with the global (whole-body) motion — correlation ≈ 1 ⇒ artifact ⇒ drop. The
bug: it correlates the per-frame *deltas* against the *cumulative* global
signal; a signal and its own increments are nearly uncorrelated, so the
textbook artifact scores 0.06 against the 0.90 threshold and is kept.
Measured: with `motion_abs` (cumulative vs cumulative) the same episode scores
1.00. **Impact:** the filter becomes real — whole-body episodes start being
discarded, so episode counts and downstream patterns will differ from all
previous runs (which is the documented intent). Caveats: the 0.90 threshold
has never been exercised on data, and this is cyf's active file. A strict
xfail test pins the bug and will flip when fixed.

---

## 2 · High-confidence next tranche (Claude recommends; results change by design)

### ~~B-070 — warm-start motion fed back transposed~~ **RESOLVED → 521cf33** (severity raised to 🟥 after re-verification)
The 3D registration returns each motion field transposed to (Z,Y,X,3), but the
next batch re-ingests `motion[-1]` into a slot expecting (X,Y,Z,3); `imresize`
then silently stretches the garbled field into shape. Net effect: every batch
after the first is initialized with spatially scrambled motion instead of the
previous batch's solution — the warm start has been noise. **Fix:** transpose
back at the hand-off (one line) — components don't need permuting (verified:
component c ↔ axis c). **Impact:** warm starts become meaningful; likely
faster convergence / better continuity between batches. Shape-level logic is
verified; a full-pipeline GPU validation run is recommended before trusting
new outputs. **Committed 2026-08-11 after fresh re-verification: unfixed warm
start corr 0.50 (x-comp) with wrong 0.33x/3.0x magnitude divisors; fixed
round-trip bit-exact.**

### B-071 — the dataset's real zRatio never reaches the algorithm (registration.py:262)
`DefineParams` reads the true z-anisotropy into `config["MetaData"]["zRatio"]`,
but nothing copies it into the `option` dict the flow solver reads — so every
pipeline 3D run has used the hardcoded default **27.693** regardless of
objective/z-step. Pass 2 verified the solver handles zRatio *correctly* when
given one — it's just never given yours. **Fix:** one line in
`wbi_registration_3d`: `option["zRatio"] = config["MetaData"]["zRatio"]`.
**Impact:** z-motion scaling becomes dataset-correct; results change for any
dataset whose true ratio ≠ 27.693 — that's the point, but past/future outputs
won't be comparable. Sanity-check the config value on a real run first.

### ~~M-C4 — getMask misses dim artifacts when a bright one coexists~~ **RESOLVED → 3a3bb10**
Outlier masking thresholds on |z-score| with a global mean/std — but the std
is computed *including* the outliers. A large bright artifact inflates σ
(measured 10 → 1013), so a coexisting dim artifact sits at |z|=1.16 and is
missed entirely. **Fix:** use the robust (percentile-clipped) mean/std that
already exists in `preprocess.robust_mean_std`. **Committed 2026-08-11 —
verified: dim artifact 0.00→1.00 caught, single-artifact cases unchanged,
clean-data masks measured IDENTICAL post-morphology (better than the
'slight change' predicted).**

### B-075 — one frame registered twice at the middle/forward seam (main_function.py:649/855/1437)
The middle block registers frames `mid_start..mid_end` **inclusive**; the
forward pass then starts at `mid_end` — so that one frame is registered twice
and its output overwritten with a different reference (and its own converged
motion is used to warm-start its re-registration). **Fix options:** start the
forward pass at `mid_end + 1` (keep the middle-block result), or make the
middle block exclusive. Small, but changes which reference produced one output
frame per run. Backward seam verified clean.

### B-058 — canny edge-map suppresses against the wrong diagonal (preprocess.py:307)
Non-maximum suppression folds the gradient angle with `abs()` instead of
mod-180°, so angles in (−157.5°,−112.5°)∪(−67.5°,−22.5°) are compared against
the perpendicular diagonal — measured ~15% differing edge pixels on curved
edges (under-suppression: thicker/spurious edges). **Fix:** fold with
`angle[angle<0] += 180`. Strict-xfail test pins it. **Impact:** edge maps
change on diagonal/curved structure.

### B-049 — event detection's `mad_k` knob is inert on sparse traces (motion_correlation_pattern.py:5001)
Class activation traces are mostly exact zeros, so median=0 and MAD=0: the
threshold collapses to ~1e-12 and *any* nonzero frame counts as an event —
mad_k=3 and mad_k=1000 give identical output (measured). The sibling function
already guards this with a std fallback. **Fix:** same fallback here. cyf's
file; strict-xfail test pins it. **Impact:** event detection starts respecting
mad_k on exactly the traces the caller produces.

---

## 3 · Virginia's design decision

### B-063 — `maximum` vs `minimum` in structural_difference_map (reliableAnalysis.py:447)
The comment block above the line documents *why minimum is required* ("only
trust difference where BOTH images have structure; maximum flags background →
false positives"), and your commit `b17778d`'s message says "set back to
minimum — to be checked" — but its diff flipped the code **to maximum**, and a
`# minimum?...` was left on the line. Code, comment, and commit message
disagree three ways; no downstream compensation exists. **Decision needed:**
minimum (per your own comment — recommended) or maximum (then rewrite the
comment). Changes the reliability weighting of difference maps.

---

## 4 · Needs cyf's sign-off (his files / his semantics)

- **B-089** 🟥 `neiDiff[:,:,2] *= zRatio_hr` (calFlowCrossResolution:2063) —
  scales z-*slice* 2 instead of the z-*component*; crashes when nz≤2.
  GPU-verified with real cupy. Fix is `[..., 2]`, but it changes the
  smoothness regularization of every HR run, and the final-error recompute
  applies no scaling at all (second inconsistency to resolve with him).
- **B-090** 🟧 `_make_ball` uses the reciprocal zRatio convention — trap-mask
  ball spans 21 z-slices where physics says 1 (GPU-measured). Fix is invert
  the factor; affects wrong-region reruns incl. current zRatio_HR=1 runs at
  coarse layers.
- **B-091** 🟨 `option["tol"]` ignored whenever wrong-region correction is on
  (default): plumb `tol` through `correct_wrong_regions_one_layer`.
- **B-092** 🟨 "highresidual" mode excludes the most-*improved* voxels instead
  of the highest-residual ones — inverts name, docstring, and the detection
  stage's own criterion. Likely `error_last0` was meant.
- **B-017/B-019/B-020** 🟨 gap-closing semantics: CPU path ignores
  `close_gap_frames` entirely (CPU/GPU divergence); the GPU closing merges
  gaps one frame wider than documented and erodes border-touching runs;
  explicit `use_gpu=True` without cupy needs a clear error. Fix as one
  coherent decision about intended gap semantics.
- **B-008** 🟧 `merge_small_regions=True` is a silent no-op (fragments are
  discarded before the merge helper runs) and the metadata misreports the
  strategy — needs cyf's intended merge semantics.
- **B-013/B-014/B-016/B-046** 🟨 API semantics: dead `enforce_spatial_gate`
  flag; OR-vs-documented-AND thresholds in pattern overlap; mode-unit
  patterns' union path dead (support_mask never read); `lag_mode` value sets
  incompatible between sibling APIs. Each is a small fix once he confirms
  intent (xfail tests pin B-016/B-046).
- **B-035/B-036** 🟨 latent: MAD=0 flags everything in `detect_significant_mad`
  (currently zero callers); z_init rint-vs-truncate off-by-one for fractional
  spacing.
- **B-039/B-040/B-042** 🟧 visualization: gallery overlay misaligned
  (transposed background, patch-vs-pixel units); region arrows vs mode
  quivers draw perpendicular-swapped directions for the same field (evidence
  says the mode-quiver "xy" default is the wrong side); diagnostics ignore
  the stored `use_velocity` flag. Fix after he confirms the intended
  component convention.
- **M-D1** 🟦 `K_min > Kmax` silently overrides the Kmax cap in mode-count
  selection — document or clamp.

## 5 · Restructuring / environment work (bigger than one-line fixes)

- **B-067** 🟥 unconditional `import cupy` in `process_directional_chunks`
  kills serial CPU runs — route through the shim; part of the larger
  AUDIT.md F2 import cleanup.
- **B-034** 🟧 module-level `cp.RawKernel` makes calFlowCrossResolution
  unimportable on CPU machines (GPU nodes fine — verified) — wrap kernels in
  lazy initialization.
- **B-024/B-031/B-032** 🟧/🟨 ND2/TIFF shape handling: single-channel 2D ND2s
  crash metadata reading (nd2 drops singleton axes); 3D→5D padding puts
  z-stacks in the channel slot; single-frame ND2s crash the framerate
  fallback. Fix together as one IO shape-handling pass with synthetic-file
  tests.

## 6 · Minor / latent cleanups (batch when convenient)

Latent-but-real in dead or unreachable paths: B-094/B-095/M-B3 (dead
`getMapping`: missing import, unbound variable, reciprocal z-units — consider
deleting the function instead), B-096 (wrong-axis zRatio, sole caller passes
1), B-097/B-098 (imresize dead 2D/`"org"` paths), M-C2 (`imfilter "valid"`
broken, zero callers), M-C3 (even-kernel MATLAB alignment). Cosmetic: B-003
(half-initialized empty MotionPattern), B-011 (+B-010) dead duplicate
splitting helpers, B-045/B-053 figure leaks in 7 viz functions, B-065 dead
threshold param, B-102 GPU-resident error_log entries, B-038/B-093 dead
allocations, M-A1 uint8 tie-rounding, M-C5/M-D2/M-D3/M-D4 doc/diagnostic
nits, D-001 (`ImmuneCell.py` dead — delete or archive).
