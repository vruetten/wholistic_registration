# Pass 2 — math & numerics

**Date:** 2026-08-10 · **Method:** four parallel reviewers running ground-truth
experiments on CPU (numpy fallback of the `cp` shim) — analytic cases,
reference implementations (scipy/skimage/MATLAB semantics), gradient checks,
and synthetic-recovery tests — plus a GPU job (`153303848`, L4, cupy 13.4.1)
re-verifying Pass-1 `[gpu-unverified]` items against real cupy. All findings
below are CONFIRMED-run by their own executed experiments; the significant one
(M-C1) is independently triple-corroborated (two Pass-1 verifiers + agent C).

## Verdict

**The numerical core of the package is sound.** The four deep dives came back
overwhelmingly clean:

- `imresize` is a faithful MATLAB `imresize`/`imresize3` port: Keys a=−0.5
  kernel exact to 9e-16, correct half-pixel grid mapping (symmetric inputs
  stay symmetric across 7 size combinations), exact antialiasing on
  downsample, exact symmetric boundary reflection, all three axes identical.
- The warp/interp core is empirically correct end-to-end: identity bit-exact;
  motion component c displaces axis c (roll-test, all axes — **no
  transposition**); subpixel interpolation matches `scipy.ndimage.shift`
  exactly; warping is pull/backward and `getMotion`'s update sign is
  consistent — a synthetic +1-voxel shift is recovered with correct sign,
  axis, and magnitude, **including in z under anisotropic zRatio=2** (the
  `Iz/zRatio` → `update/zRatio` round-trip is self-consistent).
- The LK solve's adjugate inverse matches `np.linalg.solve` to 2.9e-12 over
  15,000 random tensors — no component swap. `imfilter(output="same")` is
  bit-identical to scipy, including asymmetric kernels. `getSmPnltNormFctr`
  is exact on analytic ramps, and the mask polarity question left open in
  Pass 1 is **resolved: mask True = outlier = excluded, consistent
  end-to-end**.
- cyf's sparse-compact mode decomposition matches its methods write-up
  term-for-term and is mathematically correct: exact synthetic recovery
  (activation |corr| = 1.0000; overlapping modes at SNR 25 → 0.9946+),
  analytic gradients verified to 1.1e-8, the prox is the exact weighted
  group-lasso operator (stationarity 8.7e-19), objective monotone, SVD
  K-selection energies match `np.linalg.svd`, reported R² matches independent
  recomputation to 5.8e-12.

## Findings

| ID | Sev | Location | Claim | Status |
|---|---|---|---|---|
| M-C1 | 🟨 | utils/reference.py:84 | `CCsort[:, 1:ncorr]` averages ncorr−1 values (off-by-one): T=2 → all-NaN scores → silently selects frame 0; T=3 → single-value "average". At healthy T the selected frame **set** never changed in 20 trials (imax moved in 3/20). Fix: `1:ncorr+1` + guard T≥3. | run |
| M-C4 | 🟨 | utils/mask.py:83 | getMask uses non-robust global mean/std: a bright artifact inflates σ (10→1013) so a coexisting dim artifact (z=1.16) is missed entirely; `robust_mean_std` already exists in preprocess.py | run |
| M-B3 | 🟨 (latent) | calFlow3d_Wei_v1.py:822,948 | getMapping's z phase-update factor is the reciprocal of its init factor → returned z-motion scaled by (zRatio/zRatio_hr)² (measured 1 → 4.000 at zRatio=2); phase converges anyway (GN absorbs it) but motion output / movRange / penalty mix units. Newer cfcr.py copy removed exactly this factor. Dead code today (B-094). | run |
| M-C2 | 🟨 (latent) | calculate.py:58 | `imfilter(output="valid")` broken 3 ways (size-1 kernel dim → empty output; 3D third axis never trimmed; even kernels over-trim). Zero callers — proper filing of Pass-1's refuted-as-unreachable B-099. | run |
| M-D1 | 🟦 | motion_correlation_pattern.py:1935 | `K_min > Kmax` silently overrides the Kmax cap (Kmax=2, K_min=4 → K=4); doc presents Kmax as ceiling | run |
| M-C3 | 🟦 (latent) | calculate.py:50 | even-size kernels align one pixel differently from MATLAB imfilter (all pipeline kernels are odd) | run |
| M-A1 | 🟦 | imresize.py:114,133 | uint8 path rounds .5 ties to even (numpy) vs MATLAB half-away-from-zero → ±1 gray on exact ties; float pipeline unaffected | run |
| M-C5 | 🟦 | mask.py:95, preprocess.py:47 | doc nits: "4x4x1" comment vs (3,3,1) kernel; docstring axis order (z,x,y) vs actual (x,y,z) | read |
| M-D2/3/4 | info | mcp.py:1770,1885,1735 | signed-vs-absolute convergence test (functionally equivalent); two recon-loss conventions coexist (all R² verified correct); step_H logged nonzero on a no-step branch | run |

Duplicates absorbed: agent B's M-B1/M-B2 = Pass-1 B-094/B-095 (dead
`getMapping` NameError / UnboundLocalError) — not renumbered.

## GPU re-verification (Pass-1 `[gpu-unverified]` items, real cupy 13.4.1 on L4)

- **B-001 fix verified on GPU**: `HAS_CUPY=True` and `cupy_ndi.median_filter`
  executed on-device. The one-character fix restores GPU acceleration.
- **B-019 → CONFIRMED-run**: cupyx `binary_closing` matches scipy exactly —
  structure `cgf+2` merges gaps ≤ `cgf+1` (one wider than documented) and
  erodes border-touching runs, on-device.
- **B-089 → CONFIRMED-run**: with real cupy, `neiDiff[:, :, 2]` on a
  (4,4,7,3) control-point grid selects the (4,4,3) z-slice×components block,
  and nz=2 raises IndexError — the wrong-axis claim holds on GPU.
- **B-090 → CONFIRMED-run**: `_make_ball(2, z_ratio=5)` spans **21 z-slices**
  where physical isotropy dictates 1.
- **B-034 scope narrowed**: the module imports fine with real cupy (RawKernel
  objects construct on a GPU node) — the import bomb is strictly a
  CPU-fallback problem.
- Still pending: a functional GPU run of `getMotionUnit` (needs 3-array
  synthetic inputs; signature captured) for B-017/B-020's GPU half.

## Regression suite (Pass 1 exit debt — closed)

`tests/regressions/`: **20 passing tests** pinning the 26 fixed findings'
correct behavior + **5 strict-xfail tests** pinning deferred bugs (B-018,
B-049, B-058, B-046, B-016 — each verified to fail for its documented
mechanism; a future fix surfaces as XPASS and forces promotion). Full tree:
`37 passed, 5 xfailed`.

## Coverage

Executed-ground-truth coverage: imresize (kernel, grid mapping, antialiasing,
boundaries, 3 axes, sizing) · interp/warp (identity, convention, subpixel,
direction, boundary, end-to-end shift recovery in x and z) · calculate
(determinants/adjugate, imfilter both modes, hann2d, zncc) · preprocess
(getSmPnltNormFctr + mask polarity) · mask (getMask semantics,
bwareafilt3 bounds/connectivity) · reference (correlation math, selection,
degenerate T) · mode decomposition (recovery, objective, prox, gradients,
Lipschitz, μ update, retraction, K selection, spec-vs-code). Not covered:
GPU-only kernels' numerical output (RawKernel projection kernels — reachable
only on-device; deferred to a cluster-side Pass 3 profiling session), and the
full Registration_v3 pipeline on real data (needs cluster + data).
