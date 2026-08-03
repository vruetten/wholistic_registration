#!/usr/bin/env python3
"""
F260517 Registration — Flat Z-Plane Projection Pipeline  (2025-06-25)

Key settings:
  - smoothPenalty_raw = 0.03
  - mask_ref & smoothPenalty: computed ONCE at init
  - Ref update every 40 frames, calibration target = RAW moving frames
  - Projection: GPU splatting onto fixed_target_z (from warmup 0-4)
  - Metrics: mov vs mem_mapped  (warped reference, same as v3/v4)

Output (under f260517_0625/):
  raw_moving_mem/          raw moving membrane per frame
  raw_moving_sparseCell/   raw moving sparse-cell per frame
  projected_mem/           z-plane-projected membrane
  projected_sparseCell/    z-plane-projected sparse-cell
  diagnostics/             CSVs (errors_membrane, errors_sparse, hole_summary)
"""

import os, sys, time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as label_ndi, sobel
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# GPU + paths
# ---------------------------------------------------------------------------
cp.cuda.Device(1).use()

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
SRC_DIR = PKG_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PKG_DIR))
sys.path.insert(0, str(HERE))

from utils import IO, calFlowCrossResolution, mask, preprocess as prep
from utils.calFlowCrossResolution import project_coords_to_fixed_planes_gpu
import f260517_helpers as fh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
F260517_mov_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
F260517_ref_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_anat_00003_TZCYX.ome.tiff"

BASE_OUT = Path("/home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0625")
DIRS = {
    "raw_moving_mem":        BASE_OUT / "raw_moving_mem",
    "raw_moving_sparseCell": BASE_OUT / "raw_moving_sparseCell",
    "projected_mem":         BASE_OUT / "projected_mem",
    "projected_sparseCell":  BASE_OUT / "projected_sparseCell",
    "diagnostics":           BASE_OUT / "diagnostics",
}
for d in DIRS.values():
    os.makedirs(str(d), exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
option = {}
option["r"] = 5
option["layer"] = 3
option["iter"] = 10
option["movRange"] = 5.0
option["tol"] = 1e-6
option["zRatio_HR"] = 1
option["wrong_region_enable"] = False

thresFactor = 5.0
maskRange = [5.0, 4000.0]
smoothPenalty_raw = 0.03
ref_update_every = 40
Z_WINDOW = 3.0
FILL_VALUE = -200.0
WARMUP_FRAMES = [0, 1, 2, 3, 4]

percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]

# ===========================================================================
# 1. Load data
# ===========================================================================
print("=" * 80)
print("F260517 Z-Plane Projection Pipeline")
print(f"  smoothPenalty_raw = {smoothPenalty_raw}")
print(f"  ref_update_every  = {ref_update_every}")
print("=" * 80)

print("\n[1/7] Loading data ...")
t0 = time.time()
F260517_mov, _ = IO.readTiff(F260517_mov_path)
F260517_ref, _ = IO.readTiff(F260517_ref_path)

ref_mem_raw    = F260517_ref[90:310, 1, :, :].astype(np.float32)
ref_sparse_raw = F260517_ref[90:310, 0, :, :].astype(np.float32)
mov_mem_all    = F260517_mov[:, :, 1, :, :].astype(np.float32)
mov_sparse_all = F260517_mov[:, :, 0, :, :].astype(np.float32)

print(f"  loaded in {time.time()-t0:.1f}s")

# ===========================================================================
# 2. z_init + coords
# ===========================================================================
print("\n[2/7] Initial setup ...")
z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
    mov_mem_all[0].transpose(2, 1, 0),
    ref_mem_raw.transpose(2, 1, 0),
    delta_ref_idx=10, use_gradient=False,
)
z_init = z_init.astype(np.float32)
z_idx = np.rint(z_init).astype(np.int32)
z_idx = np.clip(z_idx, 0, ref_mem_raw.shape[0] - 1)

K, T = int(z_init.shape[0]), int(mov_mem_all.shape[0])

x_coord = np.arange(mov_mem_all[0].shape[2], dtype=np.float32)
y_coord = np.arange(mov_mem_all[0].shape[1], dtype=np.float32)
k_coord = np.arange(K, dtype=np.int32)
X_grid, Y_grid, K_grid = np.meshgrid(x_coord, y_coord, k_coord, indexing="ij")
coords_xyz = np.empty((len(x_coord), len(y_coord), K, 3), dtype=np.float32)
coords_xyz[..., 0] = X_grid
coords_xyz[..., 1] = Y_grid
coords_xyz[..., 2] = z_init[K_grid]
option["phase"] = coords_xyz.copy()

print(f"  K={K}  T={T}")

# ===========================================================================
# 3. Initial reference calibration (mask + smoothPenalty ONCE)
# ===========================================================================
print("\n[3/7] Initial ref calibration ...")
init_target = np.mean(mov_mem_all[WARMUP_FRAMES].astype(np.float32), axis=0)
ref_mem_adj, src_q_fixed, tgt_q_current, _ = fh.update_reference_intensity_mapping_from_target_stack(
    F260517_ref_mem=ref_mem_raw,
    target_stack_zyx=init_target, z_idx=z_idx, option=option,
    thresFactor=thresFactor, maskRange=maskRange,
    smoothPenalty_raw=smoothPenalty_raw, percentiles=percentiles,
)
print("  mask_ref & smoothPenalty fixed.")

# ===========================================================================
# 4. Warmup: register frames 0-4, determine fixed_target_z
# ===========================================================================
print("\n[4/7] Warmup — registering frames 0-4 ...")
warmup_phase = {}

for idx, i in enumerate(WARMUP_FRAMES):
    if idx == 0:
        option["phase"] = coords_xyz.copy()
        option.pop("motion", None)

    mov_i = mov_mem_all[i].transpose(2, 1, 0).astype(np.float32, copy=False)
    option["mask_mov"] = mask.getMask(mov_i, thresFactor)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

    phase_new, motion_current, _ = calFlowCrossResolution.getMotion_v2(
        mov_i, ref_mem_adj, option, verbose=False)

    if hasattr(phase_new, "get"):      phase_new = phase_new.get()
    if hasattr(motion_current, "get"): motion_current = motion_current.get()

    warmup_phase[i] = np.asarray(phase_new, dtype=np.float32)
    option["motion"] = (0.7 * np.asarray(motion_current, dtype=np.float32))
    print(f"  frame {i} done")

# Determine fixed_target_z
target_z_list = []
for i in WARMUP_FRAMES:
    tz, _ = fh.estimate_projection_z_from_phase_simple(
        phase_new=warmup_phase[i], z_init=z_init, ref_shape=ref_mem_raw.shape,
        ref_volume_order="zyx", method="trimmed_mean", trim_percentiles=(5, 95), frame_idx=i)
    target_z_list.append(tz)
fixed_target_z = fh.robust_average_target_z(target_z_list, method="median")
fixed_target_z[~np.isfinite(fixed_target_z)] = z_init[~np.isfinite(fixed_target_z)]
print(f"  fixed_target_z: {fixed_target_z}")

# ===========================================================================
# Metrics helpers
# ===========================================================================

def zncc_2d(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    a_c, b_c = a - np.mean(a), b - np.mean(b)
    numer = np.dot(a_c, b_c)
    denom = np.sqrt(np.dot(a_c, a_c) * np.dot(b_c, b_c) + eps)
    return float(numer / denom) if denom >= eps else np.nan


def symmetric_edge_distance_2d(a, b):
    gx_a, gy_a = sobel(a.astype(np.float32), axis=-1, mode='nearest'), sobel(a.astype(np.float32), axis=-2, mode='nearest')
    gx_b, gy_b = sobel(b.astype(np.float32), axis=-1, mode='nearest'), sobel(b.astype(np.float32), axis=-2, mode='nearest')
    mag_a, mag_b = np.sqrt(gx_a**2 + gy_a**2 + 1e-8), np.sqrt(gx_b**2 + gy_b**2 + 1e-8)
    ea = (mag_a >= np.percentile(mag_a, 90)).astype(np.uint8)
    eb = (mag_b >= np.percentile(mag_b, 90)).astype(np.uint8)
    if not np.any(ea) or not np.any(eb): return np.nan
    dt_b, dt_a = distance_transform_edt(1 - eb), distance_transform_edt(1 - ea)
    return float(0.5 * (np.mean(dt_b[ea > 0]) + np.mean(dt_a[eb > 0])))


def sparse_centroid_metrics_2d(mov_p, mapped_p, thresh=3.0, radius=5.0):
    def get_centroids(img):
        mu, sig = float(np.mean(img)), float(np.std(img))
        bm = img > (mu + thresh * sig)
        if not np.any(bm): return np.empty((0, 2), dtype=np.float32)
        lbl, _ = label_ndi(bm)
        return np.array([p.centroid for p in regionprops(lbl)], dtype=np.float32)
    cm, cp = get_centroids(mov_p), get_centroids(mapped_p)
    if len(cm) == 0 or len(cp) == 0: return np.nan, np.nan, np.nan
    d_m2p = np.array([np.min(np.sqrt(np.sum((cp - c)**2, axis=1))) for c in cm])
    d_p2m = np.array([np.min(np.sqrt(np.sum((cm - c)**2, axis=1))) for c in cp])
    return (float(0.5*(np.nanmean(d_m2p)+np.nanmean(d_p2m))),
            float(np.mean(d_m2p <= radius)), float(np.mean(d_p2m <= radius)))


def compute_frame_metrics(mov_zyx, mapped_zyx, mask_mov_zyx=None):
    """Compute per-plane membrane metrics on (K,Y,X) arrays."""
    Kk = mov_zyx.shape[0]
    out = {"MAE": [], "nMAE": [], "NCC": [], "edge": []}
    for kk in range(Kk):
        mm, mp = mov_zyx[kk], mapped_zyx[kk]
        valid = np.ones_like(mm, dtype=bool)
        if mask_mov_zyx is not None:
            valid = mask_mov_zyx[kk].astype(bool)
            if not np.any(valid): valid = np.ones_like(mm, dtype=bool)
        diff = np.abs(mm.astype(np.float32) - mp.astype(np.float32))
        diff[~valid] = 0.0
        mae = float(np.sum(diff) / mm.size)
        p1, p99 = np.percentile(mm[valid], [1, 99])
        dyn = max(p99 - p1, 1e-8)
        out["MAE"].append(mae); out["nMAE"].append(mae/dyn)
        out["NCC"].append(zncc_2d(mm[valid], mp[valid]))
        out["edge"].append(symmetric_edge_distance_2d(mm, mp))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def compute_sparse_metrics(mov_zyx, mapped_zyx, mask_mov_zyx=None):
    Kk = mov_zyx.shape[0]
    out = {"MAE": [], "nMAE": [], "NN": [], "recall": [], "precision": []}
    for kk in range(Kk):
        ms, mp = mov_zyx[kk], mapped_zyx[kk]
        valid = np.ones_like(ms, dtype=bool)
        if mask_mov_zyx is not None:
            valid = mask_mov_zyx[kk].astype(bool)
            if not np.any(valid): valid = np.ones_like(ms, dtype=bool)
        diff = np.abs(ms.astype(np.float32) - mp.astype(np.float32))
        diff[~valid] = 0.0
        mae = float(np.sum(diff) / ms.size)
        p1, p99 = np.percentile(ms[valid], [1, 99])
        dyn = max(p99 - p1, 1e-8)
        out["MAE"].append(mae); out["nMAE"].append(mae/dyn)
        nn, rec, prec = sparse_centroid_metrics_2d(ms, mp)
        out["NN"].append(nn); out["recall"].append(rec); out["precision"].append(prec)
    return {k: float(np.nanmean(v)) for k, v in out.items()}


# ===========================================================================
# 5. Forward loop
# ===========================================================================
print("\n[5/7] Forward loop ...")
print(f"      Ref update every {ref_update_every} frames (raw moving target)")

registered_cache = {}
error_mem = []
error_sparse = []
hole_records = []

frames_since_ref_update = 0
ref_update_id = 0

option["phase"] = coords_xyz.copy()
option.pop("motion", None)

total_start = time.time()

for i in range(0, T):
    frame_start = time.time()

    raw_mem_zyx    = mov_mem_all[i]
    raw_sparse_zyx = mov_sparse_all[i]
    mov_mem_xyk    = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)

    option["mask_mov"] = mask.getMask(mov_mem_xyk, thresFactor)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

    phase_new, motion_current, mem_mapped_xyk = calFlowCrossResolution.getMotion_v2(
        mov_mem_xyk, ref_mem_adj, option, verbose=False)

    if hasattr(phase_new, "get"):      phase_new = phase_new.get()
    if hasattr(motion_current, "get"): motion_current = motion_current.get()
    if hasattr(mem_mapped_xyk, "get"): mem_mapped_xyk = mem_mapped_xyk.get()

    phase_new      = np.asarray(phase_new, dtype=np.float32)
    motion_current = np.asarray(motion_current, dtype=np.float32)
    mem_mapped_zyx = np.asarray(mem_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)  # (K,Y,X)

    registered_cache[i] = mem_mapped_zyx

    # ---- Z-plane projection ----
    phase_for_proj = fh.upsample_phase_xy_for_supersurface(phase_new, upsample_factor=2)
    raw_mem_xyk  = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    raw_sparse_xyk = raw_sparse_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    mem_vals  = fh.upsample_values_xy_for_supersurface(raw_mem_xyk, upsample_factor=2, order=1)
    sparse_vals = fh.upsample_values_xy_for_supersurface(raw_sparse_xyk, upsample_factor=2, order=0)

    proj_mem_zyx = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_proj, ref_volume=ref_mem_adj,
        target_z_planes=fixed_target_z, values_xyk=mem_vals,
        ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
        fill_value=FILL_VALUE, return_numpy=True, output_order="zyx",
        xy_splat_mode="subpixel_footprint", xy_extra_radius=0)
    proj_sparse_zyx = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_proj, ref_volume=ref_sparse_raw.transpose(2,1,0),
        target_z_planes=fixed_target_z, values_xyk=sparse_vals,
        ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
        fill_value=FILL_VALUE, return_numpy=True, output_order="zyx",
        xy_splat_mode="subpixel_footprint", xy_extra_radius=0)

    # Coverage map
    cov = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_proj, ref_volume=ref_mem_adj,
        target_z_planes=fixed_target_z,
        values_xyk=np.ones(phase_for_proj.shape[:-1], dtype=np.float32),
        ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
        fill_value=0.0, return_numpy=True, output_order="zyx",
        xy_splat_mode="subpixel_footprint", xy_extra_radius=0)
    if hasattr(cov, "get"): cov = cov.get()
    cov = np.asarray(cov, dtype=np.float32)
    hole_frac = float(np.mean(cov == 0))
    hole_per_k = [float(np.mean(cov[kk] == 0)) for kk in range(cov.shape[0])]

    hole_records.append({
        "frame": i, "ref_update_id": ref_update_id,
        "hole_frac_global": hole_frac,
        "max_hole_frac_k": float(np.max(hole_per_k)),
        **{f"hole_frac_k{kk:02d}": hole_per_k[kk] for kk in range(cov.shape[0])},
    })

    # ---- Save ----
    fh.save_single_channel_ome_tiff(raw_mem_zyx, str(DIRS["raw_moving_mem"]), frame_idx=i, label="F260517_raw_mem")
    fh.save_single_channel_ome_tiff(raw_sparse_zyx, str(DIRS["raw_moving_sparseCell"]), frame_idx=i, label="F260517_raw_sparseCell")
    fh.save_single_channel_ome_tiff(proj_mem_zyx, str(DIRS["projected_mem"]), frame_idx=i, label="F260517_projected_mem")
    fh.save_single_channel_ome_tiff(proj_sparse_zyx, str(DIRS["projected_sparseCell"]), frame_idx=i, label="F260517_projected_sparseCell")

    # ---- Metrics: mov vs mem_mapped ----
    mask_mov = option["mask_mov"]
    if hasattr(mask_mov, "get"): mask_mov = mask_mov.get()
    mask_mov_zyx = np.asarray(mask_mov, dtype=bool).transpose(2, 1, 0)

    mem_metrics = compute_frame_metrics(raw_mem_zyx, mem_mapped_zyx, mask_mov_zyx)

    # Sample sparse-cell reference at phase_new for mem_mapped comparison
    from utils.calFlowCrossResolution import generate_continuous_H_gpu as genH, apply_H_to_matrix_gpu as applyH
    H_sp = genH(cp.asarray(ref_sparse_raw.transpose(2, 1, 0), dtype=cp.float32), zRatio=1)
    sparse_mapped_xyk = applyH(cp.asarray(phase_new, dtype=cp.float32), H_sp)
    if hasattr(sparse_mapped_xyk, "get"): sparse_mapped_xyk = sparse_mapped_xyk.get()
    sparse_mapped_zyx = np.asarray(sparse_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)

    sparse_metrics = compute_sparse_metrics(raw_sparse_zyx, sparse_mapped_zyx, mask_mov_zyx)

    elapsed = time.time() - frame_start

    error_mem.append({
        "frame": i, "ref_update_id": ref_update_id,
        "MAE": mem_metrics["MAE"], "nMAE": mem_metrics["nMAE"],
        "NCC": mem_metrics["NCC"], "edge": mem_metrics["edge"],
        "hole_frac": hole_frac, "elapsed_s": elapsed,
    })
    error_sparse.append({
        "frame": i, "ref_update_id": ref_update_id,
        "MAE": sparse_metrics["MAE"], "nMAE": sparse_metrics["nMAE"],
        "NN": sparse_metrics["NN"],
        "recall": sparse_metrics["recall"], "precision": sparse_metrics["precision"],
        "hole_frac": hole_frac, "elapsed_s": elapsed,
    })

    print(f"[Frame {i:03d}/{T-1:03d}] "
          f"mem_MAE={mem_metrics['MAE']:.1f}  mem_NCC={mem_metrics['NCC']:.4f}  "
          f"mem_nMAE={mem_metrics['nMAE']:.4f}  "
          f"sparse_MAE={sparse_metrics['MAE']:.1f}  "
          f"sparse_NN={sparse_metrics['NN']:.2f}px  "
          f"sparse_R={sparse_metrics['recall']:.3f}  "
          f"holes={hole_frac*100:.1f}%  {elapsed:.1f}s")

    # ---- Temporal init ----
    option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

    # ---- Ref update ----
    frames_since_ref_update += 1
    if frames_since_ref_update >= ref_update_every:
        calib_frames = sorted(registered_cache.keys())[-5:]
        ref_update_id += 1

        stacks = [mov_mem_all[fi].astype(np.float32, copy=False) for fi in calib_frames]
        if len(stacks) > 0:
            target = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32)
            ref_source = ref_mem_raw[z_idx].astype(np.float32, copy=False)
            _, new_tgt_q, _ = prep.learn_quantile_mapping(
                source=ref_source, target=target, percentiles=percentiles)
            ref_mem_adj = prep.apply_quantile_mapping(
                ref_mem_raw, src_q_fixed, new_tgt_q,
            ).transpose(2, 1, 0).astype(np.float32, copy=False)
            print(f"\n  >>> Ref Update #{ref_update_id} — raw frames {calib_frames}  "
                  f"tgt_q[0]={new_tgt_q[0]:.1f}\n")

        frames_since_ref_update = 0

total_elapsed = time.time() - total_start
print(f"\n  Done in {total_elapsed:.1f}s ({total_elapsed/T:.1f}s/frame)")

# ===========================================================================
# 6. Save CSVs
# ===========================================================================
print("\n[6/7] Saving CSVs ...")
df_mem = pd.DataFrame(error_mem)
df_sparse = pd.DataFrame(error_sparse)
df_holes = pd.DataFrame(hole_records)

df_mem.to_csv(str(DIRS["diagnostics"] / "errors_membrane.csv"), index=False)
df_sparse.to_csv(str(DIRS["diagnostics"] / "errors_sparse.csv"), index=False)
df_holes.to_csv(str(DIRS["diagnostics"] / "hole_summary.csv"), index=False)

# ===========================================================================
# 7. Summary
# ===========================================================================
print("\n[7/7] Summary — Global means")
print("=" * 60)
for label, df in [("Membrane", df_mem), ("Sparse", df_sparse)]:
    print(f"\n  {label}:")
    for col in df.columns:
        if col in ("frame", "ref_update_id", "elapsed_s"): continue
        vals = df[col].dropna()
        if len(vals):
            print(f"    {col:16s}: mean={vals.mean():.4f}  std={vals.std():.4f}")

print(f"\n  Hole fraction (global): {df_holes['hole_frac_global'].mean():.4f}")
print(f"\nDone. Output: {BASE_OUT}")
