#!/usr/bin/env python3
"""
Test projection on a single frame (no f260517_helpers dependency).

Usage:
  python test_projection_single_frame.py --frame 40
  python test_projection_single_frame.py --frame 200  # auto-selects ref_adj_NN.npy

Pipeline:
  1. Load mov tiff + select ref_adj snapshot by frame_idx (frame//40)
  2. Register that frame with getMotion_v2 (same params as PP baseline)
  3. Project coords with project_coords_to_fixed_planes_gpu (from calFlowCrossResolution)
  4. Output: projected membrane + sparse images (npy) + hole metrics (stdout)
"""

import os, sys, time, argparse, json
from pathlib import Path

import numpy as np
import cupy as cp

# ---------------------------------------------------------------------------
# Path setup (edit these for your machine)
# ---------------------------------------------------------------------------
CODE_DIR = Path("/home/cyf/wbi/Virginia/code")
COARSEFLOW_DIR = CODE_DIR / "CoarseFlow"
WBI_SRC = CODE_DIR / "wbi_0123/wholistic_registration/src"
PKG_DIR = WBI_SRC / "wholistic_registration"
TEST_DIR = PKG_DIR / "tests"
for p in [CODE_DIR, COARSEFLOW_DIR, WBI_SRC, PKG_DIR, TEST_DIR]:
    sys.path.insert(0, str(p))

from utils import IO, calFlowCrossResolution, mask, preprocess as prep
from utils.calFlowCrossResolution import (
    project_coords_to_fixed_planes_gpu,
    generate_continuous_H_gpu as genH,
    apply_H_to_matrix_gpu as applyH,
)
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# Config — user-editable
# ---------------------------------------------------------------------------
MOV_PATH = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
REF_PATH = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_anat_00003_TZCYX.ome.tiff"
REF_SNAP_DIR = Path("/mnt/data21T_2/cyf/f260517/PP_mapped_v3_layer3/ref_snapshots")
OUT_DIR = Path("/home/cyf/wbi/Virginia/exp/plots/projection_test")
CUPY_DEVICE = 1

LAYER = 3
THRES_FACTOR = 5.0
MASK_RANGE = [5.0, 4000.0]
SMOOTH_PENALTY_RAW = 0.03

OPTION_BASE = {
    "r": 5, "layer": LAYER, "iter": 10, "movRange": 5.0,
    "tol": 1e-6, "zRatio_HR": 1, "wrong_region_enable": False,
}

# Projection params
Z_WINDOW = 3.0
SURFACE_UPSAMPLE = 2
XY_EXTRA_RADIUS = 0
DOWN_SAMPLE_XY = 1

# Fixed z_init — computed once from frame 0 (deterministic), hardcoded.
# moving slice k maps to reference z = 20 + 10*k  (z_idx = 20..210, step 10)
Z_INIT = np.array(
    [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
     110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
     190.0, 200.0, 210.0],
    dtype=np.float32,
)
Z_IDX = np.rint(Z_INIT).astype(np.int32)  # 20, 30, ..., 210

# ---------------------------------------------------------------------------
# Self-contained helpers (no f260517_helpers)
# ---------------------------------------------------------------------------
def upsample_phase_xy_for_supersurface(phase_new, upsample_factor=2):
    """Bilinearly upsample phase_new (X,Y,K,3) -> (X*f, Y*f, K, 3) in XY."""
    phase = np.asarray(phase_new, dtype=np.float32)
    if phase.ndim != 4 or phase.shape[-1] != 3:
        raise ValueError(f"phase_new should be (X,Y,K,3), got {phase.shape}")
    factor = int(upsample_factor)
    if factor == 1:
        return phase
    X, Y, K_local, C = phase.shape
    Xup, Yup = X * factor, Y * factor
    coords = np.zeros((2, Xup * Yup), dtype=np.float64)
    x_fine = np.clip((np.arange(Xup, dtype=np.float64) + 0.5) / factor - 0.5, 0, X - 1)
    y_fine = np.clip((np.arange(Yup, dtype=np.float64) + 0.5) / factor - 0.5, 0, Y - 1)
    XX, YY = np.meshgrid(x_fine, y_fine, indexing="ij")
    coords[0] = XX.ravel()
    coords[1] = YY.ravel()
    phase_up = np.empty((Xup, Yup, K_local, C), dtype=np.float32)
    for k_idx in range(K_local):
        for c_idx in range(C):
            phase_up[:, :, k_idx, c_idx] = map_coordinates(
                phase[:, :, k_idx, c_idx], coords, order=1, mode="nearest"
            ).reshape(Xup, Yup)
    return phase_up


def register_frame(mov_mem_xyk, ref_mem_adj, option, coords_xyz):
    """Run getMotion_v2 for one frame. Returns (phase_new, motion_current, mem_mapped)."""
    option = dict(option)
    option["mask_ref"] = mask.getMask(ref_mem_adj, THRES_FACTOR)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], MASK_RANGE)
    Pnlt = prep.getSmPnltNormFctr(ref_mem_adj, option)
    option["smoothPenalty"] = Pnlt * SMOOTH_PENALTY_RAW
    option["mask_mov"] = mask.getMask(mov_mem_xyk, THRES_FACTOR)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], MASK_RANGE)
    option["phase"] = coords_xyz.copy()
    option.pop("motion", None)  # single-frame test: identity init

    pn, mc, mm = calFlowCrossResolution.getMotion_v2(mov_mem_xyk, ref_mem_adj, option, verbose=False)
    if hasattr(pn, "get"): pn = pn.get()
    if hasattr(mc, "get"): mc = mc.get()
    if hasattr(mm, "get"): mm = mm.get()
    return np.asarray(pn, np.float32), np.asarray(mc, np.float32), np.asarray(mm, np.float32)


def classify_holes(coverage, phase_up, target_z, ref_shape, xy_radius=2.0, z_window=3.0):
    """Classify hole pixels into xy_gap / z_gap / no_coverage (notebook Part 2b)."""
    from scipy.spatial import cKDTree

    Nplanes, Hy, Hx = coverage.shape
    Xref, Yref, Zref = ref_shape

    coords_flat = phase_up.reshape(-1, 3).astype(np.float32)
    xref_flat, yref_flat, zref_flat = coords_flat[:, 0], coords_flat[:, 1], coords_flat[:, 2]
    in_bounds = (
        (xref_flat >= 0) & (xref_flat <= Xref - 1) &
        (yref_flat >= 0) & (yref_flat <= Yref - 1) &
        (zref_flat >= 0) & (zref_flat <= Zref - 1) &
        np.isfinite(xref_flat) & np.isfinite(yref_flat) & np.isfinite(zref_flat)
    )
    xo_valid = xref_flat[in_bounds]
    yo_valid = yref_flat[in_bounds]
    zref_valid = zref_flat[in_bounds]

    tree = cKDTree(np.column_stack([xo_valid, yo_valid]))

    hole_stats = []
    hole_detail = []

    for m in range(Nplanes):
        tz = float(target_z[m])
        hole_mask = coverage[m] == 0
        n_holes = int(hole_mask.sum())

        if n_holes == 0:
            hole_stats.append({"plane": m, "target_z": tz, "n_holes": 0,
                               "xy_gap": 0, "z_gap": 0, "no_coverage": 0})
            continue

        hy, hx = np.where(hole_mask)
        hole_coords = np.column_stack([hx, hy]).astype(np.float32)
        dists, idxs = tree.query(hole_coords, k=1)

        n_xy = 0; n_z = 0; n_nocov = 0
        for hi in range(len(hole_coords)):
            if dists[hi] > xy_radius:
                n_nocov += 1
                hole_detail.append({"plane": m, "hole_x": hx[hi], "hole_y": hy[hi],
                                    "category": "no_coverage"})
            else:
                z_near = zref_valid[idxs[hi]]
                if abs(z_near - tz) <= z_window:
                    n_xy += 1
                    hole_detail.append({"plane": m, "hole_x": hx[hi], "hole_y": hy[hi],
                                        "category": "xy_gap"})
                else:
                    n_z += 1
                    hole_detail.append({"plane": m, "hole_x": hx[hi], "hole_y": hy[hi],
                                        "category": "z_gap"})

        hole_stats.append({"plane": m, "target_z": tz, "n_holes": n_holes,
                           "xy_gap": n_xy, "z_gap": n_z, "no_coverage": n_nocov})

    return hole_stats, hole_detail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-frame projection test")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to register+project")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (overrides OUT_DIR)")
    args = parser.parse_args()
    frame_idx = args.frame
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if frame_idx < 0 or frame_idx >= 200:
        print(f"Error: frame must be in [0, 200), got {frame_idx}")
        sys.exit(1)

    cp.cuda.Device(CUPY_DEVICE).use()
    print(f"CUDA device: {CUPY_DEVICE}")
    print(f"Frame: {frame_idx}")

    # --- 1. Load data ---
    print("\n[1/5] Loading data...")
    F260517_mov, _ = IO.readTiff(MOV_PATH)
    F260517_ref, _ = IO.readTiff(REF_PATH)

    ref_mem_raw = F260517_ref[90:310, 1, :, :].astype(np.float32)   # (D,Y,X)
    ref_sp_raw  = F260517_ref[90:310, 0, :, :].astype(np.float32)   # (D,Y,X)
    mov_mem_all = F260517_mov[:, :, 1, :, :].astype(np.float32)      # (T,K,Y,X)
    mov_sp_all  = F260517_mov[:, :, 0, :, :].astype(np.float32)      # (T,K,Y,X)

    D, Y, X = ref_mem_raw.shape
    K, T = mov_mem_all.shape[1], mov_mem_all.shape[0]
    print(f"  ref=(D={D},Y={Y},X={X})  mov=(T={T},K={K},Y={Y},X={X})")

    # --- 2. Fixed z_init (hardcoded) ---
    z_init = Z_INIT.copy()
    z_idx = Z_IDX.copy()
    print(f"\n[2/5] z_init (hardcoded): {z_init.tolist()}")

    # Coordinate grid
    x_c = np.arange(X, dtype=np.float32)
    y_c = np.arange(Y, dtype=np.float32)
    k_c = np.arange(K, dtype=np.int32)
    Xg, Yg, Kg = np.meshgrid(x_c, y_c, k_c, indexing="ij")
    coords_xyz = np.empty((X, Y, K, 3), np.float32)
    coords_xyz[..., 0] = Xg
    coords_xyz[..., 1] = Yg
    coords_xyz[..., 2] = z_init[Kg]

    # --- 3. Auto-select ref snapshot ---
    ref_id = frame_idx // 40
    ref_mem_adj = np.load(str(REF_SNAP_DIR / f"ref_adj_{ref_id:02d}.npy")).astype(np.float32)  # (X,Y,D)
    print(f"\n[3/5] Selected ref_adj_{ref_id:02d}.npy (frame {frame_idx} in [{ref_id*40}, {(ref_id+1)*40}))")

    # Sparse ref: same coords as membrane, sample from raw sparse ref (no recalibration)
    # The sparse channel is projected with the SAME phase field as membrane.
    ref_sp_adj = ref_sp_raw.transpose(2, 1, 0).astype(np.float32, copy=False)  # (X,Y,D)

    # --- 4. Register frame ---
    print(f"\n[4/5] Registering frame {frame_idx}...")
    mov_mem_zyx = mov_mem_all[frame_idx]
    mov_mem_xyk = mov_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)

    option = dict(OPTION_BASE)
    t0 = time.time()
    phase_new, motion_current, mem_mapped = register_frame(mov_mem_xyk, ref_mem_adj, option, coords_xyz)
    reg_time = time.time() - t0
    print(f"  Registered in {reg_time:.1f}s  phase shape={phase_new.shape}")

    # --- 5. Project both channels ---
    print(f"\n[5/5] Projecting...")
    fixed_target_z = z_init.copy()  # target planes = z_init

    phase_up = upsample_phase_xy_for_supersurface(phase_new, upsample_factor=SURFACE_UPSAMPLE)
    print(f"  Upsampled phase: {phase_up.shape}")

    results = {}
    for ch, ref_vol in [("membrane", ref_mem_adj), ("sparse", ref_sp_adj)]:
        print(f"  Projecting {ch}...")
        proj = project_coords_to_fixed_planes_gpu(
            coords_ref_xyk_xyz=phase_up,
            ref_volume=ref_vol,
            target_z_planes=fixed_target_z,
            values_xyk=None,           # sample from ref_volume
            ref_volume_order="xyz",
            z_window=Z_WINDOW,
            downsample_xy=DOWN_SAMPLE_XY,
            fill_value=0.0,
            return_numpy=True,
            output_order="zyx",
            xy_splat_mode="subpixel_footprint",
            xy_extra_radius=XY_EXTRA_RADIUS,
        )
        if hasattr(proj, "get"):
            proj = proj.get()
        proj = np.asarray(proj, dtype=np.float32)
        results[ch] = proj
        print(f"    {ch}: {proj.shape}")

    # Save projected images
    for ch, proj in results.items():
        np.save(str(out_dir / f"projected_{ch}_frame{frame_idx:03d}.npy"), proj)
    print(f"\n  Saved projections to {out_dir}")

    # --- Coverage + hole classification (membrane ref, same as notebook) ---
    print("\n" + "=" * 70)
    print(f"Frame {frame_idx} — Hole Classification Summary")
    print("=" * 70)

    coverage = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_up,
        ref_volume=ref_mem_adj,
        target_z_planes=fixed_target_z,
        values_xyk=np.ones(phase_up.shape[:-1], dtype=np.float32),
        ref_volume_order="xyz",
        z_window=Z_WINDOW,
        downsample_xy=DOWN_SAMPLE_XY,
        fill_value=0.0,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=XY_EXTRA_RADIUS,
    )
    if hasattr(coverage, "get"):
        coverage = coverage.get()
    coverage = np.asarray(coverage, dtype=np.float32)

    hole_frac = float(np.mean(coverage == 0))
    print(f"Coverage map: {coverage.shape}")
    print(f"Total output pixels: {coverage.size:,}")
    print(f"Hole fraction (fill_value pixels): {hole_frac:.4f} ({hole_frac*100:.2f}%)")

    # Classify holes
    hole_stats, hole_detail = classify_holes(
        coverage, phase_up, fixed_target_z,
        ref_shape=ref_mem_adj.shape, xy_radius=2.0, z_window=Z_WINDOW)

    total_holes = sum(s["n_holes"] for s in hole_stats)
    total_xy = sum(s["xy_gap"] for s in hole_stats)
    total_z = sum(s["z_gap"] for s in hole_stats)
    total_nocov = sum(s["no_coverage"] for s in hole_stats)

    print(f"\n  Total holes:           {total_holes:>12,}  ({100*total_holes/coverage.size:.2f}%)")
    print(f"  - XY gap:              {total_xy:>12,}  ({100*total_xy/max(total_holes,1):.1f}% of holes)")
    print(f"  - Z out-of-window:     {total_z:>12,}  ({100*total_z/max(total_holes,1):.1f}% of holes)")
    print(f"  - No coverage:         {total_nocov:>12,}  ({100*total_nocov/max(total_holes,1):.1f}% of holes)")

    if total_z > total_xy:
        print("  → VERDICT: Z gaps dominate → widen z_window or add more target planes")
    elif total_xy > total_z:
        print("  → VERDICT: XY gaps dominate → increase supersurface upsampling")
    else:
        print("  → VERDICT: Mixed — both XY and Z contribute similarly")

    # --- Per-plane breakdown ---
    print("\n  Per-plane breakdown:")
    for s in hole_stats:
        n = s["n_holes"]
        if n == 0:
            print(f"    plane {s['plane']:2d}  z={s['target_z']:.0f}:  NO HOLES")
        else:
            print(f"    plane {s['plane']:2d}  z={s['target_z']:.0f}:  {n:>6} holes  "
                  f"(xy={s['xy_gap']}, z={s['z_gap']}, nocov={s['no_coverage']})")

    # --- Supersurface sweep ---
    print("\n  Supersurface factor vs hole fraction:")
    for test_factor in [1, 2, 4]:
        p_test = upsample_phase_xy_for_supersurface(phase_new, upsample_factor=test_factor)
        cov_test = project_coords_to_fixed_planes_gpu(
            coords_ref_xyk_xyz=p_test,
            ref_volume=ref_mem_adj,
            target_z_planes=fixed_target_z,
            values_xyk=np.ones(p_test.shape[:-1], dtype=np.float32),
            ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
            fill_value=0.0, return_numpy=True, output_order="zyx",
            xy_splat_mode="subpixel_footprint", xy_extra_radius=0,
        )
        if hasattr(cov_test, "get"):
            cov_test = cov_test.get()
        cov_test = np.asarray(cov_test, dtype=np.float32)
        hf = float(np.mean(cov_test == 0))
        print(f"    supersurface={test_factor}×  →  hole fraction = {hf:.4f} ({hf*100:.2f}%)")

    # Save config
    config = {
        "frame": frame_idx, "ref_id": ref_id,
        "z_init": z_init.tolist(),
        "z_window": Z_WINDOW, "supersurface": SURFACE_UPSAMPLE,
        "out_dir": str(out_dir), "reg_time_s": reg_time,
        "hole_frac": hole_frac,
        "holes": {"total": total_holes, "xy_gap": total_xy, "z_gap": total_z, "no_coverage": total_nocov},
    }
    with open(str(out_dir / f"config_frame{frame_idx:03d}.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Config saved to {out_dir / f'config_frame{frame_idx:03d}.json'}")
    print("\nDiagnosis complete.")


if __name__ == "__main__":
    main()
