#!/usr/bin/env python3
"""
F260517 Registration — Flat Z-Plane Projection Pipeline  (2025-06-25)
QC variant of run_F260517_0625.py: identical pipeline, plus six
independently toggleable diagnostic-save additions (env vars, default ON):
alignment-QC curves, per-frame/reference masks, motion-field arrays (raw
copies plus a flat-directory copy matched to compare_projectors.py's input
format), projection-state parameters needed to redo the projection call,
and per-frame coverage hole maps.

Key settings:
  - smoothPenalty_raw = 0.03
  - mask_ref & smoothPenalty: computed ONCE at init
  - Ref update every 40 frames, calibration target = RAW moving frames
  - Projection: GPU splatting onto fixed_target_z (from warmup 0-4)
  - Metrics: mov vs mem_mapped  (warped reference, same as v3/v4)

Output (under f260517_0625_qc/ by default, see BASE_OUT below):
  raw_moving_mem/          raw moving membrane per frame
  raw_moving_sparseCell/   raw moving sparse-cell per frame
  projected_mem/           z-plane-projected membrane
  projected_sparseCell/    z-plane-projected sparse-cell
  refspace_mem/            per-frame moving membrane scattered into the full
                            (REF_Z1-REF_Z0, 1500, 630) reference grid, no
                            z-window gate                      [SAVE_REF_SPACE_VOLUME]
  refspace_sparseCell/     per-frame moving sparse-cell scattered into the
                            full reference grid, no z-window gate
                            [SAVE_REF_SPACE_VOLUME]
  ref_surface_mem/         the membrane reference sampled onto the same K
                            fixed target planes as projected_mem/, one volume
                            per run, not per frame                    [SAVE_REF_SURFACE]
  ref_surface_sparseCell/  the sparse-cell reference sampled onto the same K
                            fixed target planes as projected_sparseCell/, one
                            volume per run, not per frame             [SAVE_REF_SURFACE]
  diagnostics/             CSVs (errors_membrane, errors_sparse, hole_summary,
                            refspace_summary)
    alignment_qc/          target_z_offset_per_plane.npy, zinit_match_curve.png,
                            zinit_zncc_heatmap.png       [SAVE_ALIGNMENT_QC]
    masks_mov/              mask_mov_{i:06d}.npz          [SAVE_MASKS]
    mask_ref.npz                                          [SAVE_MASKS]
    ref_shape.npy, fixed_target_z.npy,
    projection_params.json                                [SAVE_PROJECTION_STATE]
    phase_new_f{i}.npy, motion_current_f{i}.npy           [SAVE_MOTION_FIELD]
    mov_mem_f{i}.npy                                      [SAVE_COMPARE_INPUTS, default off]
    coverage/                no_coverage_{i:06d}.npz       [SAVE_COVERAGE_MAP]
    refspace_summary.csv                                  [SAVE_REF_SPACE_VOLUME]
  Each array above (phase_new, motion_current, mov_mem, per frame) is
  written exactly once, flat under diagnostics/; no motion_field/
  subdirectory and no duplicate copy exist.
"""

import json, os, sys, time
from pathlib import Path

import cupy as cp
import cupyx
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label as label_ndi, sobel
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# GPU + paths
# ---------------------------------------------------------------------------
# Device index is configurable so the script runs on single-GPU nodes, where
# device 1 does not exist. Default 1 keeps the original behaviour.
GPU_DEVICE = int(os.environ.get("GPU_DEVICE", "1"))
cp.cuda.Device(GPU_DEVICE).use()

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
# Original paths on the collaborator's machine, kept for reference:
# F260517_mov_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_exp_00001_TZCYX.ome.tiff"
# F260517_ref_path = "/home/cyf/wbi/Virginia/raw_data/f260517/260517_anat_00003_TZCYX.ome.tiff"
# BASE_OUT = Path("/home/cyf/wbi/Virginia/registrated_data/f260517/f260517_0625")

# Janelia paths.
F260517_DATA_DIR = "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b"

# The full moving stack (15.1 GB on disk); only N_LOAD timepoints are read from
# it, so RAM does not scale with file size. test_F260517_v2.py uses this file.
F260517_mov_path = F260517_DATA_DIR + "/260517_exp_00001_TZCYX.ome.tiff"
# 8-timepoint truncation (605 MB), if the full file is ever unavailable:
# F260517_mov_path = F260517_DATA_DIR + "/260517_exp_00001_TZCYX_first8.ome.tiff"
F260517_ref_path = F260517_DATA_DIR + "/260517_anat_00003_TZCYX.ome.tiff"

# QC default output dir differs from run_F260517_0625.py's repro_3frame_out
# so this script's writes can never land in that concurrent process's tree
# or in the delivered temp_result_by_Yunfeng tree.
BASE_OUT = Path(os.environ.get(
    "F260517_OUT_DIR",
    F260517_DATA_DIR + "/registration_out/f260517_0625_qc",
))
DIRS = {
    "raw_moving_mem":        BASE_OUT / "raw_moving_mem",
    "raw_moving_sparseCell": BASE_OUT / "raw_moving_sparseCell",
    "projected_mem":         BASE_OUT / "projected_mem",
    "projected_sparseCell":  BASE_OUT / "projected_sparseCell",
    "refspace_mem":          BASE_OUT / "refspace_mem",
    "refspace_sparseCell":   BASE_OUT / "refspace_sparseCell",
    "ref_surface_mem":        BASE_OUT / "ref_surface_mem",
    "ref_surface_sparseCell": BASE_OUT / "ref_surface_sparseCell",
    "diagnostics":           BASE_OUT / "diagnostics",
    # QC additions only (new keys; existing keys/names above are untouched).
    "alignment_qc":          BASE_OUT / "diagnostics" / "alignment_qc",
    "masks_mov":              BASE_OUT / "diagnostics" / "masks_mov",
    "coverage":                BASE_OUT / "diagnostics" / "coverage",
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
# Fill value the COVERAGE projection call (Section 5, cov=...) passes, distinct
# from FILL_VALUE above which the two DATA projection calls pass. The hole
# statistic np.mean(cov == 0) depends on this value being 0.0, not FILL_VALUE.
COVERAGE_FILL_VALUE = 0.0
WARMUP_FRAMES = [0, 1, 2, 3, 4]

# Number of frames the forward loop registers. 0 or unset means every frame in
# the moving file. Named to match N_FRAMES_LIMIT in test_F260517_v2.py.
# The warmup is not limited by N_FRAMES_LIMIT: it always uses WARMUP_FRAMES,
# because fixed_target_z is the median over those frames, so a shorter warmup
# would change the target planes the projection writes onto.
N_FRAMES_LIMIT = int(os.environ.get("N_FRAMES_LIMIT", "0")) or None

# Timepoints to read off disk: enough for the warmup and the forward loop.
N_LOAD = None if N_FRAMES_LIMIT is None else max(N_FRAMES_LIMIT, max(WARMUP_FRAMES) + 1)

# Half-open z crop [REF_Z0, REF_Z1) applied to the (Z,C,Y,X) reference stack.
# Every downstream reference-space coordinate is expressed in this cropped
# frame: reference plane p in the source file is plane p - REF_Z0 here, and the
# cropped volume has REF_Z1 - REF_Z0 planes. Named once so the two channel
# slices below and the reference_crop entry in projection_params.json cannot
# drift apart.
REF_Z0 = int(os.environ.get("REF_Z0", "165"))
REF_Z1 = int(os.environ.get("REF_Z1", "243"))
REF_CHANNEL_MEM = 1
REF_CHANNEL_SPARSE = 0

# Which moving z slices to register, as indices into the moving file's z axis
# (axis 1 of the (T,Z,C,Y,X) stack). K is len(MOV_SLICES); nothing downstream
# reads a literal slice count, every consumer takes K from an array shape.
MOV_SLICES = [int(s) for s in os.environ.get("MOV_SLICES", "8,9,10,11").split(",")
              if s.strip() != ""]
if len(MOV_SLICES) == 0:
    raise ValueError("MOV_SLICES selected no slices")

# Reference z plane (in the cropped [REF_Z0, REF_Z1) frame) each entry of
# MOV_SLICES starts at. Supplied rather than searched for:
# FindInitZ_stack_global_fixed_spacing slides the whole rigidly-spaced comb of
# K slices through the reference and scores each integer offset, so with K=4
# slices in a 78-plane crop the summed-ZNCC curve is scored from 4 planes and
# the search is not constrained enough to trust. Set Z_INIT_PLANES="" to run
# the search instead, in which case SAVE_ALIGNMENT_QC's two PNGs are produced
# (they read the search's debug output) and are skipped otherwise.
_z_init_env = os.environ.get("Z_INIT_PLANES", "25,35,45,55")
Z_INIT_PLANES = [float(s) for s in _z_init_env.split(",") if s.strip() != ""]
if Z_INIT_PLANES and len(Z_INIT_PLANES) != len(MOV_SLICES):
    raise ValueError(
        f"Z_INIT_PLANES has {len(Z_INIT_PLANES)} entries but MOV_SLICES has "
        f"{len(MOV_SLICES)}; one starting reference plane per moving slice is required")

percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]

# ---------------------------------------------------------------------------
# QC instrumentation (this script only; NOT present in run_F260517_0625.py)
# ---------------------------------------------------------------------------
# Six additions, each independently toggleable via an env var, default ON:
#   1. SAVE_ALIGNMENT_QC     -- target_z_offset_per_plane.npy +
#                                zinit_match_curve.png + zinit_zncc_heatmap.png
#   2. SAVE_MASKS             -- mask_mov (per frame) and mask_ref (once)
#   3. SAVE_MOTION_FIELD      -- phase_new_f{i}.npy, motion_current_f{i}.npy,
#                                written once each into the flat diagnostics/
#                                directory (unpadded frame index, matching the
#                                filenames make_phase_new.py / compare_projectors.py
#                                expect on feat/inverse-gather-projector), stride
#                                via PHASE_SAVE_STRIDE, frame 0 and the final
#                                processed frame always included
#   4. SAVE_PROJECTION_STATE  -- ref_shape.npy, fixed_target_z.npy,
#                                projection_params.json (state needed to redo
#                                the projection call later). This script's own
#                                choice of name; no env var for this addition
#                                is specified elsewhere.
#   5. SAVE_COMPARE_INPUTS    -- mov_mem_f{i}.npy in the flat diagnostics/
#                                directory: the one array compare_projectors.py
#                                needs that SAVE_MOTION_FIELD does not already
#                                write (phase_new_f{i}.npy comes from
#                                SAVE_MOTION_FIELD). Same frame selection as
#                                SAVE_MOTION_FIELD.
#   6. SAVE_COVERAGE_MAP      -- diagnostics/coverage/no_coverage_{i:06d}.npz,
#                                a packed boolean map of where the per-frame
#                                projection coverage array cov is zero.
#   7. SAVE_REF_SPACE_VOLUME  -- refspace_mem/, refspace_sparseCell/: per-frame
#                                moving intensity scattered (trilinear weights,
#                                NO z-window gate) into the full
#                                (REF_Z1-REF_Z0, 1500, 630)
#                                reference grid, plus diagnostics/refspace_summary.csv
#                                (written every run). Stride via
#                                REF_SPACE_SAVE_STRIDE, frame 0 and the final
#                                processed frame always included, same
#                                _frame_selected_for_stride predicate as
#                                SAVE_MOTION_FIELD. The cropped reference volume
#                                itself is NOT written: it is
#                                F260517_ref_path[REF_Z0:REF_Z1, REF_CHANNEL_MEM]
#                                already on disk, and projection_params.json
#                                records that path and crop under
#                                reference_source_path / reference_crop instead
#                                of copying the volume.
#   8. SAVE_REF_SURFACE       -- ref_surface_mem/, ref_surface_sparseCell/: the
#                                reference sampled onto the same fixed_target_z
#                                planes as projected_mem/projected_sparseCell,
#                                through the same
#                                project_coords_to_fixed_planes_gpu call with
#                                values_xyk=None (that call's documented mode in
#                                which values are sampled from ref_volume at the
#                                supplied coordinates), on the frame-0
#                                phase_for_proj geometry. One volume per channel
#                                per run, not per frame.
SAVE_ALIGNMENT_QC     = os.environ.get("SAVE_ALIGNMENT_QC", "1") == "1"
SAVE_MASKS            = os.environ.get("SAVE_MASKS", "1") == "1"
SAVE_MOTION_FIELD     = os.environ.get("SAVE_MOTION_FIELD", "1") == "1"
PHASE_SAVE_STRIDE     = int(os.environ.get("PHASE_SAVE_STRIDE", "1"))
SAVE_PROJECTION_STATE = os.environ.get("SAVE_PROJECTION_STATE", "1") == "1"
# Default off: mov_mem_f{i}.npy holds the same array raw_moving_mem/ already
# stores as a TIFF, so the default run does not pay for the duplicate. Set
# SAVE_COMPARE_INPUTS=1 when compare_projectors.py is going to be run.
SAVE_COMPARE_INPUTS   = os.environ.get("SAVE_COMPARE_INPUTS", "0") == "1"
SAVE_COVERAGE_MAP     = os.environ.get("SAVE_COVERAGE_MAP", "1") == "1"
SAVE_REF_SPACE_VOLUME = os.environ.get("SAVE_REF_SPACE_VOLUME", "1") == "1"
REF_SPACE_SAVE_STRIDE = int(os.environ.get("REF_SPACE_SAVE_STRIDE", "1"))
SAVE_REF_SURFACE      = os.environ.get("SAVE_REF_SURFACE", "1") == "1"

# Bytes for one frame's phase_new and one frame's motion_current, each of shape
# (Xmov, Ymov, K, 3) float32 plus a 128-byte .npy header. The formula below
# reproduces 226_800_128 bytes at (630,1500,20,3), which is the value
# os.path.getsize returned on a saved phase_new_f0.npy from the K=20 runs; the
# formula rather than that constant is used so the printed projection tracks K
# instead of restating a 20-slice number for a 4-slice run.
_NPY_HEADER_BYTES = 128


def _phase_or_motion_bytes(x_mov, y_mov, k):
    return int(x_mov) * int(y_mov) * int(k) * 3 * 4 + _NPY_HEADER_BYTES


def _frame_selected_for_stride(i, stride, final_idx):
    """True if frame i should be saved: on the stride, or frame 0, or the
    final processed frame (fix for the stride never guaranteeing the run's
    last frame is saved when stride > 1)."""
    return i == 0 or i == final_idx or i % stride == 0


def _save_mask_sparse_or_dense(path, mask_arr, label):
    """Save a boolean mask compactly: sparse (x, y, z) coordinates if <=20%
    of voxels are True (measured on this dataset: mask_mov is ~0.009%
    nonzero), else np.packbits of the dense boolean array, printing which
    label took the dense fallback. mask_arr's axis order is whatever the
    caller's mask carries (X,Y,Z for both mask_mov and mask_ref in this
    script, since both are built on mov_i/F260517_ref_mem_adj, which are
    already transposed to (X,Y,Z) before mask.getMask is called); that order
    is recorded in the saved file's axis_order field so a reader never has
    to infer it."""
    m = mask_arr.get() if hasattr(mask_arr, "get") else mask_arr
    m = np.asarray(m, dtype=bool)
    frac_nonzero = float(np.count_nonzero(m)) / m.size
    if frac_nonzero <= 0.20:
        xs, ys, zs = np.nonzero(m)
        np.savez_compressed(str(path), x=xs, y=ys, z=zs,
                             shape=np.array(m.shape, dtype=np.int64), dense=False,
                             axis_order="xyz")
    else:
        packed = np.packbits(m.reshape(-1))
        np.savez_compressed(str(path), packed=packed,
                             shape=np.array(m.shape, dtype=np.int64), dense=True,
                             axis_order="xyz")
        print(f"[QC] {label}: {frac_nonzero*100:.1f}% nonzero (not sparse) "
              f"-> saved dense (packbits) instead of coordinates")


# Tracks which mask_mov frame indices have already been saved, so the warmup
# loop and the forward loop (which reprocess the same indices whenever
# WARMUP_FRAMES is a subset of range(0, T) -- true here whenever T > 4, i.e.
# whenever N_FRAMES_LIMIT does not cap the forward loop below 5 frames;
# verified by reading both loops' bounds, not assumed) do not double-save.
_mask_mov_saved_frames = set()
# Same double-processing hazard as _mask_mov_saved_frames above, for the
# motion-field / compare-inputs saves and for the coverage-map save.
_motion_saved_frames = set()
_coverage_saved_frames = set()
_refspace_saved_frames = set()


def _save_mask_mov(i, mask_arr):
    if not SAVE_MASKS or i in _mask_mov_saved_frames:
        return
    _mask_mov_saved_frames.add(i)
    path = DIRS["masks_mov"] / f"mask_mov_{i:06d}.npz"
    _save_mask_sparse_or_dense(path, mask_arr, f"mask_mov frame {i}")


def _save_motion_field(i, phase_new_arr, motion_current_arr):
    """Save phase_new/motion_current for frame i, already pulled off the GPU
    by the caller (this reuses those numpy arrays, it does not recompute
    anything), flat into diagnostics/ as phase_new_f{i}.npy and
    motion_current_f{i}.npy (unpadded frame index, the form
    compare_projectors.py's phase_new_f{i}.npy expects). Each array is
    written exactly once here; no separate motion_field/ copy exists. Also
    writes the raw moving membrane frame mov_mem_all[i] to
    diagnostics/mov_mem_f{i}.npy, in the shape compare_projectors.py expects
    (K,Ymov,Xmov), when SAVE_COMPARE_INPUTS is on -- the one array
    SAVE_MOTION_FIELD does not already produce."""
    if not (SAVE_MOTION_FIELD or SAVE_COMPARE_INPUTS) or i in _motion_saved_frames:
        return
    if not _frame_selected_for_stride(i, PHASE_SAVE_STRIDE, T - 1):
        return
    _motion_saved_frames.add(i)
    phase_f32 = np.asarray(phase_new_arr, dtype=np.float32)
    motion_f32 = np.asarray(motion_current_arr, dtype=np.float32)
    if SAVE_MOTION_FIELD:
        np.save(str(DIRS["diagnostics"] / f"phase_new_f{i}.npy"), phase_f32)
        np.save(str(DIRS["diagnostics"] / f"motion_current_f{i}.npy"), motion_f32)
    if SAVE_COMPARE_INPUTS:
        np.save(str(DIRS["diagnostics"] / f"mov_mem_f{i}.npy"),
                mov_mem_all[i].astype(np.float32))


def _save_coverage_map(i, cov_arr, axis_order):
    """Save where the per-frame projection coverage array cov_arr is zero,
    packed to one bit per voxel via np.packbits, into diagnostics/coverage/.
    axis_order must be the same output_order string passed to the
    project_coords_to_fixed_planes_gpu call that produced cov_arr, so the
    saved axis order can never drift from the call that made the array."""
    if not SAVE_COVERAGE_MAP or i in _coverage_saved_frames:
        return
    _coverage_saved_frames.add(i)
    no_cov = np.asarray(cov_arr) == 0
    np.savez_compressed(
        str(DIRS["coverage"] / f"no_coverage_{i:06d}.npz"),
        packed=np.packbits(no_cov.reshape(-1)),
        shape=np.array(no_cov.shape, dtype=np.int64),
        axis_order=axis_order,
    )


def _scatter_trilinear_to_refspace(coords_xyz, values_xyk, ref_shape_zyx, eps=1e-6):
    """Scatter one frame's upsampled moving samples into the full reference
    grid via trilinear weights over the 2x2x2 neighbourhood of each sample's
    continuous reference coordinate. No z-window gate: every finite,
    in-bounds sample contributes (this is the property SAVE_REF_SPACE_VOLUME
    exists for, contrasting with the z-window-gated projected_mem/
    projected_sparseCell outputs above). coords_xyz: (Xup,Yup,K,3) with
    [...,0]=x_ref, [...,1]=y_ref, [...,2]=z_ref (same layout
    project_coords_to_fixed_planes_gpu documents for coords_ref_xyk_xyz).
    values_xyk: (Xup,Yup,K) moving intensity, already upsampled by the
    caller's fh.upsample_values_xy_for_supersurface call -- this function
    does not upsample or resample the input.
    Returns (out, occupied): out is (Zref,Yref,Xref) float32 with
    weighted-mean intensity where the total scattered weight exceeds eps,
    np.nan elsewhere; occupied is the boolean (Zref,Yref,Xref) array
    out == out (i.e. weight > eps) the caller uses for the per-plane
    occupancy count without a second NaN scan."""
    Zref, Yref, Xref = ref_shape_zyx
    coords = cp.asarray(coords_xyz, dtype=cp.float32)
    values = cp.asarray(values_xyk, dtype=cp.float32)
    x_ref, y_ref, z_ref = coords[..., 0], coords[..., 1], coords[..., 2]

    valid = (
        cp.isfinite(x_ref) & cp.isfinite(y_ref) & cp.isfinite(z_ref) & cp.isfinite(values)
        & (x_ref >= 0) & (x_ref <= Xref - 1)
        & (y_ref >= 0) & (y_ref <= Yref - 1)
        & (z_ref >= 0) & (z_ref <= Zref - 1)
    )
    valid_w = valid.astype(cp.float32)
    # Invalid samples get their coordinate/value zeroed (safe, in-bounds) and
    # their corner weights zeroed via valid_w below, rather than being
    # filtered out, so the 8-corner loop never needs a variable-length index.
    x_s = cp.where(valid, x_ref, cp.float32(0.0))
    y_s = cp.where(valid, y_ref, cp.float32(0.0))
    z_s = cp.where(valid, z_ref, cp.float32(0.0))
    val_s = cp.where(valid, values, cp.float32(0.0))
    del coords, values, x_ref, y_ref, z_ref, valid

    x0 = cp.floor(x_s).astype(cp.int32); x1 = cp.minimum(x0 + 1, Xref - 1)
    y0 = cp.floor(y_s).astype(cp.int32); y1 = cp.minimum(y0 + 1, Yref - 1)
    z0 = cp.floor(z_s).astype(cp.int32); z1 = cp.minimum(z0 + 1, Zref - 1)
    fx, fy, fz = x_s - x0, y_s - y0, z_s - z0
    del x_s, y_s, z_s

    sum_val = cp.zeros(Zref * Yref * Xref, dtype=cp.float32)
    sum_w = cp.zeros(Zref * Yref * Xref, dtype=cp.float32)
    val_flat = val_s.ravel()
    for zi, wz in ((z0, 1.0 - fz), (z1, fz)):
        for yi, wy in ((y0, 1.0 - fy), (y1, fy)):
            for xi, wx in ((x0, 1.0 - fx), (x1, fx)):
                w = (wx * wy * wz * valid_w).ravel()
                idx = ((zi * Yref + yi) * Xref + xi).ravel()
                cupyx.scatter_add(sum_val, idx, w * val_flat)
                cupyx.scatter_add(sum_w, idx, w)
    del x0, x1, y0, y1, z0, z1, fx, fy, fz, valid_w, val_s, val_flat

    sum_val = sum_val.reshape(Zref, Yref, Xref)
    sum_w = sum_w.reshape(Zref, Yref, Xref)
    occupied = sum_w > eps
    out = cp.where(occupied, sum_val / cp.maximum(sum_w, eps), cp.float32(np.nan))
    out_np = cp.asnumpy(out)
    occupied_np = cp.asnumpy(occupied)
    del sum_val, sum_w, occupied, out
    cp.get_default_memory_pool().free_all_blocks()
    return out_np, occupied_np


_ref_surface_saved = False


def _save_ref_surface(i, coords_xyk_xyz, ref_mem_volume_xyz, ref_sparse_volume_xyz):
    """Sample the reference onto the fixed_target_z planes and write one volume
    per channel for the whole run (frame-0 geometry, not per frame).

    The call below is the same project_coords_to_fixed_planes_gpu call the
    forward loop makes for proj_mem_zyx / proj_sparse_zyx -- same
    target_z_planes, ref_volume_order, z_window, downsample_xy, fill_value,
    output_order, xy_splat_mode and xy_extra_radius -- with values_xyk=None
    instead of the moving intensity. values_xyk=None is that function's
    documented first mode: it samples ref_volume at coords_ref_xyk_xyz via
    generate_continuous_H_gpu / apply_H_to_matrix_gpu and splats those samples,
    so the reference passes through the identical z-slab selection and XY
    footprint splatting as the moving data. No second projection path is
    introduced.

    coords_xyk_xyz is the caller's frame-0 phase_for_proj, so the reference is
    sampled at the same reference coordinates the frame-0 moving projection
    used."""
    global _ref_surface_saved
    if not SAVE_REF_SURFACE or _ref_surface_saved or i != 0:
        return
    _ref_surface_saved = True
    for label, ref_volume, out_dir in (
        ("mem", ref_mem_volume_xyz, DIRS["ref_surface_mem"]),
        ("sparseCell", ref_sparse_volume_xyz, DIRS["ref_surface_sparseCell"]),
    ):
        surf = project_coords_to_fixed_planes_gpu(
            coords_ref_xyk_xyz=coords_xyk_xyz, ref_volume=ref_volume,
            target_z_planes=fixed_target_z, values_xyk=None,
            ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
            fill_value=FILL_VALUE, return_numpy=True, output_order="zyx",
            xy_splat_mode="subpixel_footprint", xy_extra_radius=0)
        surf = np.asarray(surf, dtype=np.float32)
        fh.save_single_channel_ome_tiff(surf, str(out_dir), frame_idx=0,
                                         label=f"F260517_ref_surface_{label}")
        print(f"[QC] ref_surface {label}: shape={surf.shape} "
              f"min={float(np.min(surf)):.2f} max={float(np.max(surf)):.2f} "
              f"mean={float(np.mean(surf)):.2f} "
              f"frac_covered={float(np.mean(surf != FILL_VALUE)):.4f} "
              f"(pixels not left at fill_value={FILL_VALUE})")
        del surf
        cp.get_default_memory_pool().free_all_blocks()


def _save_refspace_volume(i, coords_xyz, mem_vals, sparse_vals, ref_shape_zyx):
    """For frame i, scatter both channels' already-upsampled moving samples
    into the full reference grid (_scatter_trilinear_to_refspace), write one
    OME-TIFF per channel, and return the per-frame occupancy row for
    diagnostics/refspace_summary.csv. Frees each channel's GPU accumulators
    before starting the next channel, so both channels' accumulators are
    never resident at once."""
    if not SAVE_REF_SPACE_VOLUME or i in _refspace_saved_frames:
        return None
    if not _frame_selected_for_stride(i, REF_SPACE_SAVE_STRIDE, T - 1):
        return None
    _refspace_saved_frames.add(i)

    Zref = ref_shape_zyx[0]
    row = {"frame": i}
    for label, vals, out_dir in (
        ("mem", mem_vals, DIRS["refspace_mem"]),
        ("sparse", sparse_vals, DIRS["refspace_sparseCell"]),
    ):
        vol, occ = _scatter_trilinear_to_refspace(coords_xyz, vals, ref_shape_zyx)
        fh.save_single_channel_ome_tiff(vol, str(out_dir), frame_idx=i,
                                         label=f"F260517_refspace_{label}")
        z_occupied = np.flatnonzero(occ.any(axis=(1, 2)))
        row[f"{label}_nonnan_frac"] = float(np.mean(occ))
        row[f"{label}_n_planes_occupied"] = int(z_occupied.size)
        row[f"{label}_z_min"] = int(z_occupied.min()) if z_occupied.size else -1
        row[f"{label}_z_max"] = int(z_occupied.max()) if z_occupied.size else -1
        del vol, occ

    print(f"[QC] refspace frame {i}: "
          f"mem {row['mem_nonnan_frac']*100:.2f}% non-NaN, "
          f"{row['mem_n_planes_occupied']}/{Zref} planes occupied "
          f"(z={row['mem_z_min']}..{row['mem_z_max']}) | "
          f"sparse {row['sparse_nonnan_frac']*100:.2f}% non-NaN, "
          f"{row['sparse_n_planes_occupied']}/{Zref} planes occupied "
          f"(z={row['sparse_z_min']}..{row['sparse_z_max']})")
    return row


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
# IO.readTiff calls tifffile.asarray() on the whole file, which for the 15.1 GB
# moving stack reads every timepoint before any slicing. read_ome_tiff_timepoints
# slices a lazy zarr view instead, so only N_LOAD timepoints leave the disk.
# test_F260517_v2.py loads the same file the same way.
F260517_mov, _ = fh.read_ome_tiff_timepoints(F260517_mov_path, n_timepoints=N_LOAD)
F260517_ref, _ = IO.readTiff(F260517_ref_path)

ref_mem_raw    = F260517_ref[REF_Z0:REF_Z1, REF_CHANNEL_MEM, :, :].astype(np.float32)
if SAVE_PROJECTION_STATE:
    # QC addition 4: state needed to redo the projection call later, saved
    # immediately after ref_mem_raw is loaded/cropped. ref_mem_raw is a slice
    # F260517_ref[REF_Z0:REF_Z1, REF_CHANNEL_MEM, :, :] of the
    # (Z,C,Y,X) reference stack, so its own axis order is (Z,Y,X).
    np.save(str(DIRS["diagnostics"] / "ref_shape.npy"),
            np.array(ref_mem_raw.shape, dtype=np.int64))
    # Values read from the project_coords_to_fixed_planes_gpu call sites in
    # Section 5 (proj_mem_zyx / proj_sparse_zyx) and from the
    # upsample_phase_xy_for_supersurface / upsample_values_xy_for_supersurface
    # call sites that feed them, not copied from any external description.
    projection_params = {
        "z_window": Z_WINDOW,
        "fill_value": FILL_VALUE,
        "coverage_fill_value": COVERAGE_FILL_VALUE,
        "downsample_xy": 1,
        "xy_splat_mode": "subpixel_footprint",
        "xy_extra_radius": 0,
        "upsample_factor": 2,
        "values_interp_order_mem": 1,
        "values_interp_order_sparse": 0,
        "ref_volume_order": "xyz",
        "output_order": "zyx",
        "ref_shape_axis_order": "zyx",
        "save_ref_space_volume": SAVE_REF_SPACE_VOLUME,
        "ref_space_save_stride": REF_SPACE_SAVE_STRIDE,
        # The cropped reference volume is not copied into the output tree; the
        # path and crop that reproduce it are recorded instead. reference_crop
        # is the half-open z range [z0, z1) applied to axis 0 of the (Z,C,Y,X)
        # file, so ref_shape.npy[0] == z1 - z0.
        "reference_source_path": F260517_ref_path,
        "reference_crop": {"z0": REF_Z0, "z1": REF_Z1,
                           "channel_mem": REF_CHANNEL_MEM,
                           "channel_sparse": REF_CHANNEL_SPARSE},
        "moving_source_path": F260517_mov_path,
        "mov_slices": list(MOV_SLICES),
        "z_init_planes_supplied": list(Z_INIT_PLANES),
        "save_ref_surface": SAVE_REF_SURFACE,
    }
    with open(str(DIRS["diagnostics"] / "projection_params.json"), "w") as f_params:
        json.dump(projection_params, f_params, indent=2)
ref_sparse_raw = F260517_ref[REF_Z0:REF_Z1, REF_CHANNEL_SPARSE, :, :].astype(np.float32)
# Select the registered moving slices before the float32 cast, so RAM holds
# len(MOV_SLICES) slices per channel rather than every slice in the file.
mov_mem_all    = F260517_mov[:, MOV_SLICES, 1, :, :].astype(np.float32)
mov_sparse_all = F260517_mov[:, MOV_SLICES, 0, :, :].astype(np.float32)

print(f"  loaded in {time.time()-t0:.1f}s")
print(f"  mov file: {F260517_mov_path}")
print(f"  mov raw shape (T,Z,C,Y,X) = {F260517_mov.shape}  dtype={F260517_mov.dtype}")
print(f"  ref raw shape            = {F260517_ref.shape}  dtype={F260517_ref.dtype}")
print(f"  ref crop z [{REF_Z0},{REF_Z1}) -> ref_mem_raw {ref_mem_raw.shape} (Z,Y,X)")
print(f"  mov slices {MOV_SLICES} -> mov_mem_all {mov_mem_all.shape} (T,K,Y,X)")

# ===========================================================================
# 2. z_init + coords
# ===========================================================================
print("\n[2/7] Initial setup ...")
if Z_INIT_PLANES:
    # z_init supplied, search skipped. The entries are reference planes in the
    # cropped [REF_Z0, REF_Z1) frame, one per entry of MOV_SLICES, checked
    # against len(MOV_SLICES) where Z_INIT_PLANES is parsed.
    z_init = np.asarray(Z_INIT_PLANES, dtype=np.float32)
    z_init_debug = None
    print(f"  z_init supplied (search skipped): {z_init} "
          f"(cropped frame; source planes {z_init + REF_Z0})")
else:
    # return_debug=True: the ZNCC-vs-z0 debug dict is requested so QC addition
    # 1's zinit_match_curve.png/zinit_zncc_heatmap.png can be produced; z_init
    # itself is bit-identical to the un-debugged call (return_debug only adds a
    # second return value).
    z_init, z_init_debug = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
        mov_mem_all[0].transpose(2, 1, 0),
        ref_mem_raw.transpose(2, 1, 0),
        delta_ref_idx=10, use_gradient=False, return_debug=True,
    )
    z_init = z_init.astype(np.float32)
    print(f"  z_init searched: {z_init}")

if z_init.shape[0] != mov_mem_all.shape[1]:
    raise ValueError(
        f"z_init has {z_init.shape[0]} entries but mov_mem_all carries "
        f"{mov_mem_all.shape[1]} slices")
if np.any(z_init < 0) or np.any(z_init > ref_mem_raw.shape[0] - 1):
    raise ValueError(
        f"z_init {z_init} falls outside the cropped reference z range "
        f"[0, {ref_mem_raw.shape[0] - 1}]")

z_idx = np.rint(z_init).astype(np.int32)
z_idx = np.clip(z_idx, 0, ref_mem_raw.shape[0] - 1)

# K comes from the moving array's slice axis, not from a literal count.
K, T = int(mov_mem_all.shape[1]), int(mov_mem_all.shape[0])
T_FILE = T

# The warmup indexes mov_mem_all directly, so a moving file with fewer frames
# than max(WARMUP_FRAMES)+1 would raise IndexError at the calibration step.
# Drop the frames that are not in the file rather than let the index fail.
if max(WARMUP_FRAMES) >= T_FILE:
    WARMUP_FRAMES = [w for w in WARMUP_FRAMES if w < T_FILE]
    if not WARMUP_FRAMES:
        raise ValueError(f"moving file has {T_FILE} frames; warmup needs >= 1")
    print(f"  WARMUP_FRAMES clamped to {WARMUP_FRAMES} "
          f"(moving file has {T_FILE} frames)")

# N_FRAMES limits only the forward loop, not the warmup.
if N_FRAMES_LIMIT is not None:
    T = min(T, N_FRAMES_LIMIT)
print(f"  frames in file={T_FILE}  forward-loop frames={T}  warmup={WARMUP_FRAMES}")

if SAVE_MOTION_FIELD:
    # Corrects the earlier draft of this banner, which named only phase_new
    # and so understated the per-frame write by 2x: motion_current is saved
    # too, and the two arrays are the same shape. The byte count is computed
    # from this run's (Xmov, Ymov, K) rather than restated from a K=20
    # measurement. n_frames_saved counts against the actual loop bound T,
    # using the same predicate _save_motion_field applies, so this projection
    # cannot drift from what the run actually writes.
    n_frames_saved = sum(
        1 for i in range(T) if _frame_selected_for_stride(i, PHASE_SAVE_STRIDE, T - 1)
    )
    per_frame_bytes = 2 * _phase_or_motion_bytes(
        mov_mem_all.shape[3], mov_mem_all.shape[2], K)
    print(f"[QC] SAVE_MOTION_FIELD: {per_frame_bytes/1e6:.1f}MB/frame "
          f"(phase_new_f{{i}}.npy + motion_current_f{{i}}.npy) x {n_frames_saved}/{T} frames "
          f"(stride={PHASE_SAVE_STRIDE}, frame 0 and frame {T-1} always saved) "
          f"= ~{per_frame_bytes*n_frames_saved/1e9:.2f}GB projected for this run. "
          f"SAVE_COMPARE_INPUTS (mov_mem_f{{i}}.npy) / SAVE_COVERAGE_MAP add "
          f"further per-frame bytes not included in this total "
          f"(not measured on this dataset).")

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
if SAVE_MASKS:
    # QC addition 2, mask_ref: update_reference_intensity_mapping_from_target_stack
    # sets option["mask_ref"] itself (verified by reading
    # f260517_helpers.py: `option["mask_ref"] = mask.getMask(...)` then
    # `option["mask_ref"] = mask.bwareafilt3_wei(...)`, not a return value).
    _save_mask_sparse_or_dense(DIRS["diagnostics"] / "mask_ref.npz", option["mask_ref"], "mask_ref")
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
    _save_mask_mov(i, option["mask_mov"])

    phase_new, motion_current, _ = calFlowCrossResolution.getMotion_v2(
        mov_i, ref_mem_adj, option, verbose=False)

    if hasattr(phase_new, "get"):      phase_new = phase_new.get()
    if hasattr(motion_current, "get"): motion_current = motion_current.get()
    _save_motion_field(i, phase_new, motion_current)

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

if SAVE_ALIGNMENT_QC:
    # QC addition 1: alignment QC for fixed_target_z.
    alignment_qc_dir = DIRS["alignment_qc"]
    # target_z_offset_per_plane.npy, shape (2, K): row 0 = fixed_target_z -
    # z_init (per-plane offset the warmup shifted the target away from the
    # initial z guess), row 1 = z_init itself. Named target_z_offset_per_plane
    # rather than zinit_offset_curve to resolve a filename collision with
    # test_F260517_v2.py's own diagnostics/zinit_offset_curve.npy, which
    # holds a different schema: (z0_grid, ZNCC curve) of length M from the
    # FindInitZ z0 scan, versus this file's (per-plane offset, z_init) of
    # length K. The rename means the two scripts' diagnostics/ outputs no
    # longer collide on this filename.
    offset = fixed_target_z - z_init
    np.save(str(alignment_qc_dir / "target_z_offset_per_plane.npy"),
            np.stack([offset.astype(np.float64), z_init.astype(np.float64)]))

if SAVE_ALIGNMENT_QC and z_init_debug is None:
    # Both PNGs plot arrays that exist only inside
    # FindInitZ_stack_global_fixed_spacing's return_debug dict: the (K, Zref)
    # ZNCC score matrix and the summed-ZNCC-vs-z0 curve. Z_INIT_PLANES supplied
    # z_init directly, so that search never ran and neither array exists. No
    # substitute is plotted, because any curve drawn without the search would
    # not be the quantity the filenames name.
    print("  [QC] alignment QC: target_z_offset_per_plane.npy written; "
          "zinit_match_curve.png and zinit_zncc_heatmap.png skipped "
          "(z_init supplied via Z_INIT_PLANES, so the z-init search that "
          "produces the ZNCC curves did not run)")

if SAVE_ALIGNMENT_QC and z_init_debug is not None:
    # zinit_match_curve.png / zinit_zncc_heatmap.png: ported from
    # test_F260517_v2.py (grepped for those two exact filenames at lines 169
    # and 185 of that file), using the z_init_debug this script requests
    # via return_debug=True on the FindInitZ_stack_global_fixed_spacing call
    # above -- the data the port needs is directly available from that call,
    # so no substitute quality metric is needed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pl

    pl.figure()
    pl.imshow(z_init_debug["scores"], aspect="auto", origin="lower", cmap="viridis")
    pl.xlabel("reference z plane")
    pl.ylabel("moving slice k")
    pl.title("ZNCC scores for each moving slice at each reference z plane")
    pl.colorbar(label="ZNCC")
    pl.savefig(str(alignment_qc_dir / "zinit_zncc_heatmap.png"), dpi=130, bbox_inches="tight")
    pl.close()

    z0_grid, curve, _ = fh.compute_zinit_offset_curve(z_init_debug)
    pl.figure()
    pl.plot(z0_grid, curve)
    pl.axvline(z_init_debug["best_z0"], color="crimson", ls="--",
               label=f"chosen z0={z_init_debug['best_z0']}")
    pl.xlabel("starting z-offset z0 (reference plane of slice k=0)")
    pl.ylabel("summed ZNCC over all moving slices")
    pl.title("z-init match curve (rigid comb slid through reference)")
    pl.legend()
    pl.savefig(str(alignment_qc_dir / "zinit_match_curve.png"), dpi=130, bbox_inches="tight")
    pl.close()
    print(f"  [QC] alignment QC (offset array + 2 PNGs) saved to {alignment_qc_dir}")

if SAVE_PROJECTION_STATE:
    # fixed_target_z.npy: confirmed missing from run_F260517_0625.py by
    # grepping that file for np.save/savez -- zero matches -- so this is a
    # new save, not a duplicate of an existing one.
    np.save(str(DIRS["diagnostics"] / "fixed_target_z.npy"),
            fixed_target_z.astype(np.float32))

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
refspace_records = []

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
    _save_mask_mov(i, option["mask_mov"])

    phase_new, motion_current, mem_mapped_xyk = calFlowCrossResolution.getMotion_v2(
        mov_mem_xyk, ref_mem_adj, option, verbose=False)

    if hasattr(phase_new, "get"):      phase_new = phase_new.get()
    if hasattr(motion_current, "get"): motion_current = motion_current.get()
    if hasattr(mem_mapped_xyk, "get"): mem_mapped_xyk = mem_mapped_xyk.get()

    phase_new      = np.asarray(phase_new, dtype=np.float32)
    motion_current = np.asarray(motion_current, dtype=np.float32)
    mem_mapped_zyx = np.asarray(mem_mapped_xyk, dtype=np.float32).transpose(2, 1, 0)  # (K,Y,X)
    _save_motion_field(i, phase_new, motion_current)

    registered_cache[i] = mem_mapped_zyx

    # ---- Z-plane projection ----
    phase_for_proj = fh.upsample_phase_xy_for_supersurface(phase_new, upsample_factor=2)
    raw_mem_xyk  = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    raw_sparse_xyk = raw_sparse_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    mem_vals  = fh.upsample_values_xy_for_supersurface(raw_mem_xyk, upsample_factor=2, order=1)
    sparse_vals = fh.upsample_values_xy_for_supersurface(raw_sparse_xyk, upsample_factor=2, order=0)

    # ---- Full reference-space scatter (no z-window gate) ----
    # Reuses phase_for_proj/mem_vals/sparse_vals computed just above (same
    # arrays the z-window-gated projection below consumes), so this output
    # never re-upsamples or resamples anything the projection already did.
    refspace_row = _save_refspace_volume(i, phase_for_proj, mem_vals, sparse_vals, ref_mem_raw.shape)
    if refspace_row is not None:
        refspace_records.append(refspace_row)

    # ---- Reference sampled onto the same fixed target planes (frame 0 only) ----
    # Placed here so it sees the same phase_for_proj the two projection calls
    # below consume. ref_mem_adj and ref_sparse_raw.transpose(2,1,0) are the
    # same two ref_volume arguments those calls pass.
    _save_ref_surface(i, phase_for_proj, ref_mem_adj, ref_sparse_raw.transpose(2, 1, 0))

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
    cov_output_order = "zyx"
    cov = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_proj, ref_volume=ref_mem_adj,
        target_z_planes=fixed_target_z,
        values_xyk=np.ones(phase_for_proj.shape[:-1], dtype=np.float32),
        ref_volume_order="xyz", z_window=Z_WINDOW, downsample_xy=1,
        fill_value=COVERAGE_FILL_VALUE, return_numpy=True, output_order=cov_output_order,
        xy_splat_mode="subpixel_footprint", xy_extra_radius=0)
    if hasattr(cov, "get"): cov = cov.get()
    cov = np.asarray(cov, dtype=np.float32)
    _save_coverage_map(i, cov, axis_order=cov_output_order)
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

df_refspace = pd.DataFrame(refspace_records)
df_refspace.to_csv(str(DIRS["diagnostics"] / "refspace_summary.csv"), index=False)

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
