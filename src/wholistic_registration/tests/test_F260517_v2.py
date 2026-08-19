# %% [markdown]
# ## F260517 Registration -- Forward-Only Pipeline (First-5-Frames Warmup)
#
# Registers a sparse moving stack (K z-planes acquired over time) to a dense
# reference volume and re-projects the moving signal into the reference's
# coordinate frame, producing a stable, motion-corrected movie.
#
# Acquisition metadata: zRatio = 10, frame_rate = 4.07066 s/frame, z_window = 4.0
#
# Interactive-cell (`# %%`) port of `test_F260517_v2.ipynb`, repointed to the
# Janelia lab server. All machine-specific values live in the CONFIG cell below.

# %%
# ============================================================================
# CONFIG -- the only cell that should differ between machines. Keep edits here
# out of shared commits (machine-specific paths / GPU index).
# ============================================================================
from pathlib import Path

# GPU index to run on. Collaborator's machine used device 1; this host has a
# single GPU at index 0.
GPU_DEVICE = 0

# Input data (read-only). exp = moving time-lapse, anat = dense reference.
DATA_DIR = Path("/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b")
F260517_mov_path = str(DATA_DIR / "260517_exp_00001_TZCYX.ome.tiff")
F260517_ref_path = str(DATA_DIR / "260517_anat_00003_TZCYX.ome.tiff")

# Output root (writable, large). One subfolder per saved channel is created here.
# Tagged with the run owner so it is clearly distinct from collaborators' runs.
RUN_TAG = "ruettenv"  # change per person / experiment so runs don't collide
base_out_dir = str(DATA_DIR / "registration_out" / f"f260517_{RUN_TAG}")

# Smoke-test control: process only the first N time frames so we can check the
# pipeline runs end-to-end before committing to a full multi-hour run. Set to
# None to process every frame. Must be >= 6 to exercise the forward loop
# (warmup consumes frames 0-4).
N_FRAMES_LIMIT = 8

# %%
import os
import sys

import cupy as cp

cp.cuda.Device(GPU_DEVICE).use()

import numpy as np
import pandas as pd
import tifffile
import zarr
from collections import deque
from importlib import reload
from typing import Any
from numpy import ndarray
from scipy.ndimage import distance_transform_edt, map_coordinates, maximum_filter

# Package-relative imports (work from any cwd after `pip install -e .`), so the
# original `os.chdir(...) + from utils import ...` hack is no longer needed.
from wholistic_registration.utils import IO
from wholistic_registration.utils import calFlowCrossResolution
from wholistic_registration.utils import preprocess as prep
from wholistic_registration.utils import mask
from wholistic_registration.utils import option

print("Imports done.")


# %%
# ============================================================================
# Helper functions live in f260517_helpers.py (sibling module) so they can be
# hot-reloaded without restarting the kernel. Call them via the `fh.` namespace
# (e.g. fh.read_ome_tiff_timepoints) -- that is what makes reload(fh) actually
# rebind the functions you call.
# ============================================================================
from importlib import reload

# tests/ is not an importable package, so add it to sys.path. Works both as a
# script (python test_F260517_v2.py) and when running cells interactively.
_HELPER_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.path.join(
        "/groups/ahrens/home/ruttenv/python_packages/wholistic_registration",
        "src/wholistic_registration/tests",
    )
)
if _HELPER_DIR not in sys.path:
    sys.path.insert(0, _HELPER_DIR)

import f260517_helpers as fh
reload(fh)   # re-run this cell after editing f260517_helpers.py to pick up changes

print("Helper module loaded:", fh.__file__)


# %%
# ============================================================================
# Data loading + setup
# ============================================================================
reload(calFlowCrossResolution)
reload(prep)

# Moving (exp, ~15 GB): load only the first N_FRAMES_LIMIT timepoints. Loading
# more than we process wastes time and RAM, so cap the read at what the pipeline
# will actually use. Reference (anat, ~1.3 GB) is small -> load in full.
F260517_mov, F260517_mov_desc = fh.read_ome_tiff_timepoints(
    F260517_mov_path, n_timepoints=N_FRAMES_LIMIT,
)

F260517_ref, F260517_ref_desc = IO.readTiff(F260517_ref_path)
print(f"Finish reading the dataset (moving timepoints loaded: {F260517_mov.shape[0]})")
#%%

# Split each acquisition into its two channels.
# Reference (anat) is (Zref, C, Y, X); planes 90:310 hold the imaged volume.
F260517_ref_mem = F260517_ref[90:310, 1, :, :]
F260517_ref_sparseCell = F260517_ref[90:310, 0, :, :]

# Moving (exp) is (T, K, C, Y, X). Channel 1 = membrane, channel 0 = sparse cell.
F260517_mov_mem: ndarray[tuple[int, ...], Any] = F260517_mov[:, :, 1, :, :]
F260517_mov_sparseCell: ndarray[tuple[int, ...], Any] = F260517_mov[:, :, 0, :, :]

print("F260517_ref_mem:", F260517_ref_mem.shape)          # (Zref, Y, X)
print("F260517_ref_sparseCell:", F260517_ref_sparseCell.shape)
print("F260517_mov_mem:", F260517_mov_mem.shape)          # (T, K, Y, X)
print("F260517_mov_sparseCell:", F260517_mov_sparseCell.shape)
print("Finish splitting the dataset")
#%%
# --- Registration options ---------------------------------------------------
option["r"] = 5
option["layer"] = 3
option["iter"] = 10
option["movRange"] = 5.0
option["tol"] = 1e-6
option["zRatio_HR"] = 1
option["wrong_region_enable"] = False

thresFactor = 5.0
maskRange = [5.0, 4000.0]
smoothPenalty_raw = 0.01

# --- Find the initial reference z slice for each moving slice ----------------
# Establishes phi_0's z component: which reference plane each of the K moving
# slices sits on at rest. The K moving slices are assumed equally spaced in the
# reference, so the routine does a single global search over one starting offset
# z0, scoring each candidate by image similarity (ZNCC) of every moving slice k
# placed at z0 + k*delta_ref_idx, and returns z_init = z0 + k*delta_ref_idx (K,).
z_init, z_init_debug = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
    F260517_mov_mem[0].transpose(2, 1, 0),     # moving frame 0 as (X, Y, K) -- the stack to place
    F260517_ref_mem.transpose(2, 1, 0),        # reference volume as (X, Y, Zref) -- searched over z
    delta_ref_idx=10,                          # fixed gap (in ref planes) between consecutive moving slices
    use_gradient=False,                        # match raw intensity; True would match gradient-magnitude edges
    return_debug=True,                         # also return the per-(slice, z) ZNCC scores for diagnostics
)

#%%
import matplotlib.pyplot as pl

plot_dir = os.path.join(base_out_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

pl.figure()
pl.imshow(z_init_debug["scores"], aspect="auto", origin="lower", cmap="viridis")
pl.xlabel("reference z plane")
pl.ylabel("moving slice k")
pl.title("ZNCC scores for each moving slice at each reference z plane")
pl.colorbar(label="ZNCC")
pl.savefig(os.path.join(plot_dir, "zinit_zncc_heatmap.png"), dpi=130, bbox_inches="tight")
pl.show()

#%%
plot_dir = os.path.join(base_out_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

z0_grid, curve, _ = fh.compute_zinit_offset_curve(z_init_debug)
pl.figure()
pl.plot(z0_grid, curve)
pl.axvline(z_init_debug["best_z0"], color="crimson", ls="--",
           label=f"chosen z0={z_init_debug['best_z0']}")  # this max == z_init's z0
pl.xlabel("starting z-offset z0 (reference plane of slice k=0)")
pl.ylabel("summed ZNCC over all moving slices")
pl.title("z-init match curve (rigid comb slid through reference)")
pl.legend()
pl.savefig(os.path.join(plot_dir, "zinit_match_curve.png"), dpi=130, bbox_inches="tight")
pl.show()

#%%
print("z_init_debug:", z_init_debug)


#%%
z_init = np.asarray(z_init, dtype=np.float32)
z_idx = np.rint(z_init).astype(np.int32)
z_idx = np.clip(z_idx, 0, F260517_ref_mem.shape[0] - 1)

K = z_init.shape[0]
T = F260517_mov_mem.shape[0]

# Build the initial deformation field phi_0: identity in XY, z_init in z.
x = np.arange(F260517_mov_mem[0].shape[2], dtype=np.float32)
y = np.arange(F260517_mov_mem[0].shape[1], dtype=np.float32)
k = np.arange(K, dtype=np.int32)

X_grid, Y_grid, K_grid = np.meshgrid(x, y, k, indexing="ij")

coords_xyz = np.empty(
    (F260517_mov_mem[0].shape[2], F260517_mov_mem[0].shape[1], K, 3),
    dtype=np.float32,
)
coords_xyz[..., 0] = X_grid
coords_xyz[..., 1] = Y_grid
coords_xyz[..., 2] = z_init[K_grid]

option["phase"] = coords_xyz

print("z_init:", z_init)
print("z_idx:", z_idx)
print("phase shape:", option["phase"].shape)
print("T (total frames):", T)

# Percentiles used for quantile-based intensity mapping.
percentiles = [
    0.1, 0.5, 1, 2, 5,
    10, 25, 50, 75, 90,
    95, 99, 99.5, 99.8,
]

# --- Projection settings ----------------------------------------------------
projection_z_window = 3.0
projection_downsample_xy = 1
projection_fill_value = -200.0
projection_xy_extra_radius = 0

surface_upsample_factor = 2
surface_mem_value_order = 1
surface_sparse_value_order = 0

enable_coverage_diagnostics = True
coverage_threshold = 0.0
enable_projection_hole_filling = False

# --- Output directories (base_out_dir comes from the CONFIG cell) -----------
out_dirs = {
    "raw_moving_mem": os.path.join(base_out_dir, "raw_moving_mem"),
    "raw_moving_sparseCell": os.path.join(base_out_dir, "raw_moving_sparseCell"),
    "projected_mem": os.path.join(base_out_dir, "projected_mem"),
    "projected_sparseCell": os.path.join(base_out_dir, "projected_sparseCell"),
}

for _d in out_dirs.values():
    os.makedirs(_d, exist_ok=True)

diagnostic_dir = os.path.join(base_out_dir, "diagnostics")
os.makedirs(diagnostic_dir, exist_ok=True)

# Re-learn the reference intensity mapping every this-many forward frames.
ref_update_every = 40

print("Initial setup done.")


# %%
# ============================================================================
# Diagnostic: z-initialisation quality
# Plot the global match score vs the candidate z-offset z0. A sharp, isolated
# peak => the z init is well constrained; a flat or multi-peaked curve => the
# init is ambiguous / noise-driven and downstream registration may start wrong.
# ============================================================================
import matplotlib.pyplot as plt

z0_grid, zinit_curve, zinit_per_k_z = fh.compute_zinit_offset_curve(z_init_debug)
zinit_stats = fh.summarize_zinit_peakiness(z0_grid, zinit_curve, z_init_debug["best_z0"])

print("[z-init] peakiness:", {k: round(v, 3) for k, v in zinit_stats.items()})

# Save the raw curve so it can be reloaded without re-running registration.
np.save(os.path.join(diagnostic_dir, "zinit_offset_curve.npy"),
        np.stack([z0_grid.astype(np.float64), zinit_curve]))

fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(13, 4.2))

# Left: the 1-D match-vs-offset curve with the chosen offset marked.
ax_curve.plot(z0_grid, zinit_curve, lw=1.5)
ax_curve.axvline(z_init_debug["best_z0"], color="crimson", ls="--",
                 label=f"chosen z0={z_init_debug['best_z0']}")
ax_curve.set_xlabel("starting z-offset z0 (reference plane of slice k=0)")
ax_curve.set_ylabel("summed ZNCC over all moving slices")
ax_curve.set_title(
    f"z-init match curve  (peak/sigma={zinit_stats['peak_z_over_sigma']:.1f}, "
    f"rel. prominence={zinit_stats['rel_prominence']:.2f})"
)
ax_curve.legend()

# Right: full per-slice ZNCC landscape (K x Z) with the selected z path overlaid.
scores_kz = np.asarray(z_init_debug["scores"], dtype=np.float64)  # (K, Z)
im = ax_heat.imshow(scores_kz, aspect="auto", origin="lower", cmap="viridis")
best_idx = int(np.argmax(zinit_curve))
ax_heat.plot(zinit_per_k_z[best_idx], np.arange(scores_kz.shape[0]),
             color="crimson", lw=1.2, marker=".", ms=3, label="chosen z per slice")
ax_heat.set_xlabel("reference z plane")
ax_heat.set_ylabel("moving slice k")
ax_heat.set_title("per-slice ZNCC(slice k, ref z)")
ax_heat.legend(loc="upper right")
fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

fig.tight_layout()
_zinit_png = os.path.join(diagnostic_dir, "zinit_offset_curve.png")
fig.savefig(_zinit_png, dpi=130)
print(f"[z-init] saved diagnostic plot -> {_zinit_png}")
plt.show()


# %%
# ============================================================================
# Initial reference intensity mapping
# Target = mean of RAW moving frames 0..4 (start of the recording).
# ============================================================================
init_calibration_frames = [0, 1, 2, 3, 4]
init_target_stack = np.mean(
    F260517_mov_mem[init_calibration_frames].astype(np.float32),
    axis=0,
)  # (K, Y, X)

print("Initial calibration frames:", init_calibration_frames)
print("Target stack shape:", init_target_stack.shape)

F260517_ref_mem_adj, src_q, tgt_q, used_percentiles = (
    fh.update_reference_intensity_mapping_from_target_stack(
        F260517_ref_mem=F260517_ref_mem,
        target_stack_zyx=init_target_stack,
        z_idx=z_idx,
        option=option,
        thresFactor=thresFactor,
        maskRange=maskRange,
        smoothPenalty_raw=smoothPenalty_raw,
        percentiles=percentiles,
    )
)

print("[Init] Updated reference intensity mapping using mean of raw frames", init_calibration_frames)
print("src_q:", src_q)
print("tgt_q:", tgt_q)
#%% # check color mapping

F260517_ref_mem_adj.shape
vmin, vmax = np.percentile(F260517_ref_mem[0,:,:], [5, 99])
pl.figure()
pl.imshow(F260517_ref_mem[20,:,:], vmin=vmin, vmax=vmax)
pl.colorbar()
pl.figure()
pl.imshow(F260517_ref_mem_adj[:,:,20].T , vmin=vmin, vmax=vmax)
pl.colorbar()
pl.figure()
pl.imshow(F260517_mov_mem[0,0], vmin=vmin, vmax=vmax)
pl.colorbar()

# %%
# ============================================================================
# Forward-only registration pipeline
#   Phase 1 (warmup): register frames 0..4, derive fixed_target_z.
#   Phase 2 (forward): register 5..T_run, re-learning the reference intensity
#                      mapping every `ref_update_every` frames.
# ============================================================================

# How many time frames to actually process (smoke-test cap from CONFIG).
T_run = T if N_FRAMES_LIMIT is None else min(T, int(N_FRAMES_LIMIT))
print(f"Processing {T_run} of {T} frames (N_FRAMES_LIMIT={N_FRAMES_LIMIT}).")
#%%
# --- Global state -----------------------------------------------------------
registered_mem_mapped = {}   # frame_idx -> mem_mapped_zyx (K, Y, X)
processed_count = 0
frames_since_ref_update = 0

target_z_pred_list = []
all_z_plane_stats = []
all_coverage_stats = []
records_warmup = []

fixed_target_z = None
target_z_combine_method = "median"
warmup_n = 5


def update_ref_from_recent_frames(frame_indices):
    """Re-learn the reference intensity mapping from the mean of the RAW MOVING
    frames at the given indices (same target as the initial calibration). The raw
    moving data is what carries the brightness drift over time; the warped
    reference reconstruction does not, so it must not be the calibration target."""
    stacks = []
    for fi in frame_indices:
        if fi in registered_mem_mapped:
            stacks.append(F260517_mov_mem[fi].astype(np.float32, copy=False))
        else:
            print(f"  WARNING: frame {fi} not processed yet, skipping for ref update")

    if len(stacks) == 0:
        print("  WARNING: no cached frames available for ref update, skipping")
        return None, None, None

    target_stack = np.mean(np.stack(stacks, axis=0), axis=0).astype(np.float32, copy=False)

    ref_adj, sq, tq, up = fh.update_reference_intensity_mapping_from_target_stack(
        F260517_ref_mem=F260517_ref_mem,
        target_stack_zyx=target_stack,
        z_idx=z_idx,
        option=option,
        thresFactor=thresFactor,
        maskRange=maskRange,
        smoothPenalty_raw=smoothPenalty_raw,
        percentiles=percentiles,
    )

    print(f"[RefUpdate] Calibrated from frames {frame_indices}, using {len(stacks)} stacks")
    print(f"  src_q: {sq}")
    print(f"  tgt_q: {tq}")

    return ref_adj, sq, tq


def process_single_frame(i, ref_mem_adj):
    """Register moving frame i against ref_mem_adj and return the result dict."""
    raw_mem_zyx = F260517_mov_mem[i].astype(np.float32, copy=False)
    raw_sparse_zyx = F260517_mov_sparseCell[i].astype(np.float32, copy=False)

    mov_mem_i = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)  # (X, Y, K)

    # Moving-frame mask for this timepoint.
    option["mask_mov"] = mask.getMask(mov_mem_i, thresFactor)
    option["mask_mov"] = mask.bwareafilt3_wei(option["mask_mov"], maskRange)

    # Core registration: phase_new = phi(X), motion_current = optimised part,
    # mem_mapped = reference sampled at phi (the model's reconstruction).
    phase_new, motion_current, mem_mapped = calFlowCrossResolution.getMotion_v2(
        mov_mem_i, ref_mem_adj, option, verbose=False,
    )

    if hasattr(phase_new, "get"):
        phase_new = phase_new.get()
    if hasattr(motion_current, "get"):
        motion_current = motion_current.get()
    if hasattr(mem_mapped, "get"):
        mem_mapped = mem_mapped.get()

    phase_new = phase_new.astype(np.float32, copy=False)
    motion_current = motion_current.astype(np.float32, copy=False)
    mem_mapped = mem_mapped.astype(np.float32, copy=False)

    # Registration residual (large by construction: mem_mapped is ref-derived).
    mem_err = float(np.mean(np.abs(mov_mem_i - mem_mapped))) # actual pixels recorded in low res space, and the sampled points are in high res space

    mem_mapped_zyx = mem_mapped.transpose(2, 1, 0).astype(np.float32, copy=False)

    # ONLY FOR VISUALIZATION: estimate the target z plane for each moving slice
    target_z_pred_i, df_z_i = fh.estimate_projection_z_from_phase_simple(
        phase_new=phase_new,
        z_init=z_init,
        ref_shape=F260517_ref_mem.shape,
        ref_volume_order="zyx",
        method="trimmed_mean",
        trim_percentiles=(5, 95),
        frame_idx=i,
    )

    # Warm-start the next frame from a decayed copy of this frame's motion.
    option["motion"] = (0.7 * motion_current).astype(np.float32, copy=False)

    return {
        "frame": i,
        "phase_new": phase_new,
        "motion_current": motion_current,
        "mem_mapped_zyx": mem_mapped_zyx,
        "mem_err": mem_err,
        "target_z_pred": target_z_pred_i,
        "df_z": df_z_i,
        "raw_mem_zyx": raw_mem_zyx,
        "raw_sparse_zyx": raw_sparse_zyx,
    }


def project_and_save_result(result, ref_mem_adj_to_use):
    """Project one frame result to the fixed planes and save its four volumes."""
    coverage_rows = fh.project_and_save_single_frame_moving_derived(
        frame_idx=result["frame"],
        phase_new=result["phase_new"],
        fixed_target_z=fixed_target_z,
        ref_mem_adj_for_projection=ref_mem_adj_to_use,
        F260517_ref_sparseCell=F260517_ref_sparseCell,
        raw_mem_zyx=result["raw_mem_zyx"],
        raw_sparseCell_zyx=result["raw_sparse_zyx"],
        out_dirs=out_dirs,
        surface_upsample_factor=surface_upsample_factor,
        surface_mem_value_order=surface_mem_value_order,
        surface_sparse_value_order=surface_sparse_value_order,
        enable_coverage_diagnostics=enable_coverage_diagnostics,
        coverage_threshold=coverage_threshold,
        enable_hole_filling=enable_projection_hole_filling,
        projection_z_window=projection_z_window,
        projection_downsample_xy=projection_downsample_xy,
        projection_fill_value=projection_fill_value,
        projection_xy_extra_radius=projection_xy_extra_radius,
    )
    return coverage_rows


# First frame starts from phi_0 with no temporal prior.
option["phase"] = coords_xyz.copy()
option.pop("motion", None)

# --- Phase 1: warmup --------------------------------------------------------
print("=" * 80)
print("Phase 1: Warmup -- registering frames 0->4")
print("=" * 80)

warmup_frames = [0, 1, 2, 3, 4]

for idx, i in enumerate(warmup_frames):
    if idx == 0:
        option["phase"] = coords_xyz.copy()
        option.pop("motion", None)

    result = process_single_frame(i, F260517_ref_mem_adj)

    registered_mem_mapped[i] = result["mem_mapped_zyx"]
    target_z_pred_list.append(result["target_z_pred"])
    all_z_plane_stats.append(result["df_z"])
    records_warmup.append(result)

    processed_count += 1
    frames_since_ref_update += 1

    print(f"[Warmup {idx+1}/{warmup_n}] Frame {i}: mem_err={result['mem_err']:.4f}")

# Fixed projection planes, combined across warmup frames for stability.
fixed_target_z = fh.robust_average_target_z(
    target_z_pred_list,
    method=target_z_combine_method,
    trim_percentiles=(10, 90),
)
bad = ~np.isfinite(fixed_target_z)
fixed_target_z[bad] = z_init[bad]

print("=" * 80)
print(f"[Projection] fixed_target_z determined from {len(warmup_frames)} warmup frames")
print("z_init:", z_init)
print("fixed_target_z:", fixed_target_z)
print("fixed_target_z - z_init:", fixed_target_z - z_init)
print("=" * 80)

# Now that fixed_target_z exists, project and save the buffered warmup frames.
for rec in records_warmup:
    coverage_rows = project_and_save_result(rec, F260517_ref_mem_adj)
    all_coverage_stats.extend(coverage_rows)
    print(f"[Projection] saved warmup frame {rec['frame']} | mem_err={rec['mem_err']:.4f}")

records_warmup.clear()

# --- Phase 2: forward -------------------------------------------------------
print("=" * 80)
print("Phase 2: Forward 5->T_run")
print("=" * 80)

for i in range(5, T_run):
    # TODO(Concern 4): `i > 75` is a per-dataset magic switch. Revisit -- drive
    # the wrong-region correction from a named parameter or a drift monitor.
    if i > 75:
        option["wrong_region_enable"] = True

    result = process_single_frame(i, F260517_ref_mem_adj)

    registered_mem_mapped[i] = result["mem_mapped_zyx"]
    all_z_plane_stats.append(result["df_z"])

    processed_count += 1
    frames_since_ref_update += 1

    coverage_rows = project_and_save_result(result, F260517_ref_mem_adj)
    all_coverage_stats.extend(coverage_rows)

    print(f"[Fwd] Frame {i}: mem_err={result['mem_err']:.4f} | "
          f"processed={processed_count} | since_ref_update={frames_since_ref_update}")

    # Periodic reference intensity recalibration from the most recent frames.
    if frames_since_ref_update >= ref_update_every:
        calib_frames = sorted(registered_mem_mapped.keys())[-5:]
        print(f"\n[RefUpdate] Updating reference mapping using recent frames {calib_frames}")

        new_ref, sq, tq = update_ref_from_recent_frames(calib_frames)
        if new_ref is not None:
            F260517_ref_mem_adj = new_ref
        frames_since_ref_update = 0

print("=" * 80)
print(f"Pipeline complete. Processed {processed_count} frames total.")
print(f"Frames in cache: {len(registered_mem_mapped)}")
print("=" * 80)


# %%
# ============================================================================
# Save diagnostics
# ============================================================================
if len(all_z_plane_stats) > 0:
    z_plane_stats_all = pd.concat(all_z_plane_stats, ignore_index=True)
    z_plane_stats_all.to_csv(
        os.path.join(diagnostic_dir, "z_plane_pred_stats.csv"),
        index=False,
    )
    print(f"Saved z_plane_stats: {len(z_plane_stats_all)} rows")

if len(all_coverage_stats) > 0:
    coverage_stats_all = pd.DataFrame(all_coverage_stats)
    coverage_stats_all.to_csv(
        os.path.join(diagnostic_dir, "coverage_stats.csv"),
        index=False,
    )
    print(f"Saved coverage_stats: {len(coverage_stats_all)} rows")

if fixed_target_z is not None:
    np.save(
        os.path.join(diagnostic_dir, "fixed_target_z.npy"),
        fixed_target_z.astype(np.float32),
    )
    print("Saved fixed_target_z.npy")

np.save(
    os.path.join(diagnostic_dir, "processing_order.npy"),
    np.array(sorted(registered_mem_mapped.keys()), dtype=np.int32),
)

print("Diagnostics saved.")
print("Done.")
