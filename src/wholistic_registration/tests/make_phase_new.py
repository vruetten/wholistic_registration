"""
Produce real ``phase_new`` arrays for the projector comparison.

``phase_new`` is computed in-memory by the F260517 pipeline and never saved, so
this script reproduces the minimal registration path (reference intensity
adjustment + ``getMotion_v2``) for a couple of frames and writes the arrays the
thin comparison runner consumes:

    phase_new_f{idx}.npy   (Xmov, Ymov, K, 3)
    mov_mem_f{idx}.npy     (K, Ymov, Xmov)
    fixed_target_z.npy     (K,)
    ref_shape.npy          (3,)  reference (Xref, Yref, Zref), order "xyz"

Mirrors the prep in ``test_F260517_v2.py`` (membrane channel, planes 90:310,
quantile intensity mapping learned from the mean of frames 0-4).
"""

import os

import cupy as cp

cp.cuda.Device(0).use()

import numpy as np
import tifffile
import zarr

from wholistic_registration.utils import IO
from wholistic_registration.utils import calFlowCrossResolution
from wholistic_registration.utils import mask
from wholistic_registration.utils import option
from wholistic_registration.utils import preprocess as prep

DATA_DIR = "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b"
MOV_PATH = os.path.join(DATA_DIR, "260517_exp_00001_TZCYX.ome.tiff")
REF_PATH = os.path.join(DATA_DIR, "260517_anat_00003_TZCYX.ome.tiff")
OUT_DIR = os.path.join(DATA_DIR, "registration_out", "projector_compare_data")

N_FRAMES = 8
FRAMES_TO_SAVE = [5, 7]

# The full-res volume (220, 1500, 630) does not fit in this host's 16 GB GPU.
# Crop a central XY sub-volume so the registration fits; this is still a real
# phase_new on real data, which is all the projector comparison needs.
CROP_Y = slice(350, 1150)   # of 1500
CROP_X = slice(15, 615)     # of 630

thresFactor = 5.0
maskRange = [5.0, 4000.0]
smoothPenalty_raw = 0.01
percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]


def read_ome_tiff_timepoints(tiff_path, n_timepoints):
    with tifffile.TiffFile(tiff_path) as tif:
        store = tif.aszarr()
        try:
            z = zarr.open(store, mode="r")
            n = min(int(n_timepoints), z.shape[0])
            sub = np.asarray(z[:n])
        finally:
            store.close()
    return sub


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading data ...")
    mov = read_ome_tiff_timepoints(MOV_PATH, N_FRAMES)        # (T, K, C, Y, X)
    ref, _ = IO.readTiff(REF_PATH)                            # (Zref, C, Y, X)

    ref_mem = ref[90:310, 1, CROP_Y, CROP_X]                  # (Zref, Y, X)
    mov_mem = mov[:, :, 1, CROP_Y, CROP_X]                    # (T, K, Y, X)
    print("ref_mem:", ref_mem.shape, "mov_mem:", mov_mem.shape)

    option["r"] = 5
    option["layer"] = 3
    option["iter"] = 10
    option["movRange"] = 5.0
    option["tol"] = 1e-6
    option["zRatio_HR"] = 1
    option["wrong_region_enable"] = False

    # Initial reference z for each moving plane -> phi_0.
    z_init = calFlowCrossResolution.FindInitZ_stack_global_fixed_spacing(
        mov_mem[0].transpose(2, 1, 0),
        ref_mem.transpose(2, 1, 0),
        delta_ref_idx=10,
        use_gradient=False,
    )
    z_init = np.asarray(z_init, dtype=np.float32)
    z_idx = np.clip(np.rint(z_init).astype(np.int32), 0, ref_mem.shape[0] - 1)
    K = z_init.shape[0]

    x = np.arange(mov_mem[0].shape[2], dtype=np.float32)
    y = np.arange(mov_mem[0].shape[1], dtype=np.float32)
    k = np.arange(K, dtype=np.int32)
    X_grid, Y_grid, K_grid = np.meshgrid(x, y, k, indexing="ij")
    coords_xyz = np.empty((mov_mem[0].shape[2], mov_mem[0].shape[1], K, 3), np.float32)
    coords_xyz[..., 0] = X_grid
    coords_xyz[..., 1] = Y_grid
    coords_xyz[..., 2] = z_init[K_grid]

    # Reference intensity adjustment from mean of frames 0..4 (initial calibration).
    target = np.mean(mov_mem[[0, 1, 2, 3, 4]].astype(np.float32), axis=0)  # (K,Y,X)
    src_q, tgt_q, _ = prep.learn_quantile_mapping(
        source=ref_mem[z_idx].astype(np.float32), target=target, percentiles=percentiles
    )
    ref_mem_adj = (
        prep.apply_quantile_mapping(ref_mem, src_q, tgt_q)
        .transpose(2, 1, 0)
        .astype(np.float32, copy=False)
    )  # (Xref, Yref, Zref)
    option["mask_ref"] = mask.bwareafilt3_wei(
        mask.getMask(ref_mem_adj, thresFactor), maskRange
    )
    option["smoothPenalty"] = prep.getSmPnltNormFctr(ref_mem_adj, option) * smoothPenalty_raw

    np.save(os.path.join(OUT_DIR, "fixed_target_z.npy"), z_init)
    np.save(os.path.join(OUT_DIR, "ref_shape.npy"), np.asarray(ref_mem_adj.shape, np.int64))
    print("ref_mem_adj (X,Y,Zref):", ref_mem_adj.shape)

    for i in FRAMES_TO_SAVE:
        mov_mem_i = mov_mem[i].transpose(2, 1, 0).astype(np.float32, copy=False)  # (X,Y,K)
        option["phase"] = coords_xyz.copy()
        option.pop("motion", None)  # independent fit (no warm start) for clean phase
        option["mask_mov"] = mask.bwareafilt3_wei(
            mask.getMask(mov_mem_i, thresFactor), maskRange
        )
        phase_new, _motion, _mem = calFlowCrossResolution.getMotion_v2(
            mov_mem_i, ref_mem_adj, option, verbose=False
        )
        if hasattr(phase_new, "get"):
            phase_new = phase_new.get()
        phase_new = np.asarray(phase_new, np.float32)

        np.save(os.path.join(OUT_DIR, f"phase_new_f{i}.npy"), phase_new)
        np.save(os.path.join(OUT_DIR, f"mov_mem_f{i}.npy"), mov_mem[i].astype(np.float32))
        print(f"[saved] frame {i}: phase_new {phase_new.shape}")

        del phase_new, _motion, _mem, mov_mem_i
        cp.get_default_memory_pool().free_all_blocks()

    print("Done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
