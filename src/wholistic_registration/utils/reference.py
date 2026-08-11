import numpy as np

from . import cp


def transform(image, k=1, method="raw"):
    if method == "raw":
        return k * image
    elif method == "sqrt":
        return k * np.sqrt(image)
    elif method == "log2":
        return k * np.log2(1 + image)
    elif method == "log10":
        return k * np.log10(1 + image)
    else:
        raise ValueError(f"Unknown method to process the image:{method}")


def pick_initial_reference(
    frames: cp.ndarray,
    max_corr_frames: int = 20,
    downsample: int = 1,
    spatial_ds: int = 4,  # 新增：用于相关性计算的空间降采样
):
    """
    Safe GPU version for picking initial reference.
    """

    ndim = frames.ndim

    # -----------------------------
    # Step 1: select slices
    # -----------------------------
    if ndim == 4:
        # frames: [T, Z, Y, X]
        T, Z, Y, X = frames.shape

        if Z >= 3:
            z_mid = Z // 2
            z_indices = slice(z_mid - 1, z_mid + 2)
        else:
            z_indices = slice(0, Z)

        frames_used = frames[:, z_indices, :, :]

    elif ndim == 3:
        # frames: [T, Y, X]
        T, Y, X = frames.shape
        frames_used = frames[:, None, :, :]  # fake Z dim

    else:
        raise ValueError("frames must be [T,Y,X] or [T,Z,Y,X]")

    # -----------------------------
    # Step 2: spatial downsample for correlation
    # -----------------------------
    frames_small = frames_used[:, :, ::spatial_ds, ::spatial_ds]

    # 强制 contiguous（非常关键）
    frames_small = cp.ascontiguousarray(frames_small, dtype=cp.float32)

    # -----------------------------
    # Step 3: flatten safely
    # -----------------------------
    T = frames_small.shape[0]
    frames_flat = frames_small.reshape(T, -1)

    # -----------------------------
    # Step 4: correlation computation
    # -----------------------------
    frames_mean = frames_flat.mean(axis=1, keepdims=True)
    frames_demeaned = frames_flat - frames_mean

    cc = frames_demeaned @ frames_demeaned.T

    diag = cp.sqrt(cp.diag(cc)) + 1e-12
    cc = cc / (diag[:, None] * diag[None, :])

    # -----------------------------
    # Step 5: pick best frame
    # -----------------------------
    ncorr = min(max_corr_frames, T - 1)
    if ncorr < 1:
        # degenerate block (T <= 1 or max_corr_frames == 0): average everything
        return frames.mean(axis=0), cp.arange(T)
    # mean of the top-ncorr correlations excluding self (column 0)
    CCsort = -cp.sort(-cc, axis=1)
    bestCC = CCsort[:, 1:ncorr + 1].mean(axis=1)

    imax = int(cp.argmax(bestCC).item())
    indsort = cp.argsort(-cc[imax])
    top_indices = indsort[:ncorr]

    # -----------------------------
    # Step 6: average at full resolution
    # -----------------------------
    refImg = frames[top_indices].mean(axis=0)

    return refImg, top_indices


def compute_reference_from_block(mem_block, config, ca_block=None):
    """Generate a reference image from a block of frames"""
    k = config["channels"]["k"]
    function = config["channels"]["function"]
    dual_channels = config["channels"]["dual_channel"]
    frames = min(len(mem_block) // 2, 50)
    mem_block = cp.asarray(mem_block)
    mem_ref, indsort = pick_initial_reference(mem_block, max_corr_frames=frames)
    if dual_channels:
        if k != 0:
            ca_block = cp.asarray(ca_block)
            Ca_ref = cp.mean(ca_block[indsort, :], axis=0)
            Ca_ref_transform = transform(Ca_ref, k, function)
            result = mem_ref + Ca_ref_transform
        else:
            result = mem_ref
    else:
        result = mem_ref

    # Return as NumPy array - use .get() for CuPy, or return directly for NumPy
    return result.get() if hasattr(result, "get") else result, indsort.get() if hasattr(
        indsort, "get"
    ) else indsort
