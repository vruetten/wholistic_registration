# reliablemask.py
import os
import re

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from . import IO, cp, cupy_ndimage


def write_multichannel_volume_as_ome_tiff(
    vol3d_list, out_dir, frame_idx, configPath=None, spacing_x=1.0, spacing_y=1.0
):
    """
    vol3d_list: list of 3 arrays, each (Z,Y,X)
        ch0, ch1, ch2

    output OME TIFF shape: (1, Z, 3, Y, X)
    """

    assert len(vol3d_list) == 3, "vol3d_list must contain exactly 3 volumes."

    processed = []
    for v in vol3d_list:
        if v.ndim == 2:
            v = v[np.newaxis, :, :]
        if v.dtype == bool:
            v = v.astype(np.uint8)
        elif v.dtype not in [np.uint8, np.float32]:
            v = v.astype(np.float32)
        processed.append(v)

    Z, Y, X = processed[0].shape
    img5d = np.stack(processed, axis=0)  # (3, Z, Y, X)
    img5d = img5d[np.newaxis, :, :, :, :]  # (1,3,Z,Y,X)
    img5d = np.transpose(img5d, (0, 2, 1, 3, 4))  # → (1,Z,3,Y,X)

    fname = os.path.join(out_dir, f"vol_{frame_idx:06d}_masked.tif")

    metadata = {"spacing_x": spacing_x, "spacing_y": spacing_y, "data_shape": img5d.shape}

    IO.saveTiff_new(img5d, fname, config_path=configPath, metadata=metadata, verbose=False)


# pre version
def local_ssim_difference(I_ref, I_mov, win_size=11, use_3d=False, sigma_3d=1.5):
    """
    Compute local SSIM difference map (0~1) between two images.
    Supports 2D and 3D images.

    Parameters
    ----------
    I_ref : np.ndarray
        Reference image (2D or 3D).
    I_mov : np.ndarray
        Registered/moving image (must match shape of I_ref).
    win_size : int
        SSIM window size for 2D (odd number like 11, 21).
    use_3d : bool
        If True, compute a true 3D SSIM approximation using Gaussian smoothing.
        If False, compute 2D SSIM slice-by-slice for 3D volumes (recommended for microscopy).
    sigma_3d : float or tuple
        Gaussian sigma used in 3D SSIM approximation mode.

    Returns
    -------
    D : np.ndarray, float32
        Difference map in [0,1], same shape as input.
    """
    I_ref = I_ref.astype(np.float32)
    I_mov = I_mov.astype(np.float32)

    if I_ref.ndim not in [2, 3]:
        raise ValueError("Input images must be 2D or 3D numpy arrays.")

    if I_ref.shape != I_mov.shape:
        raise ValueError("Input images must have the same shape.")
    # -----------------------
    # Case 1: 2D Image
    # -----------------------
    if I_ref.ndim == 2:
        _, ssim_map = ssim(
            I_ref,
            I_mov,
            win_size=win_size,
            gaussian_weights=True,
            data_range=I_ref.max() - I_ref.min(),
            full=True,
        )
        D = (1 - ssim_map) / 2.0
        return np.clip(D.astype(np.float32), 0, 1)

    # -----------------------
    # Case 2: 3D Image
    # -----------------------
    if not use_3d:
        # --- Slice-wise 2D SSIM (recommended for microscopy with low Z-resolution)
        Z = I_ref.shape[0]
        D = np.zeros_like(I_ref, dtype=np.float32)

        for z in range(Z):
            _, ssim_map = ssim(
                I_ref[z],
                I_mov[z],
                win_size=win_size,
                gaussian_weights=True,
                data_range=I_ref[z].max() - I_ref[z].min(),
                full=True,
            )
            D[z] = (1 - ssim_map) / 2.0

        return np.clip(D, 0, 1)

    else:
        # --- True 3D SSIM approximation using Gaussian filters
        #     (Useful only if Z-resolution is comparable to XY)

        C1 = (0.01 * (I_ref.max() - I_ref.min())) ** 2
        C2 = (0.03 * (I_ref.max() - I_ref.min())) ** 2

        mu_x = gaussian_filter(I_ref, sigma=sigma_3d)
        mu_y = gaussian_filter(I_mov, sigma=sigma_3d)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = gaussian_filter(I_ref * I_ref, sigma=sigma_3d) - mu_x2
        sigma_y2 = gaussian_filter(I_mov * I_mov, sigma=sigma_3d) - mu_y2
        sigma_xy = gaussian_filter(I_ref * I_mov, sigma=sigma_3d) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = numerator / (denominator + 1e-12)
        D = (1 - ssim_map) / 2

        return np.clip(D.astype(np.float32), 0, 1)


# pre version
def local_mind_difference(
    I_ref,
    I_mov,
    patch_sigma=1.5,
    offset_radius=5,
    structure_tau=0.6,
    structure_beta=0.05,
    eps=1e-6,
    debug_dir=None,
):
    """
    GPU MIND-based local misalignment map with explicit masking of background.

    Output semantics:
        - 0 → background or perfectly aligned
        - larger → worse local registration (misalignment)
        - completely ignores areas with no structure

    Supports 2D or 3D (slice-wise) images.

    Parameters
    ----------
    debug_dir : str or None
        If set, save intermediate results (M_ref, diff, weight, diff_weighted)
        as TIFF files in this directory.
    """

    # ---------------------------
    # helpers
    # ---------------------------
    def _mind_offsets_2d(radius):
        return [
            (radius, 0),
            (-radius, 0),
            (0, radius),
            (0, -radius),
            (radius, radius),
            (radius, -radius),
            (-radius, radius),
            (-radius, -radius),
        ]

    def _mind_descriptor_2d(I):
        I = cp.asarray(I, dtype=cp.float32)
        offsets = _mind_offsets_2d(offset_radius)
        H, W = I.shape
        K = len(offsets)

        # smooth image to suppress noise
        I_s = cupy_ndimage.gaussian_filter(I, patch_sigma)
        D = cp.empty((K, H, W), dtype=cp.float32)

        for k, (dy, dx) in enumerate(offsets):
            I_shift = cp.roll(I_s, shift=(dy, dx), axis=(0, 1))
            diff2 = (I_s - I_shift) ** 2
            D[k] = cupy_ndimage.gaussian_filter(diff2, patch_sigma)

        V = cp.mean(D, axis=0) + eps
        MIND = cp.exp(-D / V[None])
        return MIND

    def _soft_structure_weight(S, tau, beta):
        """
        Sigmoid soft structure mask
        """
        return 1.0 / (1.0 + cp.exp(-(S - tau) / beta))

    def _mind_diff_2d(Ir, Im):
        M_ref = _mind_descriptor_2d(Ir)
        M_mov = _mind_descriptor_2d(Im)

        diff = cp.abs(cp.mean(M_ref - M_mov, axis=0))

        structure_mov = cp.mean(M_mov, axis=0)
        structure_mov = cupy_ndimage.gaussian_filter(structure_mov, sigma=patch_sigma)
        weight_mov = _soft_structure_weight(structure_mov, structure_tau, structure_beta)
        structure_ref = cp.mean(M_ref, axis=0)
        structure_ref = cupy_ndimage.gaussian_filter(structure_ref, sigma=patch_sigma)
        weight_ref = _soft_structure_weight(structure_ref, structure_tau, structure_beta)
        weight = cp.maximum(weight_ref, weight_mov)

        diff = diff * weight

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            # arrays are numpy under the cp fallback, which has no .get()
            tonp = lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
            tifffile.imwrite(os.path.join(debug_dir, "M_ref.tif"), tonp(M_ref))
            tifffile.imwrite(
                os.path.join(debug_dir, "diff_raw.tif"),
                tonp(cp.abs(cp.mean(M_ref - M_mov, axis=0))),
            )
            tifffile.imwrite(os.path.join(debug_dir, "weight_map.tif"), tonp(weight))
            tifffile.imwrite(os.path.join(debug_dir, "diff_weighted.tif"), tonp(diff))

        return cp.clip(diff, 0.0, 1.0)

    # ---------------------------
    # main
    # ---------------------------
    I_ref = cp.asarray(I_ref)
    I_mov = cp.asarray(I_mov)

    if I_ref.shape != I_mov.shape:
        raise ValueError("[ERROR] I_ref and I_mov must have the same shape")

    if I_ref.ndim == 2:
        return _mind_diff_2d(I_ref, I_mov)

    elif I_ref.ndim == 3:
        Z, H, W = I_ref.shape
        diff = cp.zeros((Z, H, W), dtype=cp.float32)
        for z in range(Z):
            diff[z] = _mind_diff_2d(I_ref[z], I_mov[z])
        return diff

    else:
        raise ValueError("Only 2D or 3D images are supported")


# pre version
def local_zscore_difference(
    I_ref,
    I_mov,
    sigma=1.5,
    eps=1e-6,
    p_mu=20,
    p_var=40,
    clip=(0, 10),
    debug_dir=None,
):
    """
    Local z-score based difference map between reference and moving images.

    Parameters
    ----------
    debug_dir : str or None
        If set, save intermediate results (var_ref, mask_ref, mask_mov, mask)
        as TIFF files in this directory.
    """
    I_ref = I_ref.astype(np.float32)
    I_mov = I_mov.astype(np.float32)

    mu_ref = gaussian_filter(I_ref, sigma=sigma)
    mu_mov = gaussian_filter(I_mov, sigma=sigma)

    var_ref = gaussian_filter(I_ref**2, sigma=sigma) - mu_ref**2
    var_mov = gaussian_filter(I_mov**2, sigma=sigma) - mu_mov**2
    # float32 cancellation can make these slightly negative (NaN after sqrt)
    var_ref[var_ref < 0] = 0
    var_mov[var_mov < 0] = 0

    threshold_mu_ref = np.percentile(mu_ref, p_mu)
    threshold_var_ref = np.percentile(var_ref, p_var)
    mask_ref = (mu_ref > threshold_mu_ref) & (var_ref > threshold_var_ref)
    threshold_mu_mov = np.percentile(mu_mov, p_mu)
    threshold_var_mov = np.percentile(var_mov, p_var)
    mask_mov = (mu_mov > threshold_mu_mov) & (var_mov > threshold_var_mov)
    mask = mask_ref | mask_mov

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(debug_dir, "var_ref.tif"), var_ref)
        tifffile.imwrite(os.path.join(debug_dir, "mask_ref.tif"), mask_ref.astype(np.uint8))
        tifffile.imwrite(os.path.join(debug_dir, "mask_mov.tif"), mask_mov.astype(np.uint8))
        tifffile.imwrite(os.path.join(debug_dir, "mask.tif"), mask.astype(np.uint8))

    denom = np.sqrt(var_ref + var_mov) + eps

    D = np.abs(mu_ref - mu_mov) / denom * mask
    if clip is not None:
        D = np.clip(D, clip[0], clip[1])
        D = D / (clip[1] - clip[0])
    return D.astype(np.float32)


# pre version
def reliability_map(I, sigma=1.5, eps=1e-6, outlier_sigma=5.0):
    I = I.astype(np.float32)
    mu = gaussian_filter(I, sigma)
    var = gaussian_filter(I**2, sigma) - mu**2
    var[var < 0] = 0
    scale = np.percentile(var, 95)
    if scale < eps:
        return np.zeros_like(I, dtype=np.float32)
    R_local = var / (scale + eps)
    R_local = np.clip(R_local, 0, 1)
    median = np.median(I)
    mad = np.median(np.abs(I - median))
    robust_std = 1.4826 * mad + eps
    z_robust = (I - median) / robust_std
    outlier_mask = np.abs(z_robust) > outlier_sigma
    R_local[outlier_mask] = 0.0
    return R_local.astype(np.float32)


def reliability_map_v2(template, sigma=1.5, eps=1e-6):
    """
    Continuous reliability map based on:
    - low gradient (structure stability)
    - sufficient intensity
    Output: float32 in [0,1]
    """
    template = cp.asarray(template, dtype=cp.float32)
    is_2d = template.ndim == 2

    def gradient_amplitude(volume):
        gy = cupy_ndimage.sobel(volume, axis=0)
        gx = cupy_ndimage.sobel(volume, axis=1)
        return cp.sqrt(gx**2 + gy**2)

    def gaussian_smooth_per_slice(arr, sigma):
        if arr.ndim == 2:
            return cupy_ndimage.gaussian_filter(arr, sigma=sigma)
        out = cp.empty_like(arr, dtype=cp.float32)
        for z in range(arr.shape[0]):
            out[z] = cupy_ndimage.gaussian_filter(arr[z], sigma=sigma)
        return out

    def weighted_quantile_cp(x, w, eps=1e-12):
        x = x.ravel()
        w = cp.clip(w.ravel(), 0, None)

        idx = cp.argsort(x)
        x_sorted = x[idx]
        w_sorted = w[idx]

        cw = cp.cumsum(w_sorted)
        cw = cw / (cw[-1] + eps)

        j = cp.searchsorted(cw, cp.asarray(0.5, dtype=cp.float32))
        j = cp.clip(j, 0, x_sorted.size - 1)
        return x_sorted[j]

    # gradient amplitude
    Iamp = gradient_amplitude(template)
    # gradient-based reliability
    Iamp_mu = cp.median(Iamp)
    Iamp_sigma = cp.std(Iamp) + eps
    Iz = (Iamp - Iamp_mu) / Iamp_sigma
    Iz_sm = gaussian_smooth_per_slice(Iz, sigma=sigma)
    tau_g = cp.std(Iz_sm) + eps
    R_grad = cp.exp(-(Iz_sm**2) / (2 * tau_g**2))
    R_grad = R_grad / (cp.max(R_grad) + eps)
    # intensity-based reliability
    g_mad = cp.median(cp.abs(Iamp - Iamp_mu)) + eps
    g_robust = (Iamp - Iamp_mu) / (1.4826 * g_mad + eps)

    w_bg = cp.exp(-(cp.clip(g_robust, -5, 5) ** 2) / 2.0)
    p_low = weighted_quantile_cp(template, w_bg)
    p_hi = cp.percentile(template, 99)

    R_int = (template - p_low) / (p_hi - p_low + eps)
    R_int = cp.clip(R_int, 0, 1)

    # final reliability
    R = R_grad * R_int
    R = gaussian_smooth_per_slice(R, sigma=sigma)
    R = R / (cp.max(R) + eps)

    # cp may be the numpy fallback, which has no asnumpy
    return R.get() if hasattr(R, "get") else np.asarray(R)


# pre version
def photometric_align_robust(I_ref, I_mov, R, r_threshold=0.3, eps=1e-6):
    valid = r_threshold < R

    x = I_ref[valid].flatten()
    y = I_mov[valid].flatten()
    if len(x) < 1000:
        x = I_ref.flatten()
        y = I_mov.flatten()
    a, b = np.polyfit(x, y, 1)
    I_mov_corr = (I_mov - b) / (a + eps)
    return I_mov_corr.astype(np.float32)


def photometric_align_hist(I_ref, I_mov):
    from skimage.exposure import match_histograms

    return match_histograms(I_mov, I_ref).astype(np.float32)


def structural_difference_map(
    I_ref,
    I_mov,
    sigma_structure=4.0,
    sigma_reliability=1.5,
    r_threshold=0.2,
    debug_dir=None,
):
    """
    Compute a structure-weighted difference map between reference and moving images.

    Parameters
    ----------
    debug_dir : str or None
        If set, save intermediate results (R, diff, D) as TIFF files in this directory.
    """
    I_ref = I_ref.astype(np.float32)
    I_mov = I_mov.astype(np.float32)

    R_ref = reliability_map_v2(I_ref, sigma=sigma_reliability)
    R_mov = reliability_map_v2(I_mov, sigma=sigma_reliability)

    # Use minimum — only trust difference where BOTH images have structure.
    # np.maximum would flag background regions where only one image has structure,
    # producing false positives from intensity differences unrelated to misregistration.
    R = np.maximum(R_ref, R_mov)  # minimum?...
    I_mov_corr = photometric_align_hist(I_ref, I_mov)
    if len(I_ref.shape) == 2:
        mu_ref = gaussian_filter(I_ref, sigma_structure)
        mu_mov = gaussian_filter(I_mov_corr, sigma_structure)
    else:
        mu_ref = np.empty_like(I_ref, dtype=np.float32)
        mu_mov = np.empty_like(I_mov_corr, dtype=np.float32)

        for z in range(I_ref.shape[0]):
            mu_ref[z] = gaussian_filter(I_ref[z], sigma=sigma_structure)
            mu_mov[z] = gaussian_filter(I_mov_corr[z], sigma=sigma_structure)

    diff = np.abs(mu_ref - mu_mov)

    valid_vals = diff[r_threshold < R]

    if len(valid_vals) > 100:
        scale = np.percentile(valid_vals, 99)
    else:
        scale = np.percentile(diff, 99)

    Z = diff / scale
    D = 1.0 - np.exp(-(Z**2))

    D_final = D * R

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(debug_dir, "R_reliability.tif"), R)
        tifffile.imwrite(os.path.join(debug_dir, "diff_raw.tif"), diff)
        tifffile.imwrite(os.path.join(debug_dir, "D_difference.tif"), D)

    return D_final.astype(np.float32), D.astype(np.float32), R.astype(np.float32)


def build_reference_index(ref_dir):
    """
    scan reference folder, construct frame -> filepath
    return:
        ref_map: dict(frame -> filepath)
        ref_files: list of all files
    """
    ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".tif")]
    ref_map = {}

    for f in ref_files:
        m_multi = re.match(r"vol_ref_(\d{6})_(\d{6})\.tif", f)
        m_single = re.match(r"vol_ref_(\d{6})\.tif", f)

        if m_multi:
            a = int(m_multi.group(1))
            b = int(m_multi.group(2))
            for t in range(a, b + 1):
                ref_map[t] = os.path.join(ref_dir, f)

        elif m_single:
            a = int(m_single.group(1))
            ref_map[a] = os.path.join(ref_dir, f)

        else:
            print(f"[WARNING] Unknown reference filename format: {f}")

    return ref_map, ref_files


# pre version
def ComputeMask(
    mem_dir,
    ca_dir,
    ref_dir,
    out_dir,
    dual_channel,
    frames,
    config,
    compute_cor_fn,
    configPath,
):
    """
    Computes spatial, temporal, and accumulative reliability masks.

    For each frame:
        1. Reads registered membrane and calcium images
        2. Computes correlation map
        3. Compares with reference image using SSIM
        4. Saves resulting mask as OME-TIFF

    Parameters:
    -----------
    mem_dir : str
        Directory containing registered membrane channel images
    ca_dir : str
        Directory containing registered calcium channel images
    ref_dir : str
        Directory containing reference images
    out_dir : str
        Directory where output masks will be saved
    config : dict
        Configuration parameters for reliable analysis
    compute_cor_fn : callable
        Function to compute correlation map from membrane and calcium channels
    configPath : str
        Path to the main configuration file
    downsampleXY : int, optional
        Downsampling factor for XY dimensions (default: 1)
    downsampleT : int, optional
        Downsampling factor for temporal dimension (default: 1)

    Returns:
    --------
    None

    Output:
    -------
    Creates the following directory structure under out_dir:
    """

    # Create output directories
    mask_ds_dir = out_dir  # Downsampled masks directory

    os.makedirs(mask_ds_dir, exist_ok=True)

    # Build reference image index
    ref_map, ref_files = build_reference_index(ref_dir)

    # Process each frame
    for i in frames:
        if i % 100 == 0:
            print(f"Processed {i}/{frames[-1]} frames")

        # Read registered images
        mem_i = IO.read_reg_tiff(mem_dir, i, 1)  # Channel 1: membrane
        if dual_channel:
            ca_i = IO.read_reg_tiff(ca_dir, i, 0)  # Channel 0: calcium
        else:
            ca_i = np.zeros_like(mem_i)
        # Compute correlation map
        cor_i = compute_cor_fn(mem_i, ca_i)

        # Read corresponding reference image
        ref_i = tifffile.imread(ref_map[i])

        # Extract configuration parameters
        # win_size = config['win_size']      # Window size for SSIM computation
        # use_3d = config['use_3d']          # Whether to use 3D SSIM
        # sigma_3d = config['sigma_3d']      # Sigma for 3D Gaussian blur

        # Compute reliability mask using SSIM difference
        # mask_map = local_gradient_misalignment(cor_i, ref_i)
        mask_map = local_zscore_difference(
            ref_i,
            cor_i,
        )
        if isinstance(mask_map, np.ndarray):
            # Save downsampled mask
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[mask_map],  # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label="mask",
            )
        else:
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[mask_map.get()],  # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label="mask",
            )


def ComputeMask_v2(
    mem_dir,
    ca_dir,
    ref_dir,
    out_dir,
    dual_channel,
    frames,
    config,
    compute_cor_fn,
    configPath,
):
    """
    Computes spatial, temporal, and accumulative reliability masks.

    For each frame:
        1. Reads registered membrane and calcium images
        2. Computes correlation map
        3. Compares with reference image using SSIM
        4. Saves resulting mask as OME-TIFF

    Parameters:
    -----------
    mem_dir : str
        Directory containing registered membrane channel images
    ca_dir : str
        Directory containing registered calcium channel images
    ref_dir : str
        Directory containing reference images
    out_dir : str
        Directory where output masks will be saved
    config : dict
        Configuration parameters for reliable analysis
    compute_cor_fn : callable
        Function to compute correlation map from membrane and calcium channels
    configPath : str
        Path to the main configuration file
    downsampleXY : int, optional
        Downsampling factor for XY dimensions (default: 1)
    downsampleT : int, optional
        Downsampling factor for temporal dimension (default: 1)

    Returns:
    --------
    None

    Output:
    -------
    Creates the following directory structure under out_dir:
    """

    # Create output directories
    mask_ds_dir = out_dir  # Downsampled masks directory

    IO.reset_dir(mask_ds_dir)

    # Build reference image index
    ref_map, ref_files = build_reference_index(ref_dir)
    ref_files = sorted(ref_files)  # Ensure files are processed in order

    # Process each frame
    for i in range(len(ref_files) - 1):
        print(f"Compute difference between {ref_files[i]} and {ref_files[i + 1]} ")

        ref_pre = tifffile.imread(os.path.join(ref_dir, ref_files[i]))
        ref_post = tifffile.imread(os.path.join(ref_dir, ref_files[i + 1]))

        mask_map, diff_map, rely_map = structural_difference_map(
            ref_pre,
            ref_post,
        )
        if isinstance(mask_map, np.ndarray):
            # Save downsampled mask
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[ref_pre, ref_post, mask_map, diff_map, rely_map],  # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label="mask",
            )
        else:
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[
                    ref_pre,
                    ref_post,
                    mask_map.get(),
                    diff_map.get(),
                    rely_map.get(),
                ],  # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label="mask",
            )
