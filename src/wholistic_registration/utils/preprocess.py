"""

version : 0.1
file name: preprocess.py

Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Last Update Date : 2025/8/05

Overview:
    This module provides functions for preprocessing 3D volumetric data, including normalization, smoothness penalty factor calculation, robust mean and standard deviation computation, and artificial motion generation.

functions:
    - auto_contrast(img, low_percentile=3, high_percentile=97): Applies automatic contrast adjustment to an image based on specified percentiles.
    - getSmPnltNormFctr(dat_ref, option): Calculates the smoothness penalty factor based on the gradients in the X and Y directions of the reference data.
    - robust_mean_std(data, percentile=95): Computes the mean and standard deviation of the lowest specified percentile of pixel values in an image.
    - normalize_to_255(img, lower_percentile=3, upper_percentile=99): Normalizes an image to the range [0, 255] based on specified percentiles.
    - generate_artificial_motion(image, art_R, Amp_art, zRatio, noise_level): Generates artificial motion in a 3D image and applies Gaussian noise, returning both the moved and reference images along with the true motion field.

"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates


def auto_contrast(img, low_percentile=3, high_percentile=97):
    """
    Applies automatic contrast adjustment to an image based on specified percentiles.
    Parameters:
        img (np.ndarray): Input image array.
        low_percentile (float): Lower percentile for contrast stretching. Default is 3.
        high_percentile (float): Upper percentile for contrast stretching. Default is 97.
    Returns:
        np.ndarray: Contrast-stretched image.
    """
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [low_percentile, high_percentile])
    img_clipped = np.clip(img, p_low, p_high)
    return (img_clipped - p_low) / (p_high - p_low + 1e-8)


def getSmPnltNormFctr(dat_ref, option):
    """
    Calculate the smoothness penalty factor based on the gradients in the X and Y directions.

    Args:
        dat_ref (np.ndarray): The reference data with shape (z, x, y), where z, x, y are the dimensions.
        option (dict): Contains various options, including the mask_ref.

    Returns:
        float: The smoothness penalty factor.
    """
    # Calculate the X and Y gradients using central difference
    Ix = (dat_ref[2:, :, :] - dat_ref[:-2, :, :]) / 2
    Iy = (dat_ref[:, 2:, :] - dat_ref[:, :-2, :]) / 2

    # Apply the mask to exclude certain areas from the calculation
    mask_ref = option["mask_ref"]

    # Calculate the mean squared gradients while excluding the masked areas
    Ix_squared = np.mean(Ix[~mask_ref[1:-1, :, :]] ** 2)
    Iy_squared = np.mean(Iy[~mask_ref[:, 1:-1, :]] ** 2)

    # Return the average of the squared gradients in both directions
    factor = (Ix_squared + Iy_squared) / 2
    return factor


def robust_mean_std(data, percentile=95):
    """
    Computes the mean and standard deviation of the lowest specified percentile of pixel values in an image.
    Parameters:
        data (np.ndarray): Input image array.
        percentile (float): Percentile threshold to consider for mean and std calculation. Default is 95.
    Returns:
        mean (float): Mean of the pixel values below the specified percentile.
        std (float): Standard deviation of the pixel values below the specified percentile.
    """
    data_flat = data.flatten()
    threshold = np.percentile(data_flat, percentile)
    mask = data_flat <= threshold
    selected = data_flat[mask]

    return np.mean(selected), np.std(selected)


def normalize_to_255(img, lower_percentile=3, upper_percentile=99):
    """
    Normalizes an image to the range [0, 255] based on specified percentiles.
    Parameters:
        img (np.ndarray): Input image array.
        lower_percentile (float): Lower percentile for normalization. Default is 3.
        upper_percentile (float): Upper percentile for normalization. Default is 99.
    Returns:
        norm_img.astype(np.ndarray): Normalized image in the range [0, 255].
    """
    lower = np.percentile(img, lower_percentile)
    upper = np.percentile(img, upper_percentile)

    clipped = np.clip(img, lower, upper)

    norm_img = (clipped - lower) / (upper - lower + 1e-8) * 255
    return norm_img.astype(np.float64)


def learn_quantile_mapping(
    source,
    target,
    percentiles=None,
    max_samples=5_000_000,
    seed=0,
):
    """
    Learn a monotonic intensity mapping from source distribution to target distribution.

    source: ref crop, e.g. (20, 1500, 630)
    target: mov crop, e.g. (20, 1500, 630)

    Returns:
        src_q: source quantile anchors
        tgt_q: target quantile anchors
    """
    if percentiles is None:
        percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.8]

    source = np.asarray(source)
    target = np.asarray(target)

    src_vals = source.reshape(-1).astype(np.float32)
    tgt_vals = target.reshape(-1).astype(np.float32)

    src_vals = src_vals[np.isfinite(src_vals)]
    tgt_vals = tgt_vals[np.isfinite(tgt_vals)]

    rng = np.random.default_rng(seed)

    if src_vals.size > max_samples:
        src_vals = rng.choice(src_vals, size=max_samples, replace=False)

    if tgt_vals.size > max_samples:
        tgt_vals = rng.choice(tgt_vals, size=max_samples, replace=False)

    src_q = np.percentile(src_vals, percentiles).astype(np.float32)
    tgt_q = np.percentile(tgt_vals, percentiles).astype(np.float32)

    # np.interp 要求 x 坐标单调递增且最好没有重复
    src_q_unique, unique_idx = np.unique(src_q, return_index=True)
    tgt_q_unique = tgt_q[unique_idx]

    return src_q_unique, tgt_q_unique, np.asarray(percentiles)[unique_idx]


def apply_quantile_mapping(
    source,
    src_q,
    tgt_q,
    out_dtype=np.float32,
    chunk_size=8,
):
    """
    Apply learned source -> target intensity mapping.

    source:
        full ref image, shape can be (Z,Y,X) or any array whose first axis is chunked.

    output:
        mapped source, same shape, dtype=out_dtype
    """
    source = np.asarray(source)

    out = np.empty(source.shape, dtype=out_dtype)

    n = source.shape[0]

    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)

        block = source[i0:i1].astype(np.float32, copy=False)

        mapped = np.interp(
            block.reshape(-1),
            src_q,
            tgt_q,
            left=tgt_q[0],
            right=tgt_q[-1],
        ).reshape(block.shape)

        if np.issubdtype(np.dtype(out_dtype), np.integer):
            info = np.iinfo(out_dtype)
            mapped = np.clip(mapped, info.min, info.max)
            mapped = np.rint(mapped)

        out[i0:i1] = mapped.astype(out_dtype)

    return out


def generate_artificial_motion(image, art_R, Amp_art, zRatio, noise_level):
    """
    Generates artificial motion in a 3D image and applies Gaussian noise, returning both the moved and reference images along with the true motion field.

    Parameters:
        image (np.ndarray): The original 3D image data.
        art_R (float): The radius for Gaussian smoothing of the motion field.
        Amp_art (float): The amplitude of the artificial motion.
        zRatio (float): The ratio of Z-axis spacing to X/Y-axis spacing.
        noise_level (float): The standard deviation of the Gaussian noise to be added.

    Returns:
        dat_mov (np.ndarray): The moved image with artificial motion and noise.
        dat_ref (np.ndarray): The reference image after applying the inverse motion and noise.
        motion_current_real (np.ndarray): The true motion field applied to the image.
    """
    Y, X, Z = image.shape
    N = X * Y * Z

    # generate sparse coordinates for motion perturbation
    ratio_art = 1.0 / ((2 * art_R + 1) ** 3)
    L = int(round(N * ratio_art))
    cp_art = np.random.choice(N, L, replace=False)
    coords = np.unravel_index(cp_art, (Y, X, Z))

    # generate sparse coordinates along with Gaussian smoothing
    def generate_field():
        field = np.zeros((Y, X, Z), dtype=np.float32)
        field[coords] = np.random.randn(L)
        return gaussian_filter(field, sigma=[art_R, art_R, art_R / zRatio])

    motion_X = generate_field()
    motion_Y = generate_field()
    motion_Z = generate_field()

    # normalize the motion field at the perturbation coordinates to have the specified amplitude
    std_X = np.std(motion_X[coords])
    std_Y = np.std(motion_Y[coords])
    std_Z = np.std(motion_Z[coords])

    motion_X = motion_X / std_X * Amp_art
    motion_Y = motion_Y / std_Y * Amp_art
    motion_Z = motion_Z / std_Z * Amp_art

    # convert to the motion field with shape (Y, X, Z, 3)
    motion_current_real = np.stack([motion_X, motion_Y, motion_Z], axis=-1)

    # apply the motion to the image using interpolation
    def apply_motion(img, motion):
        # generate original grid coordinates
        yy, xx, zz = np.meshgrid(np.arange(Y), np.arange(X), np.arange(Z), indexing="ij")
        coords_warped = np.array([yy + motion[..., 1], xx + motion[..., 0], zz + motion[..., 2]])
        # project the warped coordinates back to the original image space using interpolation
        return map_coordinates(img, coords_warped, order=1, mode="nearest")

    # Remind the motion: dat_mov is the result after adding motion (forward), while correctMotion_Wei is reverse sampling
    dat_mov_raw = apply_motion(image, -motion_current_real)
    dat_ref_raw = apply_motion(dat_mov_raw, motion_current_real)

    # Add Gaussian noise to both moved and reference images
    dat_mov = dat_mov_raw + np.random.randn(Y, X, Z) * noise_level
    dat_ref = dat_ref_raw + np.random.randn(Y, X, Z) * noise_level

    return dat_mov, dat_ref, motion_current_real


def michelson_edge_map(frame, sigma_xy=3, r=6, eps=1e-6):
    # frame: (Y,X), float in [0,1]
    # 1) denoise a bit
    fs = ndi.gaussian_filter(frame, sigma=sigma_xy, mode="nearest")
    # 2) gradients and unit normal
    gx = ndi.sobel(fs, axis=-1, mode="nearest")
    gy = ndi.sobel(fs, axis=-2, mode="nearest")
    nrm = np.sqrt(gx * gx + gy * gy) + eps
    ux, uy = gx / nrm, gy / nrm  # edge normal direction
    # 3) sample intensities on two sides along the normal
    H, W = frame.shape
    y, x = np.mgrid[0:H, 0:W]
    y_plus = np.clip(y + r * uy, 0, H - 1)
    x_plus = np.clip(x + r * ux, 0, W - 1)
    y_minus = np.clip(y - r * uy, 0, H - 1)
    x_minus = np.clip(x - r * ux, 0, W - 1)
    I_plus = map_coordinates(fs, [y_plus, x_plus], order=1, mode="nearest")
    I_minus = map_coordinates(fs, [y_minus, x_minus], order=1, mode="nearest")
    # 4) Michelson-normalized edge strength (absolute value)
    E = (I_plus - I_minus) / (I_plus + I_minus + eps)

    return np.abs(E)


def normalize_std(mean, std, image: np.ndarray):
    mean_prev = np.mean(image)
    std_prev = np.std(image)
    image_normalized = (image - mean_prev) / std_prev
    image_corrected = image_normalized * std + mean
    return image_corrected
