import os

import numpy as np
from skimage.metrics import structural_similarity as ssim

from . import IO, calFlow3d_Wei_v1, cp, cupy_ndimage


def generateMotion(raw_data, art_R, amp_art, zRatio=1):
    """
    generate simulation(amp and r)

    Parameters:
        res_path_name: Path to result directory
        reader: Metadata reader (must implement read_meta method)
        art_R: Artifact-related parameter
        amp_art: Amplitude parameter
        *args: Optional arguments, used to provide zRatio

    Returns:
        motion_X, motion_Y, motion_Z: Motion data arrays
        cp_art: Indices of randomly selected points
    """

    # Calculate filter sigma values for 3D Gaussian filtering
    filter_sigma = np.array([art_R, art_R, art_R / zRatio])

    [X, Y, Z] = raw_data.shape
    N = X * Y * Z  # Total number of points in the 3D volume

    # Calculate number of points to select
    ratio_art = 1 / ((art_R * 2 + 1) ** 2)
    L = round(N * ratio_art)  # Number of points to select

    # Randomly select L points using permutation
    cp_art = np.random.permutation(N)[:L]

    # Initialize motion arrays with single precision
    motion_X = np.zeros((X, Y, Z), dtype=np.float32)
    motion_Y = np.zeros((X, Y, Z), dtype=np.float32)
    motion_Z = np.zeros((X, Y, Z), dtype=np.float32)

    # Add Gaussian noise to randomly selected points
    motion_X.flat[cp_art] = np.random.randn(L)
    motion_Y.flat[cp_art] = np.random.randn(L)
    # motion_Z.flat[cp_art] = np.random.randn(L)

    # Apply 3D Gaussian filtering
    motion_X = cupy_ndimage.gaussian_filter(motion_X, sigma=filter_sigma)
    motion_Y = cupy_ndimage.gaussian_filter(motion_Y, sigma=filter_sigma)
    # motion_Z = cupy_ndimage.gaussian_filter(motion_Z, sigma=filter_sigma)  # Commented out as in original code

    # Calculate scaling factors based on standard deviation
    # Note: Using motion_Y's standard deviation for both X and Y scaling
    # Preserving original behavior even if potentially a typo
    factor_X = np.std(motion_Y.flat[cp_art])
    factor_Y = np.std(motion_Y.flat[cp_art])
    # factor_Z = np.std(motion_Z.flat[cp_art])

    # Apply amplitude scaling
    motion_X = motion_X / factor_X * amp_art
    motion_Y = motion_Y / factor_Y * amp_art
    # motion_Z = motion_Z / factor_Z * amp_art

    return motion_X, motion_Y, motion_Z, cp_art


def limit_gradient_3d(motion, max_grad):
    """
    限制 3D 位移场的梯度范数，防止 folding / 纹理折断
    """
    gx = np.gradient(motion, axis=0)
    gy = np.gradient(motion, axis=1)
    gz = np.gradient(motion, axis=2)

    grad_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    scale = np.maximum(1.0, grad_norm / max_grad)

    return motion / scale


def generateMotion_Biophysical(
    shape,
    art_R_xy=120,
    art_R_z=6,
    amp_xy=8,
    amp_z=2,
    zRatio=3.0,
    coupling=0.3,
    use_incompressibility=False,
    incompressibility_smooth_sigma=(3, 3, 2),
    center_zero_mean_z=False,
    eps=1e-8,
):
    """
    Generate a smooth 3D biophysical-inspired motion field.

    Parameters
    ----------
    shape : tuple
        (X, Y, Z)
    art_R_xy : float
        Gaussian smoothing sigma in x/y for lateral motion.
    art_R_z : float
        Gaussian smoothing sigma in z for lateral motion.
    amp_xy : float
        Target 95th-percentile amplitude of lateral motion magnitude.
    amp_z : float
        Target 95th-percentile amplitude of axial motion magnitude.
    zRatio : float
        Physical z/xy resolution ratio. Currently kept as an argument for
        consistency with your pipeline. In this implementation, the
        incompressibility construction is performed in index space.
    coupling : float
        Coupling strength used only when use_incompressibility=False.
    use_incompressibility : bool
        If True, construct motion_Z from near-incompressibility:
            dUz/dz ~= -(dUx/dx + dUy/dy)
        If False, use the original z-gradient coupling:
            Uz ~ dUx/dz + dUy/dz
    incompressibility_smooth_sigma : tuple
        Additional smoothing sigma applied to motion_Z.
    center_zero_mean_z : bool
        If True, subtract z-wise mean of motion_Z so that integrated drift
        does not accumulate excessively along z.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    motion : np.ndarray
        Shape (X, Y, Z, 3), in CPU numpy array.
    """
    X, Y, Z = shape

    # -------------------------
    # xy-dominant smooth deformation
    # -------------------------
    motion_X = cupy_ndimage.gaussian_filter(
        cp.random.randn(X, Y, Z), sigma=(art_R_xy, art_R_xy, art_R_z)
    )
    motion_Y = cupy_ndimage.gaussian_filter(
        cp.random.randn(X, Y, Z), sigma=(art_R_xy, art_R_xy, art_R_z)
    )

    scale_xy = cp.percentile(cp.sqrt(motion_X**2 + motion_Y**2), 95)
    scale_xy = cp.maximum(scale_xy, eps)

    motion_X = motion_X / scale_xy * amp_xy
    motion_Y = motion_Y / scale_xy * amp_xy

    # -------------------------
    # z motion
    # -------------------------
    if not use_incompressibility:
        # Original heuristic coupling:
        # Uz ~ dUx/dz + dUy/dz
        dXdz = cp.gradient(motion_X, axis=2)
        dYdz = cp.gradient(motion_Y, axis=2)

        motion_Z = coupling * (dXdz + dYdz)
        motion_Z = cupy_ndimage.gaussian_filter(motion_Z, sigma=incompressibility_smooth_sigma)
        # -------------------------
        # normalize z amplitude
        # -------------------------
        scale_z = cp.percentile(cp.abs(motion_Z), 95)
        scale_z = cp.maximum(scale_z, eps)
        motion_Z = motion_Z / scale_z * amp_z / zRatio
    else:
        # Near-incompressibility in index space:
        # dUx/dx + dUy/dy + dUz/dz ~= 0
        # => dUz/dz ~= -(dUx/dx + dUy/dy)

        dXdx = cp.gradient(motion_X, axis=0)
        dYdy = cp.gradient(motion_Y, axis=1)

        div_xy = dXdx + dYdy
        dZdz = -div_xy

        # Integrate along z to obtain Uz
        # current incompressible Uz
        motion_Z = cp.cumsum(dZdz, axis=2)

        if center_zero_mean_z:
            motion_Z = motion_Z - cp.mean(motion_Z, axis=2, keepdims=True)

        motion_Z = cupy_ndimage.gaussian_filter(motion_Z, sigma=incompressibility_smooth_sigma)

        # build smooth z-independent bias field
        Cxy = cupy_ndimage.gaussian_filter(cp.random.randn(X, Y), sigma=art_R_xy)
        Cxy = Cxy / (cp.std(Cxy) + eps)
        Cxy = Cxy[:, :, None]

        # choose target z RMS
        xy_rms = cp.sqrt(cp.mean(motion_X**2 + motion_Y**2))
        target_z_rms = xy_rms / zRatio

        current_z_rms = cp.sqrt(cp.mean(motion_Z**2))
        bias_rms = cp.sqrt(cp.mean(Cxy**2))

        # Since Uz_new = Uz + a*Cxy, pick a to raise RMS approximately
        needed_sq = cp.maximum(0, target_z_rms**2 - current_z_rms**2)
        a = cp.sqrt(needed_sq / (bias_rms**2 + eps))

        motion_Z = motion_Z + a * Cxy
        # motion_Z = motion_Z / zRatio

    motion = cp.stack([motion_X, motion_Y, motion_Z], axis=3)
    return motion.get() if hasattr(motion, "get") else motion


def generate_single_simulated_data(
    original_data_path,
    frame,
    crop_region,
    r_value,
    amp_value,
    noise_level,
):
    # crop region
    x_start, y_start, z_start, x_size, y_size, z_size = crop_region
    crop_range_x = slice(x_start, x_start + x_size)
    crop_range_y = slice(y_start, y_start + y_size)
    crop_range_z = slice(z_start, z_start + z_size)

    # load the data
    print("Loading raw data...")
    meta = IO.readMeta(original_data_path)
    dat_org = IO.readFrame(original_data_path, frame, 1)

    # zRatio
    channels = meta.channels[0]
    axesCalibration = channels.volume.axesCalibration
    zRatio = axesCalibration[2] / axesCalibration[0]

    # generate motion(amp and r)
    print(f"generation motion (R={r_value}, Amp={amp_value})...")
    motion_x, motion_y, motion_z, _ = generateMotion(dat_org, r_value, amp_value, zRatio)

    motion_current_real = np.stack([motion_x, motion_y, motion_z], axis=3)

    print("apply motion")
    dat_mov_raw = calFlow3d_Wei_v1.correctMotion(dat_org, -motion_current_real)  # 应用负向运动
    dat_ref_raw = calFlow3d_Wei_v1.correctMotion(dat_mov_raw, motion_current_real)  # 校正回参考位置

    print(f"add noise (noise level: {noise_level})")
    dat_mov = dat_mov_raw + np.random.randn(*dat_org.shape) * noise_level
    dat_ref = dat_ref_raw + np.random.randn(*dat_org.shape) * noise_level

    print("crop the data to the given region")
    dat_mov_cropped = dat_mov[crop_range_x, crop_range_y, crop_range_z]
    dat_ref_cropped = dat_ref[crop_range_x, crop_range_y, crop_range_z]
    motion_cropped = motion_current_real[crop_range_x, crop_range_y, crop_range_z, :]

    crop_info = {
        "crop_range_x": np.arange(x_start, x_start + x_size),
        "crop_range_y": np.arange(y_start, y_start + y_size),
        "crop_range_z": np.arange(z_start, z_start + z_size),
    }

    return dat_mov_cropped, dat_ref_cropped, motion_cropped, crop_info


def crop_valid(arr, crop=50):
    if crop <= 0:
        return arr
    if arr.ndim == 3:
        return arr[crop:-crop, crop:-crop, :]
    elif arr.ndim == 4:
        return arr[crop:-crop, crop:-crop, :, :]
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}")


def compute_intensity_mse(pred, gt):
    return float(np.mean((pred - gt) ** 2))


def compute_intensity_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_volume_ssim(pred, gt, data_range=None):
    """
    Slice-wise SSIM average over z.
    pred, gt: shape (Y, X, Z)
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    if data_range is None:
        data_range = float(max(pred.max(), gt.max()) - min(pred.min(), gt.min()))
        if data_range == 0:
            data_range = 1.0

    vals = []
    for z in range(pred.shape[2]):
        vals.append(ssim(gt[:, :, z], pred[:, :, z], data_range=data_range))
    return float(np.mean(vals))


def compute_motion_metrics(pred, gt, z_ratio=1.0, use_z=True):
    """
    pred, gt: shape (Y, X, Z, 3)
    channel order assumed: (..., 0)=y, (...,1)=x, (...,2)=z
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    diff = pred - gt
    dy = diff[..., 0]
    dx = diff[..., 1]

    metrics = {}

    # XY metrics
    epe_xy = np.sqrt(dy**2 + dx**2)
    metrics["motion_epe_xy_mean"] = float(np.mean(epe_xy))
    metrics["motion_epe_xy_median"] = float(np.median(epe_xy))
    metrics["motion_rmse_xy"] = float(np.sqrt(np.mean(dy**2 + dx**2)))
    metrics["motion_mae_y"] = float(np.mean(np.abs(dy)))
    metrics["motion_mae_x"] = float(np.mean(np.abs(dx)))
    metrics["motion_rmse_y"] = float(np.sqrt(np.mean(dy**2)))
    metrics["motion_rmse_x"] = float(np.sqrt(np.mean(dx**2)))

    # Total / physical metrics
    if use_z and pred.shape[-1] >= 3:
        dz = diff[..., 2]
        epe_3d = np.sqrt(dy**2 + dx**2 + dz**2)
        epe_phys = np.sqrt(dy**2 + dx**2 + (z_ratio * dz) ** 2)

        metrics["motion_epe_3d_mean"] = float(np.mean(epe_3d))
        metrics["motion_epe_3d_median"] = float(np.median(epe_3d))
        metrics["motion_epe_phys_mean"] = float(np.mean(epe_phys))
        metrics["motion_epe_phys_median"] = float(np.median(epe_phys))
        metrics["motion_rmse_z"] = float(np.sqrt(np.mean(dz**2)))
        metrics["motion_mae_z"] = float(np.mean(np.abs(dz)))

    return metrics


def plot_publication_metric(
    processed_results,
    experiment_groups,
    avg_key_1,
    std_key_1,
    label_1,
    avg_key_2,
    std_key_2,
    label_2,
    ylabel,
    save_dir="figures",
    file_suffix="metric",
    use_std=True,
    yscale="linear",
    figsize=(4.8, 3.6),
    linewidth=2.0,
    markersize=5.5,
    dpi=300,
):
    """
    Plot publication-style curves for any two metrics.

    Parameters
    ----------
    processed_results : dict
    experiment_groups : list
    avg_key_1, std_key_1 : str
        processed_results[group] 中第1条曲线对应的均值/标准差字段名
    label_1 : str
        第1条曲线图例名称
    avg_key_2, std_key_2 : str
        第2条曲线字段名
    label_2 : str
        第2条曲线图例名称
    ylabel : str
        y轴名称
    save_dir : str
    file_suffix : str
        保存文件名后缀
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for group in experiment_groups:
        if group not in processed_results or not processed_results[group]:
            continue

        data = processed_results[group]

        # 检查字段是否存在
        required_keys = [avg_key_1, std_key_1, avg_key_2, std_key_2]
        missing_keys = [k for k in required_keys if k not in data]
        if len(missing_keys) > 0:
            print(f"Skip {group}: missing keys {missing_keys}")
            continue

        labels = data["labels"]
        try:
            x = np.array(labels, dtype=float)
            x_tick_labels = [str(v) for v in labels]
        except Exception:
            x = np.arange(len(labels))
            x_tick_labels = labels

        avg_1 = np.array(data[avg_key_1], dtype=float)
        std_1 = np.array(data[std_key_1], dtype=float)
        avg_2 = np.array(data[avg_key_2], dtype=float)
        std_2 = np.array(data[std_key_2], dtype=float)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x, avg_1, marker="o", linewidth=linewidth, markersize=markersize, label=label_1)

        ax.plot(x, avg_2, marker="s", linewidth=linewidth, markersize=markersize, label=label_2)

        if use_std:
            ax.fill_between(x, avg_1 - std_1, avg_1 + std_1, alpha=0.18)
            ax.fill_between(x, avg_2 - std_2, avg_2 + std_2, alpha=0.18)

        ax.set_xlabel(group)
        ax.set_ylabel(ylabel)
        ax.set_title(group)

        if yscale in ["linear", "log"]:
            ax.set_yscale(yscale)

        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.legend(frameon=False, loc="best")

        fig.tight_layout()

        pdf_path = os.path.join(save_dir, f"{group}_{file_suffix}.pdf")
        png_path = os.path.join(save_dir, f"{group}_{file_suffix}.png")

        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {pdf_path}")
        print(f"Saved: {png_path}")
