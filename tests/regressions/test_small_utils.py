"""Regression tests for simulation.py, preprocess.py and generate_demo_data.py findings."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from wholistic_registration.utils import generate_demo_data, preprocess, simulation


def test_b054_generate_motion_runs():
    """generateMotion runs to completion (no cupy_ndimge NameError) and returns motion arrays of the input shape. Regression for B-054 (fixed in cbf274c)."""
    np.random.seed(0)
    raw = np.random.rand(8, 8, 2).astype(np.float32)

    motion_X, motion_Y, motion_Z, cp_art = simulation.generateMotion(
        raw, art_R=2, amp_art=1.0, zRatio=1
    )

    assert motion_X.shape == (8, 8, 2)
    assert motion_Y.shape == (8, 8, 2)
    assert motion_Z.shape == (8, 8, 2)
    assert len(cp_art) > 0


def test_b055_yunfeng_edge_map_runs():
    """Yunfeng_edge_map runs on a 48x48 array (all imports resolved) and returns a binary map of the same shape. Regression for B-055 (fixed in 8f9e3d3)."""
    yy, xx = np.mgrid[:48, :48]
    frame = ((yy - 24) ** 2 + (xx - 24) ** 2 < 15**2).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        edges = preprocess.Yunfeng_edge_map(frame)

    assert edges.shape == (48, 48)
    assert set(np.unique(edges)).issubset({0.0, 1.0})


def test_b056_plot_publication_metric_reaches_past_plt(tmp_path):
    """plot_publication_metric executes past its plt usage (no NameError) on empty inputs. Regression for B-056 (fixed in 283b255)."""
    saved_rc = dict(plt.rcParams)
    try:
        result = simulation.plot_publication_metric(
            processed_results={},
            experiment_groups=[],
            avg_key_1="avg1",
            std_key_1="std1",
            label_1="l1",
            avg_key_2="avg2",
            std_key_2="std2",
            label_2="l2",
            ylabel="metric",
            save_dir=str(tmp_path / "figures"),  # empty groups: nothing is written
        )
        assert result is None
    finally:
        plt.rcParams.update(saved_rc)
        plt.close("all")


def _canny_edge_map_mod180(frame, sigma=1.0, low_threshold=0.05, high_threshold=0.15, eps=1e-6):
    """Reference canny with the CORRECT mod-180 angle fold (identical otherwise)."""
    smoothed = ndi.gaussian_filter(frame, sigma=sigma, mode="nearest")

    gx = ndi.sobel(smoothed, axis=-1, mode="nearest")
    gy = ndi.sobel(smoothed, axis=-2, mode="nearest")

    grad_mag = np.sqrt(gx**2 + gy**2 + eps)
    grad_dir = np.arctan2(gy, gx) * (180 / np.pi)
    grad_dir = np.mod(grad_dir, 180.0)  # fold direction into [0, 180) — NOT abs()

    H, W = frame.shape
    suppressed = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            angle = grad_dir[i, j]

            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbors = [grad_mag[i, j - 1], grad_mag[i, j + 1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [grad_mag[i - 1, j + 1], grad_mag[i + 1, j - 1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [grad_mag[i - 1, j], grad_mag[i + 1, j]]
            else:  # 112.5 <= angle < 157.5
                neighbors = [grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]]

            if grad_mag[i, j] >= max(neighbors):
                suppressed[i, j] = grad_mag[i, j]

    suppressed = (suppressed - suppressed.min()) / (suppressed.max() - suppressed.min() + eps)

    high_mask = suppressed >= high_threshold
    low_mask = (suppressed >= low_threshold) & ~high_mask

    edges = high_mask.copy().astype(np.float32)
    connectivity = ndi.generate_binary_structure(2, 2)

    connected_weak = ndi.binary_dilation(high_mask, structure=connectivity) & low_mask
    edges[connected_weak] = 1.0

    return edges


def test_b058_canny_edge_map_matches_mod180_reference():
    """canny_edge_map on a disk image equals a reference implementation whose only change is the correct mod-180 angle fold. Regression for B-058 (fixed: NMS now folds the gradient orientation mod 180 instead of abs())."""
    yy, xx = np.mgrid[:64, :64]
    disk = ((yy - 32.0) ** 2 + (xx - 32.0) ** 2 < 20.0**2).astype(np.float64)
    frame = ndi.gaussian_filter(disk, sigma=2.0, mode="nearest")

    got = preprocess.canny_edge_map(frame)
    want = _canny_edge_map_mod180(frame)

    np.testing.assert_array_equal(got, want)


def test_b060_generate_cell_movement_draws_per_cell_per_frame(monkeypatch):
    """generate_cell_movement makes one uniform 2-vector draw per cell per frame (9 for 4 frames x 3 cells) and none when displacement is fixed. Regression for B-060 (fixed in d728309)."""
    real_uniform = np.random.uniform
    step_draws = {"n": 0}

    def counting_uniform(*args, **kwargs):
        size = kwargs.get("size", args[3] if len(args) > 3 else None)
        if size == 2:
            step_draws["n"] += 1
        return real_uniform(*args, **kwargs)

    monkeypatch.setattr(np.random, "uniform", counting_uniform)

    generate_demo_data.generate_cell_movement(
        num_frames=4, image_size=(64, 64), num_cells=3, max_displacement=3.0, seed=0
    )
    # (num_frames - 1) * num_cells = 3 * 3 = 9 independent step draws
    assert step_draws["n"] == 9

    step_draws["n"] = 0
    generate_demo_data.generate_cell_movement(
        num_frames=4,
        image_size=(64, 64),
        num_cells=3,
        max_displacement=3.0,
        displacement=(1.0, 0.5),
        seed=0,
    )
    # caller-fixed displacement: no random step draws at all
    assert step_draws["n"] == 0
