"""Regression tests for simulation.py, preprocess.py and generate_demo_data.py findings."""

import matplotlib.pyplot as plt
import numpy as np

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
