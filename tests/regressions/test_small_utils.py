"""Regression tests for simulation.py, preprocess.py and generate_demo_data.py findings."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import ndimage as ndi

from wholistic_registration.utils import generate_demo_data, preprocess, simulation


def test_b054_generate_motion_runs():
    """generateMotion runs to completion (no cupy_ndimge NameError) and the returned fields carry the requested amplitude (std over the seed points == amp_art), are Gaussian-smoothed, and use exactly round(N/(2R+1)^2) seed points. Regression for B-054 (fixed in cbf274c)."""
    np.random.seed(0)
    raw = np.random.rand(16, 16, 4).astype(np.float32)
    art_R, amp_art = 2, 3.0

    motion_X, motion_Y, motion_Z, cp_art = simulation.generateMotion(
        raw, art_R=art_R, amp_art=amp_art, zRatio=1
    )

    assert motion_X.shape == (16, 16, 4)
    assert motion_Y.shape == (16, 16, 4)
    assert motion_Z.shape == (16, 16, 4)
    assert len(cp_art) > 0

    # Everything above is satisfied by three all-zero arrays plus any non-empty
    # index list.  Below: derived invariants that pin the actual computation.

    # 1. seed-point count is exactly N * 1/(2R+1)^2, and the indices are a
    #    permutation sample (unique, in range) -- not a placeholder.
    n_voxels = 16 * 16 * 4
    assert len(cp_art) == round(n_voxels / (2 * art_R + 1) ** 2)
    assert len(np.unique(cp_art)) == len(cp_art)
    assert cp_art.min() >= 0 and cp_art.max() < n_voxels

    # 2. amplitude normalisation: motion_Y is divided by std(motion_Y[cp_art])
    #    and multiplied by amp_art, so that std must come back out as amp_art
    #    exactly.  Independent of the RNG draw; false for any zero/stub field.
    assert float(np.std(motion_Y.flat[cp_art])) == pytest.approx(amp_art, rel=1e-5)

    # 3. the fields are non-degenerate and distinct
    assert float(motion_X.std()) > 0.0
    assert float(motion_Y.std()) > 0.0
    assert not np.array_equal(motion_X, motion_Y)

    # 4. motion_Z is left at zero on purpose (its randn/filter/scale lines are
    #    commented out upstream) -- no-regression anchor for that choice.
    assert np.count_nonzero(motion_Z) == 0

    # 5. the Gaussian filter really ran: with sigma=art_R the field is smooth,
    #    so neighbouring voxels along x are strongly correlated.  White noise
    #    (an unfiltered field) gives ~0 here (measured: -0.027), so 0.75
    #    separates the two cases by a mile.  Threshold chosen from a 60-seed
    #    sweep: lag-1 ranges 0.892-0.969, so the tempting 0.9 would fail on
    #    seeds 27 and 40 -- fine here only because seed 0 is pinned, but a trap
    #    for anyone who reseeds.
    lag1 = np.corrcoef(motion_Y[:-1].ravel(), motion_Y[1:].ravel())[0, 1]
    assert lag1 > 0.75, f"motion_Y is not spatially smooth: lag-1 correlation {lag1:.3f}"


def test_b056_plot_publication_metric_reaches_past_plt(tmp_path, monkeypatch):
    """plot_publication_metric executes past its plt usage (no NameError) and, given real data, draws both labelled curves with the supplied y-values and writes the pdf/png pair. Regression for B-056 (fixed in 283b255)."""
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
        # `result is None` is also true of a stub that does nothing at all, so
        # run it again with a real group and check what it drew and wrote.

        avg1 = [1.0, 2.0, 4.0]
        avg2 = [0.5, 0.25, 0.125]
        data = {
            "labels": [1, 2, 3],
            "avg1": avg1,
            "std1": [0.1, 0.1, 0.1],
            "avg2": avg2,
            "std2": [0.05, 0.05, 0.05],
        }
        out_dir = tmp_path / "figures_real"
        plt.close("all")
        # keep the figure alive so its artists can be inspected after the call
        drawn = []
        real_subplots = plt.subplots

        def recording_subplots(*args, **kwargs):
            drawn.append(real_subplots(*args, **kwargs))
            return drawn[-1]

        # `simulation.plt` IS matplotlib.pyplot, so this patch is global for the
        # duration of the test.  Stub `close` so the figure survives the call and
        # its artists can be inspected -- but that also neuters this test's own
        # `finally: plt.close("all")`, since monkeypatch teardown runs after the
        # body.  Undo explicitly before the finally block.
        monkeypatch.setattr(simulation.plt, "subplots", recording_subplots)
        monkeypatch.setattr(simulation.plt, "close", lambda *a, **k: None)

        simulation.plot_publication_metric(
            processed_results={"grp": data},
            experiment_groups=["grp"],
            avg_key_1="avg1",
            std_key_1="std1",
            label_1="curve one",
            avg_key_2="avg2",
            std_key_2="std2",
            label_2="curve two",
            ylabel="metric",
            save_dir=str(out_dir),
            file_suffix="mtr",
            dpi=50,
        )

        for name in ["grp_mtr.pdf", "grp_mtr.png"]:
            path = out_dir / name
            assert path.exists(), f"missing output: {name}"
            assert path.stat().st_size > 0

        # exactly one figure was drawn, carrying both curves with the y-values
        # and legend labels that were passed in
        assert len(drawn) == 1
        ax = drawn[0][1]
        assert len(ax.lines) == 2
        assert [ln.get_label() for ln in ax.lines] == ["curve one", "curve two"]
        assert np.array_equal(ax.lines[0].get_ydata(), np.array(avg1))
        assert np.array_equal(ax.lines[1].get_ydata(), np.array(avg2))
        assert np.array_equal(ax.lines[0].get_xdata(), np.array([1.0, 2.0, 3.0]))
        assert ax.get_ylabel() == "metric"
        assert ax.get_title() == "grp"
    finally:
        monkeypatch.undo()  # restore the real plt.close before using it
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
