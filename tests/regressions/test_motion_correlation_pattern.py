"""Regression tests for utils/motion_correlation_pattern.py findings (audit pass 1)."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wholistic_registration.utils import motion_correlation_pattern as mcp


def _make_mode(response_strength, response_field, activation, episode_id=0, mode_id=0):
    """Build a MotionMode via __new__ + __dict__ (as in the verification log)."""
    mode = mcp.MotionMode.__new__(mcp.MotionMode)
    mode.__dict__.update(
        {
            "episode_id": episode_id,
            "mode_id": mode_id,
            "time_range": (0, len(activation) - 1),
            "activation": np.asarray(activation, dtype=np.float32),
            "response_field": np.asarray(response_field, dtype=np.float32),
            "response_strength": np.asarray(response_strength, dtype=np.float32),
        }
    )
    return mode


def test_b002_safe_corr_constant_traces():
    """_safe_corr scores constant traces via un-demeaned cosine: const/const ~1, const/-const ~-1, const/zero 0, and matches np.corrcoef on the normal path. Regression for B-002 (fixed in 5b3c725)."""
    assert mcp._safe_corr([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0, abs=1e-5)
    assert mcp._safe_corr([5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]) == pytest.approx(-1.0, abs=1e-5)
    assert mcp._safe_corr([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-6)

    rng = np.random.default_rng(0)
    a = rng.standard_normal(50).astype(np.float32)
    b = (0.8 * a + 0.2 * rng.standard_normal(50)).astype(np.float32)
    expected = np.corrcoef(a, b)[0, 1]
    assert mcp._safe_corr(a, b) == pytest.approx(float(expected), abs=1e-4)


def test_b009_split_mode_survives_single_nan_pixel():
    """split_mode_to_regions yields the same region count for a response map with one NaN pixel as for the clean map. Regression for B-009 (fixed in c33e6ce)."""
    A = np.zeros((40, 40), dtype=np.float32)
    A[10:20, 10:20] = 1.0
    B = np.zeros((40, 40, 2), dtype=np.float32)
    B[10:20, 10:20, 0] = 0.5
    h = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)

    clean = mcp.split_mode_to_regions(_make_mode(A, B, h))

    A_nan = A.copy()
    A_nan[12, 12] = np.nan
    with_nan = mcp.split_mode_to_regions(_make_mode(A_nan, B, h))

    assert len(clean) == 1
    assert len(with_nan) == len(clean)


def _make_region(episode_id, activation, rng):
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    field = np.zeros((20, 20, 2), dtype=np.float32)
    field[5:15, 5:15, :] = rng.standard_normal((10, 10, 2)).astype(np.float32)
    return mcp.MotionRegion(
        episode_id=episode_id,
        mode_id=0,
        region_id=episode_id,
        time_range=(0, 9),
        activation=activation,
        response_field=field,
        response_strength=np.linalg.norm(field, axis=-1).astype(np.float32),
        region_mask=mask,
    )


def test_b012_activationless_pair_gets_invalid_activation_not_stale_distance():
    """compute_region_distance_matrix_simple handles an activation-less pair (first or after a valid pair) with reason=invalid_activation and dist>=1e6, no UnboundLocalError. Regression for B-012 (fixed in 83fd55b)."""
    rng = np.random.default_rng(1)
    h = np.sin(np.linspace(0, np.pi, 10)).astype(np.float32)
    r_noact = _make_region(0, None, rng)
    r_valid1 = _make_region(1, h, rng)
    r_valid2 = _make_region(2, h * 1.1 + 0.05, rng)

    # Case 1: the activation-less pair is the FIRST pair examined.
    D, info = mcp.compute_region_distance_matrix_simple(
        [r_noact, r_valid1, r_valid2], verbose=False
    )
    assert info[(0, 1)]["reason"] == "invalid_activation"
    assert info[(0, 1)]["compatible"] is False
    assert D[0, 1] >= 1e6

    # Case 2: the activation-less pair comes AFTER a valid pair.
    D2, info2 = mcp.compute_region_distance_matrix_simple(
        [r_valid1, r_valid2, r_noact], verbose=False
    )
    assert info2[(0, 1)]["reason"] == "ok"
    assert info2[(0, 1)]["compatible"] is True
    assert np.isfinite(D2[0, 1]) and D2[0, 1] < 1e6
    for pair in [(0, 2), (1, 2)]:
        assert info2[pair]["reason"] == "invalid_activation"
        assert info2[pair]["compatible"] is False
        assert D2[pair] >= 1e6


def test_b041_unfitted_episode_skipped_and_valueerror():
    """summarize_temporal_basis_likeness skips a default (mode_model={}) episode and _get_BH_from_episode raises ValueError on it. Regression for B-041 (fixed in 867e82c)."""
    ep = mcp.MotionEpisode()
    assert ep.mode_model == {}

    rows = mcp.summarize_temporal_basis_likeness([ep])
    assert len(rows) == 0  # skipped, no KeyError('B')

    with pytest.raises(ValueError):
        mcp._get_BH_from_episode(ep)


def test_b043_tab20_colormap_lookup_works():
    """visualize_episode_regions runs end-to-end on a one-region episode (no AttributeError from the removed cm.get_cmap) and paints its scatter with tab20 colour 0 at the expected patch centres. Regression for B-043 (fixed in 8bfd0e2)."""
    source = inspect.getsource(mcp)
    assert "cm.get_cmap" not in source

    cmap = plt.get_cmap("tab20")  # the exact lookup used at visualize_episode_regions
    assert cmap.N == 20

    # The checks above only exercise matplotlib and the source text.  Actually
    # call the repo function that does the lookup.  Backend is forced to Agg in
    # tests/regressions/conftest.py.
    plt.close("all")
    rng = np.random.default_rng(43)
    region = _make_region(0, np.sin(np.linspace(0, np.pi, 10)).astype(np.float32), rng)
    region.mean_response_vector = np.array([0.3, -0.2], dtype=np.float32)
    region.center_xy = (9.0, 9.0)

    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    episode = mcp.MotionEpisode(time_range=(0, 9), region_mask=mask, episode_id=43)
    episode.regions = [region]

    fig = mcp.visualize_episode_regions(episode, patch_size=7, rel_thresh=0.10, show=False)

    assert fig is not None  # None means it bailed out before the colormap lookup

    # Independent recomputation of what the scatter must contain: the pixels of
    # the region's response_strength above 10% of its own max, mapped to patch
    # centres, all painted with tab20 entry 0.
    A = np.asarray(region.response_strength, dtype=np.float32)
    coords = np.argwhere(A > 0.10 * float(np.max(A)))
    assert len(coords) > 0  # guard: the oracle below must not be empty
    expected_xy = np.column_stack([coords[:, 1] * 7 + 3, coords[:, 0] * 7 + 3]).astype(float)

    scatters = [c for c in fig.axes[0].collections if c.get_offsets().shape[0] == len(coords)]
    assert len(scatters) == 1, f"expected exactly one scatter of {len(coords)} points"
    assert np.array_equal(np.asarray(scatters[0].get_offsets()), expected_xy)

    face = np.asarray(scatters[0].get_facecolor())
    assert face.shape == (len(coords), 4)
    assert np.allclose(face[:, :3], np.asarray(cmap(0))[:3]), (
        f"scatter is not painted with tab20 colour 0: got {face[0, :3]}"
    )
    assert face[:, 3].min() >= 0.25 and face[:, 3].max() <= 0.90

    plt.close("all")


def _make_k0_episode():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    N = int(mask.sum())
    T = 5
    ep = mcp.MotionEpisode(
        time_range=(0, T - 1),
        region_mask=mask,
        episode_id=7,
        motion_delta=np.zeros((T, N, 2), dtype=np.float32),
        motion_abs=np.zeros((T, N, 2), dtype=np.float32),
        global_motion=np.zeros((T, 2), dtype=np.float32),
    )
    ep.mode_model = {
        "B": np.zeros((2 * N, 0), dtype=np.float32),
        "H": np.zeros((0, T), dtype=np.float32),
    }
    return ep


def test_b044_k0_episode_source_viz_returns_cleanly():
    """Both source-visualization functions return cleanly on a K=0 mode_model and leave no open figures. Regression for B-044 (fixed in 603282b)."""
    plt.close("all")
    ep = _make_k0_episode()

    diag = mcp.visualize_episode_sources_overview(ep)
    assert diag == []

    result = mcp.compare_sources_to_observed_frames(ep)
    assert result is None

    assert plt.get_fignums() == []
