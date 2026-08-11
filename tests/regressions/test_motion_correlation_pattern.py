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
    """The module's colormap lookup (plt.get_cmap, not the removed cm.get_cmap) works on the installed matplotlib. Regression for B-043 (fixed in 8bfd0e2)."""
    source = inspect.getsource(mcp)
    assert "cm.get_cmap" not in source

    cmap = plt.get_cmap("tab20")  # the exact lookup used at visualize_episode_regions
    assert cmap.N == 20


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


# ---------------------------------------------------------------------------
# B-017 / B-019 / B-020 — temporal gap-closing semantics
# ---------------------------------------------------------------------------


def _active_trace(pattern):
    """(T,1,1) motionMag where 1 -> active under restMotion=0.5, use_abs_dev=False."""
    a = np.asarray(pattern, dtype=np.float32).reshape(-1, 1, 1)
    return a, np.zeros_like(a) + 0.5


def _run_units(pattern, close_gap_frames, extend_radius=0):
    """Return the MotionUnits of the single patch (getMotionUnit yields (units_map, mask))."""
    mag, rest = _active_trace(pattern)
    motion = np.zeros((mag.shape[0], 1, 1, 2), dtype=np.float32)
    units_map, _mask = mcp.getMotionUnit(
        motion, mag, rest, extend_radius=extend_radius,
        use_gpu=False, close_gap_frames=close_gap_frames, use_abs_dev=False,
    )
    return units_map[0][0]


def _spans(units):
    return sorted((int(u.time_range[0]), int(u.time_range[1])) for u in units)


def test_b019_closes_exactly_gaps_up_to_close_gap_frames():
    """A gap of exactly n closes; a gap of n+1 stays open — for even and odd structure sizes. Regression for B-019."""
    for n in (1, 2, 3, 4):
        closed = [1, 1] + [0] * n + [1, 1]
        assert len(_run_units(closed, n)) == 1, f"gap {n} should close at n={n}"
        open_ = [1, 1] + [0] * (n + 1) + [1, 1]
        assert len(_run_units(open_, n)) == 2, f"gap {n + 1} should stay open at n={n}"


def test_b019_border_runs_are_not_eroded():
    """Activity touching t=0 and t=T-1 survives closing (binary_closing would erode it). Regression for B-019."""
    for n in (1, 2, 3, 5):
        pattern = [1, 1, 1, 1] + [0] * 12 + [1, 1, 1, 1]
        units = _run_units(pattern, n)
        assert _spans(units) == [(0, 3), (16, 19)], f"n={n}: {_spans(units)}"


def test_b017_close_gap_frames_zero_is_a_strict_noop():
    """close_gap_frames=0 (the default) leaves the mask untouched — protects every existing config. Regression for B-017."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        pattern = (rng.random(30) > 0.5).astype(np.float32)
        assert _spans(_run_units(pattern, 0)) == _spans(_run_units(pattern, None))


def test_b020_explicit_use_gpu_without_cupy_raises_clearly():
    """use_gpu=True without CuPy raises an actionable error, not an opaque NoneType AttributeError. Regression for B-020."""
    if mcp.HAS_CUPY:
        pytest.skip("CuPy present: the GPU path is real here")
    mag, rest = _active_trace([1, 0, 1])
    motion = np.zeros((3, 1, 1, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="CuPy is unavailable"):
        mcp.getMotionUnit(motion, mag, rest, use_gpu=True, use_abs_dev=False)
    # and an unrecognised string is rejected rather than silently selecting GPU
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        mcp.getMotionUnit(motion, mag, rest, use_gpu="cpu", use_abs_dev=False)
