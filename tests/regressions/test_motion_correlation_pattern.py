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


def test_b049_mad_k_changes_events_on_sparse_trace():
    """detect_activation_events_mad gives DIFFERENT events for mad_k=3 vs mad_k=1000 on a 1000-frame sparse trace, and leaves a dense (MAD>0) trace untouched. Regression for B-049 (fixed)."""
    trace = np.zeros(1000, dtype=np.float32)
    trace[100] = 5.0
    trace[500] = 3.0
    trace[900] = 1.0

    events_lo = mcp.detect_activation_events_mad(trace, mad_k=3.0)
    events_hi = mcp.detect_activation_events_mad(trace, mad_k=1000.0)

    spans_lo = [(e["start"], e["end"]) for e in events_lo]
    spans_hi = [(e["start"], e["end"]) for e in events_hi]

    # a 333x larger threshold multiplier must change which frames are active
    assert spans_lo != spans_hi
    assert spans_hi == []  # nanstd fallback makes the high-k threshold unreachable

    # a dense trace has MAD > 0, so the fallback must not engage
    rng = np.random.default_rng(0)
    dense = rng.standard_normal(1000).astype(np.float32) * 0.3 + 1.0
    dense[200:210] += 5.0
    med = float(np.nanmedian(dense))
    mad = float(np.nanmedian(np.abs(dense - med)))
    assert mad > 0

    dense_events = mcp.detect_activation_events_mad(dense, mad_k=3.0)
    assert dense_events
    assert dense_events[0]["threshold"] == pytest.approx(med + 3.0 * 1.4826 * mad, rel=1e-6)

    # degenerate traces stay sane: no crash, nothing flagged
    assert mcp.detect_activation_events_mad(np.zeros(100, dtype=np.float32), mad_k=3.0) == []
    assert mcp.detect_activation_events_mad(np.full(100, 2.5, dtype=np.float32), mad_k=3.0) == []


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
