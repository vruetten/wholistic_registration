"""Strict-xfail regression tests for confirmed-but-deferred findings (audit pass 1).

Each test asserts the CORRECT behavior; while the finding stays unfixed, the
test xfails. When a fix lands, the strict xfail turns into an XPASS failure,
forcing the marker's removal (test promotion).
"""

import types

import numpy as np
import pytest
import scipy.ndimage as ndi

from wholistic_registration.utils import motion_correlation_pattern as mcp
from wholistic_registration.utils import preprocess


@pytest.mark.xfail(
    strict=True,
    reason="B-049 deferred: MAD=0 on sparse traces makes the threshold ~1e-12, so "
    "mad_k is inert across orders of magnitude (sibling has the nanstd guard)",
)
def test_b049_mad_k_changes_events_on_sparse_trace():
    """detect_activation_events_mad gives DIFFERENT events for mad_k=3 vs mad_k=1000 on a 1000-frame sparse trace. Regression for B-049 (deferred)."""
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


@pytest.mark.xfail(
    strict=True,
    reason="B-058 deferred: NMS folds the gradient angle with abs() instead of mod 180, "
    "suppressing (-157.5,-112.5)U(-67.5,-22.5) diagonals against the wrong diagonal; "
    "results-changing fix awaiting sign-off",
)
def test_b058_canny_edge_map_matches_mod180_reference():
    """canny_edge_map on a disk image equals a reference implementation whose only change is the correct mod-180 angle fold. Regression for B-058 (deferred)."""
    yy, xx = np.mgrid[:64, :64]
    disk = ((yy - 32.0) ** 2 + (xx - 32.0) ** 2 < 20.0**2).astype(np.float64)
    frame = ndi.gaussian_filter(disk, sigma=2.0, mode="nearest")

    got = preprocess.canny_edge_map(frame)
    want = _canny_edge_map_mod180(frame)

    np.testing.assert_array_equal(got, want)


@pytest.mark.xfail(
    strict=True,
    reason="B-046 deferred: lag_mode value sets are incompatible between sibling APIs; "
    "'both' raises ValueError here while the sibling accepts it; API semantics fix "
    "awaiting sign-off",
)
def test_b046_lagged_ca_correlation_accepts_lag_mode_both():
    """compute_lagged_ca_correlation_map accepts lag_mode='both' (the sibling API's value) without ValueError. Regression for B-046 (deferred)."""
    rng = np.random.default_rng(0)
    T = 50
    activation = rng.standard_normal(T).astype(np.float32)
    ca = rng.random((T, 4, 4)).astype(np.float32) + 1.0

    result = mcp.compute_lagged_ca_correlation_map(
        activation, ca, max_lag=5, lag_mode="both", use_dff=False
    )

    assert isinstance(result, dict)
    assert result["best_lag"].shape == (4, 4)


def _mode_unit_with_support(support_mask):
    """Build a MotionMode unit via __new__ + __dict__ (as in the verification log)."""
    m = mcp.MotionMode.__new__(mcp.MotionMode)
    m.__dict__.update(
        {
            "episode_id": 0,
            "mode_id": 0,
            "support_mask": support_mask.astype(np.uint8),
        }
    )
    return m


@pytest.mark.xfail(
    strict=True,
    reason="B-016 deferred: pattern_to_binary_mask reads only region_mask, so the "
    "union path is silently dead for mode-unit patterns (falls back to the 20%-"
    "threshold prototype); API semantics fix awaiting sign-off",
)
def test_b016_pattern_to_binary_mask_unions_mode_support_masks():
    """pattern_to_binary_mask on a mode-unit pattern returns the union of the members' support_masks, not the prototype fallback. Regression for B-016 (deferred)."""
    shape = (40, 40)
    sup1 = np.zeros(shape, dtype=bool)
    sup1[0:10, 0:20] = True  # 200 px
    sup2 = np.zeros(shape, dtype=bool)
    sup2[20:35, 0:30] = True  # 450 px

    proto = np.zeros(shape, dtype=np.float32)
    proto[5:10, 5:10] = 1.0  # fallback would give only 25 px

    pattern = types.SimpleNamespace(
        regions=[_mode_unit_with_support(sup1), _mode_unit_with_support(sup2)],
        prototype_region_map=proto,
    )

    mask = mcp.pattern_to_binary_mask(pattern)

    expected_union = sup1 | sup2  # 650 px
    np.testing.assert_array_equal(mask, expected_union)
