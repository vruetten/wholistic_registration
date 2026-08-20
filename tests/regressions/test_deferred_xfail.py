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
