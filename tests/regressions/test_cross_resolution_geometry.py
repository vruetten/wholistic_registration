"""Regression tests for the z-anisotropy geometry in calFlowCrossResolution.

The module convention (six independent sites, see PR discussion) is that
zRatio* is *physical units per z index*, with one xy index = one physical unit.
"""

import numpy as np
import pytest

from wholistic_registration.utils.calFlowCrossResolution import _make_ball


def _asnumpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def test_b090_z_ratio_one_is_isotropic():
    """z_ratio == 1 gives the plain isotropic ball — the no-regression anchor. Regression for B-090."""
    for r in (1, 2, 3, 8):
        ball = _asnumpy(_make_ball(r, 1.0))
        xs = np.arange(-r, r + 1)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        expected = (X**2 + Y**2 + Z**2) <= r**2 + 1e-6
        assert np.array_equal(ball, expected), f"r={r}"


@pytest.mark.parametrize(
    "z_ratio,expected_z_extent",
    [(3.0, 1), (2.0, 3), (1.0, 5), (0.5, 9), (0.25, 17), (0.125, 33)],
)
def test_b090_z_extent_follows_physical_reach(z_ratio, expected_z_extent):
    """The ball reaches radius_xy / z_ratio z-indices: coarser z (ratio>1) shrinks it, finer z (ratio<1) grows it. Regression for B-090."""
    ball = _asnumpy(_make_ball(2, z_ratio))
    occupied = int(np.count_nonzero(ball.any(axis=(0, 1))))
    assert occupied == expected_z_extent


@pytest.mark.parametrize("z_ratio", [0.125, 0.5, 1.0, 3.0])
def test_b090_ball_is_isotropic_in_physical_space(z_ratio):
    """Every included voxel is within radius_xy physical units and every excluded one is beyond it. Regression for B-090."""
    r = 2
    ball = _asnumpy(_make_ball(r, z_ratio))
    cx, cy, cz = (np.array(ball.shape) - 1) // 2
    idx = np.argwhere(np.ones_like(ball, dtype=bool))
    dx, dy, dz = (idx[:, 0] - cx), (idx[:, 1] - cy), (idx[:, 2] - cz)
    phys = np.sqrt(dx**2 + dy**2 + (dz * z_ratio) ** 2)
    inside = ball[idx[:, 0], idx[:, 1], idx[:, 2]]

    assert phys[inside].max() <= r + 1e-6
    if (~inside).any():
        assert phys[~inside].min() > r


def test_b090_degenerate_z_ratio_is_rejected_not_exploded():
    """A non-positive or non-finite z_ratio raises instead of allocating an unbounded element; a tiny one is capped. Regression for B-090 (defect found in skeptic review)."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive, finite"):
            _make_ball(2, bad)

    # tiny-but-valid must stay bounded rather than allocating millions of voxels
    ball = _asnumpy(_make_ball(2, 1e-12))
    assert ball.shape[2] <= 2 * 64 + 1
