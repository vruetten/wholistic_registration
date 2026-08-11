"""Regression tests for the z-anisotropy geometry in calFlowCrossResolution.

The module convention (six independent sites, see PR discussion) is that
zRatio* is *physical units per z index*, with one xy index = one physical unit.
"""

import numpy as np
import pytest

from wholistic_registration.utils import calFlowCrossResolution as cf
from wholistic_registration.utils.calFlowCrossResolution import (
    _make_ball,
    build_reference_trap_mask_from_bad_moving_fast_roi,
)


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
    [
        pytest.param(3.0, 1, id="zr3.0"),
        pytest.param(2.0, 3, id="zr2.0"),
        pytest.param(1.0, 5, id="zr1.0-ANCHOR"),
        pytest.param(0.5, 9, id="zr0.5"),
        pytest.param(0.25, 17, id="zr0.25"),
        pytest.param(0.125, 33, id="zr0.125"),
    ],
)
def test_b090_z_extent_follows_physical_reach(z_ratio, expected_z_extent):
    """The ball reaches radius_xy / z_ratio z-indices: coarser z (ratio>1) shrinks it, finer z (ratio<1) grows it. Regression for B-090.

    The ``zr1.0-ANCHOR`` case is a **no-regression anchor**, not a regression
    case: at z_ratio == 1 the pre-fix and post-fix formulas are provably
    identical (radius_xy * 1 == radius_xy / 1 and dz / 1 == dz * 1), so it
    passes on unfixed code by construction. It is kept because it is the pivot
    of the sweep -- the boundary between the "z coarser, element shrinks"
    (ratio > 1) and "z finer, element grows" (ratio < 1) regimes -- and
    dropping it would leave a hole exactly where the two regimes meet. The
    other five cases do fail on unfixed code.
    """
    ball = _asnumpy(_make_ball(2, z_ratio))
    occupied = int(np.count_nonzero(ball.any(axis=(0, 1))))
    assert occupied == expected_z_extent


@pytest.mark.parametrize(
    "z_ratio",
    [
        pytest.param(0.125, id="zr0.125"),
        pytest.param(0.5, id="zr0.5"),
        pytest.param(1.0, id="zr1.0-ANCHOR"),
        pytest.param(3.0, id="zr3.0"),
    ],
)
def test_b090_ball_is_isotropic_in_physical_space(z_ratio):
    """Every included voxel is within radius_xy physical units and every excluded one is beyond it. Regression for B-090.

    ``zr1.0-ANCHOR`` is a **no-regression anchor** for the same reason as in
    ``test_b090_z_extent_follows_physical_reach``: the two formulas coincide at
    z_ratio == 1, so that case cannot fail on unfixed code.
    """
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

    # A tiny-but-valid ratio must stay bounded rather than allocating millions
    # of voxels -- but "bounded" has to be two-sided, or collapsing radius_z to
    # 1 would satisfy it while destroying the element.
    #
    # Expected value, derived without running _make_ball: radius_z is
    # ceil(radius_xy / z_ratio) clamped to _MAX_BALL_RADIUS_Z, and
    # ceil(2 / 1e-12) = 2e12 is far above the cap, so the cap binds exactly and
    # the element is 2 * _MAX_BALL_RADIUS_Z + 1 planes deep. At that ratio the
    # anisotropy term (dz * 1e-12)**2 is ~1e-24 for every |dz| <= the cap, i.e.
    # negligible against radius_xy**2 == 4, so the membership test collapses to
    # dx**2 + dy**2 <= 4 independently of dz: every z plane carries a full copy
    # of the xy disc, so every plane must be occupied.
    cap = cf._MAX_BALL_RADIUS_Z
    ball = _asnumpy(_make_ball(2, 1e-12))
    # Absolute bound FIRST: asserting only against cf._MAX_BALL_RADIUS_Z is
    # circular -- raising the constant to 200_000 would keep that assertion
    # green while allocating a ~10-million-voxel element, the very thing the
    # cap exists to prevent (verified).
    assert ball.size < 10**6, f"structuring element is unbounded: {ball.shape}"
    assert cap <= 128, "cap no longer bounds the allocation it exists to bound"
    assert ball.shape[2] == 2 * cap + 1, f"radius_z should saturate at the cap {cap}"
    assert ball.any(), "capped element must still be a usable structuring element, not empty"
    assert ball.any(axis=(0, 1)).all(), "every z plane of the capped element must be occupied"


# ---------------------------------------------------------------------------
# The same zRatio inversion in the ROI margin of
# build_reference_trap_mask_from_bad_moving_fast_roi (fixed in the same commit
# as _make_ball, but previously untested).
# ---------------------------------------------------------------------------

_EXPAND_RADIUS_XY = 2


def _trap_mask_phantom():
    """Synthetic reference volume + moving bad-region mask for the trap-mask ROI path.

    Geometry is chosen so the trap mask's extent is decided by the structuring
    elements alone, never by the intensity constraint or by the volume border:

    - the bright slab is homogeneous and spans x,y in [16,49) and all of z, so
      the intensity-similarity test admits the whole neighbourhood the
      morphology can reach;
    - the seed sits at the centre, far enough from every face that even the
      widest case (z_ratio=0.125, reach 16 z indices) stays inside the volume.

    phase_new is the identity map, so reference seed coords == moving coords
    and the expected trap location can be stated directly from ``bad_mask``.
    """
    shape = (64, 64, 64)

    bad_mask = np.zeros(shape, dtype=bool)
    bad_mask[30:35, 30:35, 30:33] = True

    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    phase_new = np.stack([ii, jj, kk], axis=-1).astype(np.float32)

    rng = np.random.default_rng(0)
    data_ref = np.full(shape, 100.0, dtype=np.float32)
    data_ref[16:49, 16:49, :] = 900.0
    # Mild noise for realism only. It is NOT load-bearing: the reach is
    # geometry-bound, and this phantom passes identically at sigma=0 (verified).
    # The degenerate-MAD failure mode does exist, but needs a seed small/uniform
    # enough that I_mad == 0 (e.g. a single voxel, where the trap collapses to
    # the seed) -- which this 5x5x3 seed does not reproduce.
    data_ref = (data_ref + rng.normal(0.0, 10.0, shape)).astype(np.float32)

    return bad_mask, phase_new, data_ref


@pytest.mark.parametrize(
    "z_ratio,expected_z_reach",
    [
        pytest.param(0.25, 8, id="zr0.25"),
        pytest.param(0.125, 16, id="zr0.125"),
    ],
)
def test_b090_trap_mask_roi_margin_spans_the_physical_z_neighbourhood(z_ratio, expected_z_reach):
    """The ROI z-margin must be deep enough to hold the trap region, else the mask silently comes back empty. Regression for B-090 (margin_z half of the fix).

    ``margin_z`` in ``build_reference_trap_mask_from_bad_moving_fast_roi`` had
    the same inverted zRatio convention as ``_make_ball``. With only the ball
    fixed, the crop is far too thin in z: ``binary_closing``'s erosion step sees
    the ROI face inside its own footprint, wipes ``seed_crop`` out entirely, and
    the ``if not any(seed_crop): return trap_mask_ref`` guard returns an
    **all-zero** trap mask with no warning at all -- wrong-region correction is
    silently switched off for that layer.

    Expected reach, derived from the algorithm rather than from its output:
    the region is grown ``max_steps = expand_radius_xy = 2`` times by
    ``_make_ball(1, z_ratio)``, a structuring element of physical radius 1,
    so the trap extends 2 *physical* units past the seed in every direction --
    2 indices in x and y (1 index = 1 physical unit) and ``2 / z_ratio``
    indices in z, i.e. 8 at z_ratio=0.25 and 16 at z_ratio=0.125. The
    independent ``near_mask`` bound, ``_make_ball(expand_radius_xy, z_ratio)``,
    permits exactly the same reach, so neither constraint clips the other.
    """
    bad_mask, phase_new, data_ref = _trap_mask_phantom()

    trap = _asnumpy(
        build_reference_trap_mask_from_bad_moving_fast_roi(
            bad_mask=bad_mask,
            phase_new=phase_new,
            data_ref_layer=data_ref,
            z_ratio_ref=z_ratio,
            expand_radius_xy=_EXPAND_RADIUS_XY,
            sigma_grad=1.0,
            intensity_k=2.5,
        )
    )

    seed = np.argwhere(bad_mask)

    assert trap.any(), (
        "trap mask is all zero: the ROI z-margin was too thin, binary_closing eroded "
        "the seed away and the empty-seed guard silently disabled wrong-region correction"
    )
    assert trap[seed[:, 0], seed[:, 1], seed[:, 2]].all(), (
        "trap mask does not even cover the seed voxels it was built from"
    )

    trap_lo = np.argwhere(trap).min(axis=0)
    trap_hi = np.argwhere(trap).max(axis=0)
    seed_lo = seed.min(axis=0)
    seed_hi = seed.max(axis=0)

    expected = [_EXPAND_RADIUS_XY, _EXPAND_RADIUS_XY, expected_z_reach]
    assert (seed_lo - trap_lo).tolist() == expected
    assert (trap_hi - seed_hi).tolist() == expected

    # Two-sided: an "everything is a trap" mask would satisfy the reach floor
    # but is just as wrong as an empty one.
    assert int(trap.sum()) < 0.05 * trap.size


def test_b090_all_structuring_elements_in_the_trap_path_are_anisotropic():
    """Every _make_ball call in the trap-mask path must pass the real z_ratio, not a hardcoded 1.0.

    Mutation review found that replacing the CLOSING element with
    ``_make_ball(1, 1.0)`` -- the identical zRatio-convention bug four lines
    above the one under test -- left the whole suite green, because the closing
    step's anisotropy does not change the final reach in the phantom. Pinned
    structurally instead. Regression for B-090.
    """
    import ast
    import inspect
    import textwrap

    from wholistic_registration.utils import calFlowCrossResolution as cf

    src = textwrap.dedent(inspect.getsource(cf.build_reference_trap_mask_from_bad_moving_fast_roi))
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_make_ball"
    ]
    assert len(calls) == 3, f"expected 3 structuring elements in this path, found {len(calls)}"

    for call in calls:
        # the three sites are written differently (two positional, one keyword),
        # so resolve z_ratio from whichever form is used
        if len(call.args) >= 2:
            z_arg = ast.unparse(call.args[1])
        else:
            z_arg = next(
                (ast.unparse(kw.value) for kw in call.keywords if kw.arg == "z_ratio"), None
            )
        assert z_arg is not None, f"could not resolve z_ratio in {ast.unparse(call)}"
        assert not z_arg.replace(".", "").isdigit(), (
            f"{ast.unparse(call)} hardcodes an isotropic z_ratio -- the B-090 convention bug"
        )
