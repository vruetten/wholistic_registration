"""Regression tests for utils/reliableAnalysis.py findings (audit pass 1)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from wholistic_registration.utils import cp, reliableAnalysis

numpy_fallback = cp is np


def _zscore_difference_float64(I_ref, I_mov, sigma=1.5, eps=1e-6, p_mu=20, p_var=40, clip=(0, 10)):
    """float64 reference for local_zscore_difference, differing only in the clamp.

    Honest about what this is: a re-implementation of the same statistic in
    float64, NOT an independently derived oracle.  It is a paraphrase of the
    implementation, so it cannot catch a conceptual error in the statistic
    itself — if the documented formula is wrong, both agree and both are wrong.

    What it does catch is a *change* to the formula, because it is frozen
    test-side text: mutating p_var, the mask combinator, sigma, or |.| -> (.)^2
    in the implementation all make this disagree.

    The clamp is the one deliberate difference.  `E[I^2] - mu^2` cancels into a
    negative variance only at float32 precision; in float64 it does not, which
    this helper ASSERTS rather than assumes.  So it needs no clamp of its own,
    which is what makes it a usable check on B-061's clamp rather than a copy
    of it.
    """
    I_ref = np.asarray(I_ref, dtype=np.float64)
    I_mov = np.asarray(I_mov, dtype=np.float64)

    mu_ref = gaussian_filter(I_ref, sigma=sigma)
    mu_mov = gaussian_filter(I_mov, sigma=sigma)
    var_ref = gaussian_filter(I_ref**2, sigma=sigma) - mu_ref**2
    var_mov = gaussian_filter(I_mov**2, sigma=sigma) - mu_mov**2
    assert var_ref.min() >= 0.0 and var_mov.min() >= 0.0, (
        "float64 variance went negative: the oracle's premise (only float32 cancels) is broken"
    )

    mask_ref = (mu_ref > np.percentile(mu_ref, p_mu)) & (var_ref > np.percentile(var_ref, p_var))
    mask_mov = (mu_mov > np.percentile(mu_mov, p_mu)) & (var_mov > np.percentile(var_mov, p_var))
    D = np.abs(mu_ref - mu_mov) / (np.sqrt(var_ref + var_mov) + eps) * (mask_ref | mask_mov)
    if clip is None:
        return D
    return np.clip(D, clip[0], clip[1]) / (clip[1] - clip[0])


def test_b057_reliability_map_v2_runs_on_numpy_fallback():
    """reliability_map_v2 returns a numpy ndarray on the numpy fallback (no cp.asnumpy AttributeError). Regression for B-057 (fixed in 4ce9113)."""
    if not numpy_fallback:
        pytest.skip("cupy present: numpy-fallback path not active")

    rng = np.random.default_rng(0)
    template = (rng.random((32, 32)) * 1000).astype(np.float32)

    R = reliableAnalysis.reliability_map_v2(template)

    assert isinstance(R, np.ndarray)
    assert R.shape == (32, 32)
    assert np.all(np.isfinite(R))


def _high_offset_pair(shape=(256, 256), offset=3.0e4, noise=1.0, seed=0):
    """The B-061 generator: near-flat uint16-scale images with a high offset."""
    rng = np.random.default_rng(seed)
    I_ref = (offset + noise * rng.standard_normal(shape)).astype(np.float32)
    I_mov = (offset + noise * rng.standard_normal(shape)).astype(np.float32)
    return I_ref, I_mov


def _flat_plus_textured_pair(seed=0, height=128, width=128, flat_cols=38):
    """Two-regime generator for the B-061 value oracle.

    Left ~30% of the frame is a near-flat uint16-scale plateau (offset 3e4,
    noise 1) — the B-061 trigger: `gaussian_filter(I**2) - mu**2` there is a
    catastrophic cancellation at float32 and goes negative.  The remaining 70%
    is low-magnitude texture whose local variance is ~3 orders of magnitude
    larger, so (a) it is computed accurately in float32 and (b) it alone fixes
    the p_var=40 percentile threshold — which puts the plateau below the mask
    for float32 and float64 alike.  That makes the two precisions agree on the
    mask, so any residual disagreement in D is a numerical one.
    """
    rng = np.random.default_rng(seed)
    I_ref = np.empty((height, width))
    I_mov = np.empty((height, width))
    I_ref[:, :flat_cols] = 3.0e4 + rng.standard_normal((height, flat_cols))
    I_mov[:, :flat_cols] = 3.0e4 + rng.standard_normal((height, flat_cols))
    base = rng.random((height, width - flat_cols)) * 100.0
    I_ref[:, flat_cols:] = base + rng.standard_normal((height, width - flat_cols)) * 3.0
    I_mov[:, flat_cols:] = base + rng.standard_normal((height, width - flat_cols)) * 3.0
    return I_ref.astype(np.float32), I_mov.astype(np.float32)


def test_b061_local_zscore_difference_no_nans_at_high_offset():
    """local_zscore_difference produces 0 NaN pixels on high-offset float32 data AND matches a float64 reference implementation to 1e-5 (negative variances are clamped, not propagated). Regression for B-061 (fixed in e41224c).

    See `_zscore_difference_float64` on what that reference does and does not
    prove: it is a same-formula paraphrase in higher precision, not an
    independent derivation of the statistic.
    """
    I_ref, I_mov = _high_offset_pair()

    D = reliableAnalysis.local_zscore_difference(I_ref, I_mov)

    assert int(np.isnan(D).sum()) == 0
    # a returned map of zeros/constants would satisfy the NaN count above
    assert np.isfinite(D).all()
    assert float(D.std()) > 0.0
    assert float(D.min()) >= 0.0 and float(D.max()) <= 1.0

    # Value oracle: same statistic in float64, where the variance never cancels
    # negative, on data that mixes the B-061 trigger with a well-conditioned
    # textured region.  Observed max |difference| across seeds 0-7: 6.3e-7;
    # the map's own dynamic range is ~7.5e-3, so atol=1e-5 is ~750x tighter
    # than "all zeros" and ~16x looser than the observed float32 noise floor.
    I_ref2, I_mov2 = _flat_plus_textured_pair()
    D32 = reliableAnalysis.local_zscore_difference(I_ref2, I_mov2)
    D64 = _zscore_difference_float64(I_ref2, I_mov2)

    assert int(np.isnan(D32).sum()) == 0
    assert float(D32.std()) > 0.0
    assert float(D32.max()) > 1e-3  # the oracle comparison must not be vacuous
    assert np.allclose(D32, D64, atol=1e-5, rtol=0.0), (
        f"float32 map diverges from the float64 oracle: max|diff|={np.abs(D32 - D64).max():.3e}"
    )


def _over_clip_pair(seed=0, size=96):
    """Data whose raw z-difference exceeds the default clip ceiling of 10.

    A shared uniform-noise base fixes the local variance (~1e3) in both frames;
    the moving frame additionally carries a smooth 900-count bump.  Inside the
    bump |mu_ref - mu_mov| reaches ~900 against a denominator of ~45, i.e. a raw
    z-difference of ~20 — twice the clip ceiling — so clipped and unclipped
    output are unambiguously distinguishable.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    base = rng.random((size, size)) * 100.0
    bump = 900.0 * np.exp(-(((yy - size // 2) ** 2 + (xx - size // 2) ** 2) / (2 * 14.0**2)))
    return base.astype(np.float32), (base + bump).astype(np.float32)


def test_b062_local_zscore_difference_clip_none_runs():
    """local_zscore_difference(clip=None) returns a genuinely UNCLIPPED map (values > 10 on data engineered to exceed the clip ceiling) and the default path equals np.clip(raw, 0, 10)/10 of it. Regression for B-062 (fixed in c0d03ec)."""
    rng = np.random.default_rng(1)
    I_ref = rng.random((64, 64)).astype(np.float32)
    I_mov = rng.random((64, 64)).astype(np.float32)

    D_none = reliableAnalysis.local_zscore_difference(I_ref, I_mov, clip=None)
    assert D_none.shape == (64, 64)
    assert D_none.dtype == np.float32

    D_default = reliableAnalysis.local_zscore_difference(I_ref, I_mov)
    assert D_default.min() >= 0.0
    assert D_default.max() <= 1.0

    # The assertions above are all satisfied by a map of zeros.  Redo both calls
    # on data whose raw z-difference is ~20, i.e. twice the default clip ceiling.
    J_ref, J_mov = _over_clip_pair()
    raw = reliableAnalysis.local_zscore_difference(J_ref, J_mov, clip=None)
    clipped = reliableAnalysis.local_zscore_difference(J_ref, J_mov)

    # (a) clip=None really does skip the clip AND the /(clip[1]-clip[0]) rescale
    assert float(raw.max()) > 10.0, f"clip=None output never exceeds the ceiling: max={raw.max()}"
    assert int((raw > 1.0).sum()) > 100
    # (b) the raw map is the documented statistic, not an arbitrary large array
    assert np.allclose(raw, _zscore_difference_float64(J_ref, J_mov, clip=None), atol=0.01), (
        f"raw map diverges from the float64 oracle: "
        f"max|diff|={np.abs(raw - _zscore_difference_float64(J_ref, J_mov, clip=None)).max():.3e}"
    )
    # (c) the default path is exactly the clip-and-rescale of that same raw map,
    #     and the clip genuinely engages (saturated pixels sit exactly at 1.0)
    assert np.array_equal(clipped, np.clip(raw, 0.0, 10.0) / 10.0)
    assert float(clipped.max()) == pytest.approx(1.0, abs=1e-6)
    assert int((clipped >= 1.0).sum()) == int((raw >= 10.0).sum()) > 0
    assert float(clipped.std()) > 0.0

    # (d) the rescale divides by the clip WIDTH, not by the ceiling.  At the
    #     default clip=(0, 10) those coincide, so (c) above cannot tell them
    #     apart; an offset window separates them (width 10 vs ceiling 12).
    offset = reliableAnalysis.local_zscore_difference(J_ref, J_mov, clip=(2.0, 12.0))
    assert np.array_equal(offset, (np.clip(raw, 2.0, 12.0) / 10.0).astype(np.float32))
    assert float(offset.min()) == pytest.approx(0.2, abs=1e-6)


def test_b064_local_mind_difference_debug_dir_on_numpy_fallback(tmp_path):
    """local_mind_difference(debug_dir=...) runs on the numpy fallback and writes all 4 debug tifs. Regression for B-064 (fixed in fcbc933)."""
    if not numpy_fallback:
        pytest.skip("cupy present: numpy-fallback path not active")

    rng = np.random.default_rng(2)
    I_ref = rng.random((32, 32)).astype(np.float32)
    I_mov = rng.random((32, 32)).astype(np.float32)
    debug_dir = tmp_path / "mind_debug"

    diff = reliableAnalysis.local_mind_difference(I_ref, I_mov, debug_dir=str(debug_dir))

    assert diff.shape == (32, 32)
    for name in ["M_ref.tif", "diff_raw.tif", "weight_map.tif", "diff_weighted.tif"]:
        assert (debug_dir / name).exists(), f"missing debug tif: {name}"
