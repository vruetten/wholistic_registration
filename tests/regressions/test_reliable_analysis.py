"""Regression tests for utils/reliableAnalysis.py findings (audit pass 1)."""

import numpy as np
import pytest

from wholistic_registration.utils import cp, reliableAnalysis

numpy_fallback = cp is np


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


def test_b061_local_zscore_difference_no_nans_at_high_offset():
    """local_zscore_difference produces 0 NaN pixels on high-offset float32 data (negative variances are clamped). Regression for B-061 (fixed in e41224c)."""
    I_ref, I_mov = _high_offset_pair()

    D = reliableAnalysis.local_zscore_difference(I_ref, I_mov)

    assert int(np.isnan(D).sum()) == 0


def test_b062_local_zscore_difference_clip_none_runs():
    """local_zscore_difference(clip=None) runs (returns the raw z-difference); the default clip path stays in [0,1]. Regression for B-062 (fixed in c0d03ec)."""
    rng = np.random.default_rng(1)
    I_ref = rng.random((64, 64)).astype(np.float32)
    I_mov = rng.random((64, 64)).astype(np.float32)

    D_none = reliableAnalysis.local_zscore_difference(I_ref, I_mov, clip=None)
    assert D_none.shape == (64, 64)
    assert D_none.dtype == np.float32

    D_default = reliableAnalysis.local_zscore_difference(I_ref, I_mov)
    assert D_default.min() >= 0.0
    assert D_default.max() <= 1.0


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
