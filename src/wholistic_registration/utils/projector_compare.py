"""
Side-by-side comparison of the three reference-plane projectors.

Given a single frame's forward map (``phase_new``) and the moving signal, run:

  * **max-splat B** -- ``project_coords_to_fixed_planes_gpu`` with the subpixel
    footprint, max combine. This is what the F260517 pipeline currently uses.
  * **weighted-avg B** -- ``project_coords_to_fixed_planes_weighted_gpu``,
    bilinear splat with weighted-average combine (intensity faithful, free
    coverage map).
  * **inverse-gather C** -- ``project_coords_inverse_gather_gpu``, backward warp
    (scatter indices, gather intensity once).

and report, per method: runtime, coverage (hole) fraction, an intensity summary
(mean / median / p99 / max -- the max-vs-average bias shows up here), and the
output volume + coverage map for plotting. Also reports cross-method agreement
on commonly covered pixels (weighted-B vs C should agree closely in smooth
regions, per the analysis).
"""

import time

import numpy as np

from . import CUPY_AVAILABLE, cp
from . import calFlowCrossResolution as cf


def _sync():
    """Block until queued GPU work finishes, so timing is honest."""
    if CUPY_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()


def _intensity_summary(volume, covered):
    """Summary stats over covered pixels only."""
    vals = np.asarray(volume)[np.asarray(covered)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def _agreement(a, vol_a, b, vol_b, mask):
    """Mean / p95 absolute difference between two volumes on a shared mask."""
    mask = np.asarray(mask)
    if mask.sum() == 0:
        return {"pair": f"{a}_vs_{b}", "n": 0, "mae": np.nan, "p95_abs": np.nan}
    da = np.asarray(vol_a)[mask] - np.asarray(vol_b)[mask]
    da = da[np.isfinite(da)]
    return {
        "pair": f"{a}_vs_{b}",
        "n": int(da.size),
        "mae": float(np.mean(np.abs(da))),
        "p95_abs": float(np.percentile(np.abs(da), 95)),
    }


def compare_projectors(
    coords_ref_xyk_xyz,
    ref_volume,
    target_z_planes,
    values_xyk,
    ref_volume_order="xyz",
    z_window=1.5,
    z_sigma=None,
    z_weight_mode="gaussian",
    downsample_xy=1,
    fill_value=0.0,
    xy_extra_radius=0,
    eps=1e-6,
):
    """
    Run all three projectors on one frame and return a results dict.

    Returns
    -------
    dict with keys:
        "methods": {name: {"out": ndarray (Nplanes,Yout,Xout),
                           "covered": bool ndarray,
                           "runtime_s": float,
                           "hole_fraction": float,
                           "intensity": {...}}}
        "agreement": [ {pair, n, mae, p95_abs}, ... ]
        "params": {...}
    """
    coords = np.asarray(coords_ref_xyk_xyz, dtype=np.float32)
    values = np.asarray(values_xyk, dtype=np.float32)
    ones = np.ones_like(values)

    methods = {}

    # ----------------------------------------------------------------
    # max-splat B (current pipeline projector)
    # ----------------------------------------------------------------
    _sync()
    t0 = time.perf_counter()
    out_max = cf.project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=coords,
        ref_volume=ref_volume,
        target_z_planes=target_z_planes,
        values_xyk=values,
        ref_volume_order=ref_volume_order,
        z_window=z_window,
        downsample_xy=downsample_xy,
        fill_value=fill_value,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=xy_extra_radius,
    )
    _sync()
    rt_max = time.perf_counter() - t0
    # coverage for max-splat: project an all-ones signal; covered where ~1.
    cov_max_raw = cf.project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=coords,
        ref_volume=ref_volume,
        target_z_planes=target_z_planes,
        values_xyk=ones,
        ref_volume_order=ref_volume_order,
        z_window=z_window,
        downsample_xy=downsample_xy,
        fill_value=0.0,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=xy_extra_radius,
    )
    cov_max = np.asarray(cov_max_raw) > 0.5
    methods["max_splat_B"] = {
        "out": np.asarray(out_max),
        "covered": cov_max,
        "runtime_s": rt_max,
        "hole_fraction": float(1.0 - cov_max.mean()),
        "intensity": _intensity_summary(out_max, cov_max),
    }

    # ----------------------------------------------------------------
    # weighted-average B
    # ----------------------------------------------------------------
    _sync()
    t0 = time.perf_counter()
    out_w, weight = cf.project_coords_to_fixed_planes_weighted_gpu(
        coords_ref_xyk_xyz=coords,
        ref_volume=ref_volume,
        target_z_planes=target_z_planes,
        values_xyk=values,
        ref_volume_order=ref_volume_order,
        z_window=z_window,
        z_sigma=z_sigma,
        z_weight_mode=z_weight_mode,
        downsample_xy=downsample_xy,
        fill_value=fill_value,
        return_numpy=True,
        output_order="zyx",
        eps=eps,
    )
    _sync()
    rt_w = time.perf_counter() - t0
    cov_w = np.asarray(weight) > eps
    methods["weighted_B"] = {
        "out": np.asarray(out_w),
        "covered": cov_w,
        "runtime_s": rt_w,
        "hole_fraction": float(1.0 - cov_w.mean()),
        "intensity": _intensity_summary(out_w, cov_w),
    }

    # ----------------------------------------------------------------
    # inverse-gather C
    # ----------------------------------------------------------------
    _sync()
    t0 = time.perf_counter()
    out_c, valid_c = cf.project_coords_inverse_gather_gpu(
        coords_ref_xyk_xyz=coords,
        ref_volume=ref_volume,
        target_z_planes=target_z_planes,
        values_xyk=values,
        ref_volume_order=ref_volume_order,
        z_window=z_window,
        z_sigma=z_sigma,
        z_weight_mode=z_weight_mode,
        downsample_xy=downsample_xy,
        fill_value=fill_value,
        return_numpy=True,
        output_order="zyx",
        eps=eps,
    )
    _sync()
    rt_c = time.perf_counter() - t0
    valid_c = np.asarray(valid_c)
    methods["inverse_gather_C"] = {
        "out": np.asarray(out_c),
        "covered": valid_c,
        "runtime_s": rt_c,
        "hole_fraction": float(1.0 - valid_c.mean()),
        "intensity": _intensity_summary(out_c, valid_c),
    }

    # ----------------------------------------------------------------
    # cross-method agreement on commonly covered pixels
    # ----------------------------------------------------------------
    both_wc = cov_w & valid_c
    both_mw = cov_max & cov_w
    agreement = [
        _agreement("weighted_B", out_w, "inverse_gather_C", out_c, both_wc),
        _agreement("max_splat_B", out_max, "weighted_B", out_w, both_mw),
    ]

    return {
        "methods": methods,
        "agreement": agreement,
        "params": {
            "z_window": z_window,
            "z_sigma": z_sigma,
            "z_weight_mode": z_weight_mode,
            "downsample_xy": downsample_xy,
            "xy_extra_radius": xy_extra_radius,
            "n_planes": int(np.atleast_1d(target_z_planes).size),
            "moving_shape": list(values.shape),
        },
    }


def format_metrics_table(results):
    """Return a printable text table from a ``compare_projectors`` result."""
    lines = []
    lines.append(
        f"{'method':<18} {'runtime_s':>10} {'hole_frac':>10} "
        f"{'mean':>10} {'median':>10} {'p99':>10} {'max':>10}"
    )
    lines.append("-" * 82)
    for name, m in results["methods"].items():
        it = m["intensity"]
        lines.append(
            f"{name:<18} {m['runtime_s']:>10.4f} {m['hole_fraction']:>10.4f} "
            f"{it['mean']:>10.2f} {it['median']:>10.2f} {it['p99']:>10.2f} "
            f"{it['max']:>10.2f}"
        )
    lines.append("")
    lines.append("agreement (abs diff on shared coverage):")
    for a in results["agreement"]:
        lines.append(
            f"  {a['pair']:<32} n={a['n']:>10}  mae={a['mae']:>10.3f}  "
            f"p95={a['p95_abs']:>10.3f}"
        )
    return "\n".join(lines)
