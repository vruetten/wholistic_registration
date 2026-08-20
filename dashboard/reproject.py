"""GPU reprojection for the QC dashboard.

Rebuilds the phase field for one frame as `phase = base + motion_current`
(base = (x_index, y_index, z_init[k]), z_init[k] = 20 + 10*k; verified
identity, SPEC.md "Measured constants"), upsamples that field and a chosen
frame's moving-membrane values in XY the way the pipeline does, then hands
both to the pipeline's own GPU splat
(wholistic_registration.utils.calFlowCrossResolution.
project_coords_to_fixed_planes_gpu) or to a vendored port of the pipeline's
trilinear reference-space scatter
(src/wholistic_registration/tests/run_F260517_0625_qc.py:307-371,
`_scatter_trilinear_to_refspace`). The scatter is vendored, not imported,
because its source module selects a GPU device and imports pandas/zarr/
skimage at module import time (see "Spec gaps" in the implementer report);
the vendored copy calls the same cupyx.scatter_add primitive the original
does and does not change the trilinear-weight arithmetic. The XY upsample
helpers (tests/f260517_helpers.py:354-427,
`upsample_phase_xy_for_supersurface` / `upsample_values_xy_for_supersurface`)
are ported the same way, GPU-resident via `wholistic_registration.utils.
cupy_ndimage.map_coordinates` instead of host `scipy.ndimage.map_coordinates`,
so this module never imports scipy directly (SPEC.md: reproject.py's
dependency list is numpy, cupy, and the repo's utils only) and never leaves
the GPU between upsampling and splatting.

Reconstructing phase from base + motion, rather than reading the bundle's
already-computed phase_plane, is deliberate: it is what would let a caller
splice in a modified motion field later without a fresh registration run,
per SPEC.md #3. The public functions here take no such modified-field
argument yet -- only field_frame/image_frame -- so today's reconstructed
phase is numerically the stored phase_plane (residual ~6e-5, per SPEC.md).

Imports restricted per SPEC.md #3 to numpy, cupy, and the repo's utils for
project_to_planes/scatter_to_refspace themselves. os, sys, time, tifffile
and qc_bundle are added only for the --selftest CLI entry point below (run
discovery, timing, and reading the stored refspace TIFF to compare against);
tifffile is in the dashboard's general allowed dependency list (SPEC.md
top), and qc_bundle is the one sibling import SPEC.md #3 permits.
"""
import os
import sys
import time

import cupy as cp
import cupyx
import numpy as np

try:
    from wholistic_registration.utils import cupy_ndimage
    from wholistic_registration.utils.calFlowCrossResolution import (
        project_coords_to_fixed_planes_gpu,
    )
except ModuleNotFoundError:
    # Measured on ws1: the env's registered editable install of
    # wholistic_registration points (via its .pth file) at a deleted
    # directory (/groups/ahrens/home/ruttenv/tmp/wr-audit), so the normal
    # import fails even though the real checkout this file ships beside is
    # intact. This does not touch the shared conda env (no pip install) --
    # it only adds this repo's own src/ to this process's sys.path, the
    # same fallback run_F260517_0625_qc.py uses for the same reason.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
    from wholistic_registration.utils import cupy_ndimage
    from wholistic_registration.utils.calFlowCrossResolution import (
        project_coords_to_fixed_planes_gpu,
    )

_OVERRIDABLE_KEYS = ("z_window", "upsample_factor", "xy_splat_mode", "values_interp_order_mem")


def _z_init(k):
    # Same identity qc_bundle._z_init encodes for the same reason (SPEC.md
    # "Measured constants"); duplicated as one line rather than imported
    # because qc_bundle._z_init is private and reproject.py's import surface
    # is capped at SPEC.md #3.
    return 20 + 10 * k


def _resolved_params(bundle, params):
    """bundle.projection_params overridden by the four keys SPEC.md #3 names;
    every other key (fill_value, xy_extra_radius, ref_volume_order, ...)
    always comes from the on-disk JSON."""
    merged = dict(bundle.projection_params)
    for key in _OVERRIDABLE_KEYS:
        if key in params:
            merged[key] = params[key]
    return merged


def _build_phase_and_values(bundle, field_frame, image_frame):
    """(X, Y, K, 3) reconstructed phase field for field_frame and (X, Y, K)
    moving-membrane values for image_frame, both float32 on the CPU (the
    caller uploads and upsamples on GPU). field_frame and image_frame may
    differ -- that is the null test SPEC.md #3 requires be allowed, and
    nothing here special-cases it: the two loops below simply read from two
    independent frame indices."""
    K = bundle.mov_shape_zyx[0]
    Y, X = bundle.mov_shape_zyx[1:]
    x_idx = np.arange(X, dtype=np.float32)
    y_idx = np.arange(Y, dtype=np.float32)
    phase_xyk3 = np.empty((X, Y, K, 3), dtype=np.float32)
    values_xyk = np.empty((X, Y, K), dtype=np.float32)
    for k in range(K):
        motion_yx3 = bundle.motion_plane(field_frame, k)  # (Y, X, 3), comps (x, y, z)
        motion_xy3 = np.transpose(motion_yx3, (1, 0, 2))  # (X, Y, 3)
        phase_xyk3[:, :, k, 0] = x_idx[:, None] + motion_xy3[..., 0]
        phase_xyk3[:, :, k, 1] = y_idx[None, :] + motion_xy3[..., 1]
        phase_xyk3[:, :, k, 2] = _z_init(k) + motion_xy3[..., 2]
        values_xyk[:, :, k] = bundle.mov_plane(image_frame, k).T  # (Y, X) -> (X, Y)
    return phase_xyk3, values_xyk


def _xy_grid_gpu(X, Y, factor):
    Xup, Yup = X * factor, Y * factor
    x_new = cp.linspace(0, X - 1, Xup, dtype=cp.float32)
    y_new = cp.linspace(0, Y - 1, Yup, dtype=cp.float32)
    Xg, Yg = cp.meshgrid(x_new, y_new, indexing="ij")
    return cp.vstack([Xg.ravel(), Yg.ravel()]), Xup, Yup


def _upsample_phase_gpu(phase_xyk3_cpu, factor):
    # Vendored port of tests/f260517_helpers.py:374-400
    # (upsample_phase_xy_for_supersurface): identical sampling grid,
    # identical per-(k, component) call into map_coordinates(order=1,
    # mode="nearest"); ported from scipy.ndimage/numpy to
    # wholistic_registration.utils.cupy_ndimage/cupy so it stays on the GPU.
    # map_coordinates performs the interpolation; nothing here reimplements it.
    X, Y, K, C = phase_xyk3_cpu.shape
    if factor == 1:
        return cp.asarray(phase_xyk3_cpu)
    coords_2d, Xup, Yup = _xy_grid_gpu(X, Y, factor)
    phase_src = cp.asarray(phase_xyk3_cpu)
    out = cp.empty((Xup, Yup, K, C), dtype=cp.float32)
    for k in range(K):
        for c in range(C):
            out[:, :, k, c] = cupy_ndimage.map_coordinates(
                phase_src[:, :, k, c], coords_2d, order=1, mode="nearest"
            ).reshape(Xup, Yup)
    return out


def _upsample_values_gpu(values_xyk_cpu, factor, order):
    # Vendored port of tests/f260517_helpers.py:403-427
    # (upsample_values_xy_for_supersurface); same note as _upsample_phase_gpu.
    X, Y, K = values_xyk_cpu.shape
    if factor == 1:
        return cp.asarray(values_xyk_cpu)
    coords_2d, Xup, Yup = _xy_grid_gpu(X, Y, factor)
    values_src = cp.asarray(values_xyk_cpu)
    out = cp.empty((Xup, Yup, K), dtype=cp.float32)
    for k in range(K):
        out[:, :, k] = cupy_ndimage.map_coordinates(
            values_src[:, :, k], coords_2d, order=order, mode="nearest"
        ).reshape(Xup, Yup)
    return out


def _scatter_trilinear_to_refspace(coords_xyz, values_xyk, ref_shape_zyx, eps=1e-6):
    # Vendored port of
    # src/wholistic_registration/tests/run_F260517_0625_qc.py:307-371
    # (same function name), line for line: the trilinear weights and the
    # cupyx.scatter_add calls are unchanged from the source. Not imported --
    # see the module docstring for why. No z-window gate: every finite,
    # in-bounds sample contributes, matching SPEC.md's refspace_* fill
    # description (NaN where total scattered weight <= eps).
    Zref, Yref, Xref = ref_shape_zyx
    coords = cp.asarray(coords_xyz, dtype=cp.float32)
    values = cp.asarray(values_xyk, dtype=cp.float32)
    x_ref, y_ref, z_ref = coords[..., 0], coords[..., 1], coords[..., 2]

    valid = (
        cp.isfinite(x_ref) & cp.isfinite(y_ref) & cp.isfinite(z_ref) & cp.isfinite(values)
        & (x_ref >= 0) & (x_ref <= Xref - 1)
        & (y_ref >= 0) & (y_ref <= Yref - 1)
        & (z_ref >= 0) & (z_ref <= Zref - 1)
    )
    valid_w = valid.astype(cp.float32)
    x_s = cp.where(valid, x_ref, cp.float32(0.0))
    y_s = cp.where(valid, y_ref, cp.float32(0.0))
    z_s = cp.where(valid, z_ref, cp.float32(0.0))
    val_s = cp.where(valid, values, cp.float32(0.0))
    del coords, values, x_ref, y_ref, z_ref, valid

    x0 = cp.floor(x_s).astype(cp.int32); x1 = cp.minimum(x0 + 1, Xref - 1)
    y0 = cp.floor(y_s).astype(cp.int32); y1 = cp.minimum(y0 + 1, Yref - 1)
    z0 = cp.floor(z_s).astype(cp.int32); z1 = cp.minimum(z0 + 1, Zref - 1)
    fx, fy, fz = x_s - x0, y_s - y0, z_s - z0
    del x_s, y_s, z_s

    sum_val = cp.zeros(Zref * Yref * Xref, dtype=cp.float32)
    sum_w = cp.zeros(Zref * Yref * Xref, dtype=cp.float32)
    val_flat = val_s.ravel()
    for zi, wz in ((z0, 1.0 - fz), (z1, fz)):
        for yi, wy in ((y0, 1.0 - fy), (y1, fy)):
            for xi, wx in ((x0, 1.0 - fx), (x1, fx)):
                w = (wx * wy * wz * valid_w).ravel()
                idx = ((zi * Yref + yi) * Xref + xi).ravel()
                cupyx.scatter_add(sum_val, idx, w * val_flat)
                cupyx.scatter_add(sum_w, idx, w)
    del x0, x1, y0, y1, z0, z1, fx, fy, fz, valid_w, val_s, val_flat

    sum_val = sum_val.reshape(Zref, Yref, Xref)
    sum_w = sum_w.reshape(Zref, Yref, Xref)
    occupied = sum_w > eps
    out = cp.where(occupied, sum_val / cp.maximum(sum_w, eps), cp.float32(np.nan))
    out_np = cp.asnumpy(out)
    del sum_val, sum_w, occupied, out
    return out_np


def _estimate_upsampled_bytes(bundle, factor):
    K = bundle.mov_shape_zyx[0]
    Y, X = bundle.mov_shape_zyx[1:]
    Xup, Yup = X * factor, Y * factor
    coords_bytes = Xup * Yup * K * 3 * 4
    values_bytes = Xup * Yup * K * 4
    return coords_bytes, values_bytes


def _check_vram(required_bytes, label):
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    if required_bytes > free_bytes:
        raise RuntimeError(
            f"{label}: estimated {required_bytes / 2**20:.0f} MiB needed, only "
            f"{free_bytes / 2**20:.0f} MiB free of {total_bytes / 2**20:.0f} MiB total "
            f"on GPU {cp.cuda.Device().id}."
        )


def _free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def project_to_planes(bundle, field_frame, image_frame, params):
    """z-window-gated splat of image_frame's moving-membrane values onto the
    20 fixed reference z planes, along field_frame's reconstructed phase
    field. Wraps utils.calFlowCrossResolution.project_coords_to_fixed_planes_gpu
    with values_xyk supplied (the moving-signal-splat mode that function's
    own docstring calls out), so ref_volume's contents are never read --
    only its (X, Y) shape is (verified by reading the function body: the
    ref_xyz variable it derives from ref_volume is referenced only inside
    the `values_xyk is None` branch, which this call never takes). The
    dummy_ref array below exists purely to carry that shape.
    Returns (20, 1500, 630) float32, fill value from projection_params.json.
    """
    p = _resolved_params(bundle, params)
    factor = int(p["upsample_factor"])
    _check_vram(
        3 * sum(_estimate_upsampled_bytes(bundle, factor))
        + bundle.fixed_target_z.shape[0] * bundle.mov_shape_zyx[1] * bundle.mov_shape_zyx[2] * 4,
        "project_to_planes",
    )
    phase_cpu, values_cpu = _build_phase_and_values(bundle, field_frame, image_frame)
    phase_up = _upsample_phase_gpu(phase_cpu, factor)
    values_up = _upsample_values_gpu(values_cpu, factor, int(p["values_interp_order_mem"]))
    Yref, Xref = bundle.mov_shape_zyx[1:]
    dummy_ref = cp.zeros((Xref, Yref, 1), dtype=cp.float32)  # shape carrier only, see docstring
    out = project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_up,
        ref_volume=dummy_ref,
        target_z_planes=bundle.fixed_target_z,
        values_xyk=values_up,
        ref_volume_order="xyz",
        z_window=float(p["z_window"]),
        downsample_xy=1,  # fixed: the (20, 1500, 630) return-shape contract requires it
        fill_value=float(bundle.projection_params["fill_value"]),
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode=p["xy_splat_mode"],
        xy_extra_radius=int(bundle.projection_params.get("xy_extra_radius", 0)),
    )
    del phase_up, values_up, dummy_ref
    _free_gpu()
    assert out.shape == (bundle.fixed_target_z.shape[0],) + bundle.mov_shape_zyx[1:]
    return out.astype(np.float32, copy=False)


def scatter_to_refspace(bundle, field_frame, image_frame, params):
    """Ungated trilinear scatter of image_frame's moving-membrane values into
    the full (220, 1500, 630) reference grid, along field_frame's
    reconstructed phase field. z_window and xy_splat_mode from params are
    accepted but unused: the trilinear scatter has no z-window gate and no
    XY-splat-mode choice (always the 8-corner trilinear footprint), matching
    the vendored source function. Returns float32, NaN where the total
    scattered weight is <= 1e-6.
    """
    p = _resolved_params(bundle, params)
    factor = int(p["upsample_factor"])
    # A first-principles estimate (grid arrays + a constant multiple of the
    # upsampled coords/values) measured 2.7x too low on ws1: this function's
    # 8-corner accumulation loop leaves ~15 same-shape temporaries live at
    # once, and cupy's caching allocator does not return freed blocks to the
    # driver until free_all_blocks(), so device free-memory delta reflects
    # everything the pool touched, not the live working set. Anchoring to
    # the measured value directly, scaled quadratically in upsample_factor
    # (the dominant arrays are Xup*Yup, proportional to factor**2), is more
    # honest than re-deriving a bound that was already wrong once.
    # Measured: 13388 MiB peak device-memory delta, factor=2, frame 2,
    # f260517_0625_qc_v4 (ws1, RTX A4000; see implementer report).
    # Margin is 1.05x, not the usual larger safety factor: the measured peak
    # (13388 MiB) already leaves only ~2 GiB of this GPU's 16084 MiB total
    # unused, so a bigger multiplicative margin would reject the exact call
    # just measured to succeed. 1.05x still catches a device with
    # meaningfully less free memory than ws1's GPU 0 at idle.
    _MEASURED_PEAK_MIB_AT_FACTOR2 = 13388
    required_bytes = int(_MEASURED_PEAK_MIB_AT_FACTOR2 * (factor / 2) ** 2 * 1.05 * 2**20)
    _check_vram(required_bytes, "scatter_to_refspace")
    phase_cpu, values_cpu = _build_phase_and_values(bundle, field_frame, image_frame)
    phase_up = _upsample_phase_gpu(phase_cpu, factor)
    values_up = _upsample_values_gpu(values_cpu, factor, int(p["values_interp_order_mem"]))
    out_np = _scatter_trilinear_to_refspace(phase_up, values_up, bundle.ref_shape_zyx)
    del phase_up, values_up
    _free_gpu()
    assert out_np.shape == bundle.ref_shape_zyx
    return out_np.astype(np.float32, copy=False)


# ---- --selftest CLI ----------------------------------------------------


def _report_diff(label, a, b):
    both_finite = np.isfinite(a) & np.isfinite(b)
    frac_both_finite = float(np.mean(both_finite))
    if np.any(both_finite):
        d = np.abs(a[both_finite] - b[both_finite])
        print(
            f"{label}: frac_both_finite={frac_both_finite:.4f} "
            f"absdiff mean={d.mean():.6g} median={np.median(d):.6g} "
            f"p95={np.percentile(d, 95):.6g} max={d.max():.6g}"
        )
    else:
        print(f"{label}: frac_both_finite={frac_both_finite:.4f} (no overlap)")


def _timed(label, fn):
    # project_to_planes/scatter_to_refspace free their own GPU memory before
    # returning (SPEC.md #3), so sampling the memory pool after the call
    # always reads back near zero -- measured, not assumed: an earlier
    # version of this function did exactly that and printed 0 MiB on every
    # call. A background thread polling device free memory during the call
    # is the fix: min free bytes observed - free bytes at the call's start
    # is a peak-usage estimate, not a byte-exact high-water mark (sampling
    # is not synchronised to the GPU's own allocations).
    import threading

    free_before, _ = cp.cuda.Device().mem_info
    min_free = [free_before]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            f, _ = cp.cuda.Device().mem_info
            if f < min_free[0]:
                min_free[0] = f
            stop.wait(0.01)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    t0 = time.perf_counter()
    out = fn()
    elapsed_s = time.perf_counter() - t0
    stop.set()
    poller.join()
    peak_mib = (free_before - min_free[0]) / 2**20
    print(f"{label}: elapsed={elapsed_s:.3f}s peak_gpu_delta~={peak_mib:.0f} MiB (polled, not exact)")
    return out


def _selftest(run_dir, cache_dir):
    import tifffile

    import qc_bundle

    bundle = qc_bundle.QCBundle(run_dir, cache_dir)
    free0, total0 = cp.cuda.Device().mem_info
    print(f"GPU {cp.cuda.Device().id}: {free0/2**20:.0f} MiB free of {total0/2**20:.0f} MiB total")

    print("=== 1. reproject frame 2 (on-disk params) vs stored refspace_mem ===")
    stored_path = os.path.join(run_dir, "refspace_mem", "vol_F260517_refspace_mem_000002.tif")
    stored = tifffile.imread(stored_path).astype(np.float32)
    mine = _timed("scatter_to_refspace(field=2,image=2)", lambda: scatter_to_refspace(bundle, 2, 2, {}))
    assert mine.shape == stored.shape, (mine.shape, stored.shape)
    _report_diff("frame2 mine vs stored", mine, stored)

    print("=== 2. null test: frame2 field on frame3 image vs frame3's own reprojection ===")
    null = _timed(
        "scatter_to_refspace(field=2,image=3)", lambda: scatter_to_refspace(bundle, 2, 3, {})
    )
    own3 = _timed(
        "scatter_to_refspace(field=3,image=3)", lambda: scatter_to_refspace(bundle, 3, 3, {})
    )
    _report_diff("frame3-image via frame2-field vs frame3-field", null, own3)
    n_differ = int(np.sum(np.isfinite(null) & np.isfinite(own3) & (null != own3)))
    print(f"voxels finite in both and numerically different: {n_differ}")

    print("=== 3. z_window sensitivity (project_to_planes, frame 2) ===")
    for zw in (3.0, 6.0):
        planes = _timed(
            f"project_to_planes(z_window={zw})",
            lambda zw=zw: project_to_planes(bundle, 2, 2, {"z_window": zw}),
        )
        fill = float(bundle.projection_params["fill_value"])
        valid_frac = float(np.mean(planes != fill))
        print(f"z_window={zw}: fraction != fill_value({fill}) = {valid_frac:.4f}")

    print("=== selftest OK ===")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _RUN_DIR = (
            "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/"
            "registration_out/f260517_0625_qc_v4"
        )
        _CACHE_DIR = "/tmp/qc_bundle_cache_selftest"
        _selftest(_RUN_DIR, _CACHE_DIR)
