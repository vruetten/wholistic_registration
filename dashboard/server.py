"""stdlib HTTP server for the QC dashboard. Routes per dashboard/SPEC.md's
"HTTP route table". Owns HTTP only: parameter validation, response encoding,
an in-process LRU cache of rendered PNG bytes, and in-memory reproject job
storage. All data access goes through qc_bundle.QCBundle; all array-to-PNG
rendering goes through render.py. Neither sibling module is edited here.

Two boundary compromises, both because this server targets the qc_bundle.py
that actually ships (5-frame smoke-test interface: a single `bundle.frames`
list, no `frames_for`, `cache_state`, or `field_summary`), not the aspirational
200-frame interface SPEC.md's Correction 3 describes for a future qc_bundle.py:

1. `/api/summary` needs `field_summary()`, which qc_bundle.py does not have.
   `_compute_field_summary` below computes it here from `motion_plane`, a
   computation SPEC.md assigns to qc_bundle.py.
2. `/api/manifest`'s `kinds` and `cache_state` fields use `frames_for`/
   `cache_state` via `hasattr` and fall back to `bundle.frames` for every kind
   and to introspecting the private `_zmajor_cache` dict, respectively.

`reproject.py` is imported lazily inside the POST /api/reproject handler only,
wrapped in try/except ImportError, so every other route works even if that
sibling file is absent or broken.
"""
import argparse
import json
import logging
import math
import mimetypes
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import qc_bundle
import render

_DEFAULT_RUN_DIR = (
    "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/"
    "registration_out/f260517_0625_qc_v4"
)
_DEFAULT_CACHE_DIR = "/tmp/qc_dashboard_cache"
_DEFAULT_PORT = 8787
_HOST = "127.0.0.1"  # SPEC.md: bind 127.0.0.1 only; no --host flag.

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 256 MiB (assumed, not derived from a measurement): SPEC.md requires the PNG
# cache be "bounded in bytes" but names no number.
PNG_CACHE_MAX_BYTES = 256 * 1024 * 1024

ALLOWED_CMAPS = ("gray", "magma", "diverging")


class BadRequest(Exception):
    """Raised by parameter parsing; carries the offending parameter name so
    the 400 response names it, per SPEC.md's route-table requirement."""

    def __init__(self, param, msg):
        super().__init__(msg)
        self.param = param

    def to_json(self):
        return {"error": "%s (parameter: %s)" % (str(self), self.param)}


# ---- query-parameter parsing helpers --------------------------------------


def _params(query):
    parsed = urllib.parse.parse_qs(query, keep_blank_values=True)
    return {k: v[-1] for k, v in parsed.items()}


def _check_unknown(params, allowed):
    extra = sorted(set(params) - set(allowed))
    if extra:
        raise BadRequest(extra[0], "unknown parameter '%s'" % extra[0])


def _get_int(params, name, default=None, required=False, choices=None):
    if name not in params:
        if required:
            raise BadRequest(name, "missing required parameter '%s'" % name)
        return default
    raw = params[name]
    try:
        val = int(raw)
    except ValueError:
        raise BadRequest(name, "'%s' must be an integer, got %r" % (name, raw))
    if choices is not None and val not in choices:
        raise BadRequest(name, "'%s'=%r must be one of %s" % (name, val, choices))
    return val


def _get_float(params, name, default=None, lo=None, hi=None):
    if name not in params:
        return default
    raw = params[name]
    try:
        val = float(raw)
    except ValueError:
        raise BadRequest(name, "'%s' must be a number, got %r" % (name, raw))
    if math.isnan(val) or math.isinf(val):
        raise BadRequest(name, "'%s' must be finite, got %r" % (name, raw))
    if lo is not None and val < lo:
        raise BadRequest(name, "'%s'=%r below minimum %r" % (name, val, lo))
    if hi is not None and val > hi:
        raise BadRequest(name, "'%s'=%r above maximum %r" % (name, val, hi))
    return val


def _get_choice(params, name, choices, default=None, required=False):
    if name not in params:
        if required:
            raise BadRequest(name, "missing required parameter '%s'" % name)
        return default
    val = params[name]
    if val not in choices:
        raise BadRequest(name, "'%s'=%r must be one of %s" % (name, val, choices))
    return val


def _get_bool01(params, name, default):
    return bool(_get_int(params, name, default=default, choices=(0, 1)))


def _frame(params, bundle, name="frame"):
    val = _get_int(params, name, required=True)
    if val not in bundle.frames:
        raise BadRequest(name, "'%s'=%r not in available frames %s" % (name, val, bundle.frames))
    return val


def _kplane(params, bundle, name="k"):
    val = _get_int(params, name, required=True)
    nz = bundle.mov_shape_zyx[0]
    if not (0 <= val < nz):
        raise BadRequest(name, "'%s'=%r out of range [0, %d)" % (name, val, nz))
    return val


def _zplane(params, bundle, name, nz):
    val = _get_int(params, name, required=True)
    if not (0 <= val < nz):
        raise BadRequest(name, "'%s'=%r out of range [0, %d)" % (name, val, nz))
    return val


def _pct_pair(params):
    lo_pct = _get_float(params, "lo_pct", default=1.0, lo=0.0, hi=100.0)
    hi_pct = _get_float(params, "hi_pct", default=99.0, lo=0.0, hi=100.0)
    if lo_pct >= hi_pct:
        raise BadRequest("hi_pct", "'hi_pct'=%r must exceed 'lo_pct'=%r" % (hi_pct, lo_pct))
    return lo_pct, hi_pct


def _downsample_param(params):
    val = _get_int(params, "downsample", default=1)
    if val < 1:
        raise BadRequest("downsample", "'downsample'=%r must be >= 1" % val)
    return val


def _resolve_limits(cmap, arr, lo_pct, hi_pct):
    """Signed quantities go through diverging_limits so the scale is
    symmetric about zero; a diverging colormap paired with an asymmetric
    percentile pair is rejected rather than silently resolved."""
    if cmap == "diverging":
        if abs(lo_pct - (100.0 - hi_pct)) > 1e-6:
            raise BadRequest(
                "lo_pct",
                "cmap='diverging' requires symmetric percentiles (lo_pct == 100-hi_pct); "
                "got lo_pct=%r hi_pct=%r" % (lo_pct, hi_pct),
            )
        return render.diverging_limits(arr, pct=hi_pct)
    return render.contrast_limits(arr, lo_pct=lo_pct, hi_pct=hi_pct)


def _downsample(arr, factor):
    if factor <= 1:
        return arr
    return np.ascontiguousarray(arr[::factor, ::factor])


# ---- /img/motion ------------------------------------------------------------

_MOTION_PARAMS = {
    "frame", "k", "quantity", "cmap", "lo_pct", "hi_pct", "base", "alpha",
    "mask", "quiver", "stride", "downsample",
}


def _parse_motion(query, bundle):
    p = _params(query)
    _check_unknown(p, _MOTION_PARAMS)
    frame = _frame(p, bundle)
    k = _kplane(p, bundle)
    quantity = _get_choice(p, "quantity", ("norm", "dx", "dy", "dz"), default="norm")
    default_cmap = "diverging" if quantity in ("dx", "dy", "dz") else "magma"
    cmap = _get_choice(p, "cmap", ALLOWED_CMAPS, default=default_cmap)
    if quantity in ("dx", "dy", "dz") and cmap != "diverging":
        raise BadRequest(
            "cmap", "signed quantity '%s' requires cmap='diverging', got %r" % (quantity, cmap)
        )
    lo_pct, hi_pct = _pct_pair(p)
    base = _get_choice(p, "base", ("mov", "none"), default="mov")
    alpha = _get_float(p, "alpha", default=0.6, lo=0.0, hi=1.0)
    mask = _get_bool01(p, "mask", default=0)  # SPEC Correction 1: off by default.
    quiver = _get_bool01(p, "quiver", default=0)
    stride = _get_int(p, "stride", default=22)  # SPEC Correction 2: 22, not 11, under budget.
    if stride < 1:
        raise BadRequest("stride", "'stride'=%r must be >= 1" % stride)
    downsample = _downsample_param(p)
    return {
        "frame": frame, "k": k, "quantity": quantity, "cmap": cmap,
        "lo_pct": lo_pct, "hi_pct": hi_pct, "base": base, "alpha": alpha,
        "mask": int(mask), "quiver": int(quiver), "stride": stride,
        "downsample": downsample,
    }


def _render_motion(r, bundle):
    plane = bundle.motion_plane(r["frame"], r["k"])  # (Y, X, 3), comps (x, y, z)
    dx, dy, dz = plane[..., 0], plane[..., 1], plane[..., 2]
    arr = {"dx": dx, "dy": dy, "dz": dz}.get(r["quantity"])
    if arr is None:
        arr = np.sqrt(dx * dx + dy * dy + dz * dz)
    lo, hi = _resolve_limits(r["cmap"], arr, r["lo_pct"], r["hi_pct"])
    over_rgb = render.render_scalar(arr, lo, hi, r["cmap"])
    if r["base"] == "mov":
        mov = bundle.mov_plane(r["frame"], r["k"])
        b_lo, b_hi = render.contrast_limits(mov, lo_pct=r["lo_pct"], hi_pct=r["hi_pct"])
        base_rgb = render.render_scalar(mov, b_lo, b_hi, "gray")
    else:
        base_rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
    # Correction 1: mask=0 blends everywhere; mask=1 gates the overlay by
    # alpha through composite's over_mask, not by the magenta sentinel.
    over_mask = bundle.mask_mov_plane(r["frame"], r["k"]) if r["mask"] else None
    rgb = render.composite(base_rgb, over_rgb, r["alpha"], over_mask=over_mask)
    if r["quiver"]:
        rgb = render.quiver_overlay(rgb, dx, dy, dz=dz, stride=r["stride"])
    return _downsample(rgb, r["downsample"])


# ---- /img/plane ---------------------------------------------------------------

_PLANE_PARAMS = {"kind", "frame", "k", "channel", "cmap", "lo_pct", "hi_pct", "downsample"}


def _parse_plane(query, bundle):
    p = _params(query)
    _check_unknown(p, _PLANE_PARAMS)
    kind = _get_choice(p, "kind", ("mov", "raw_moving", "projected"), required=True)
    frame = _frame(p, bundle)
    k = _kplane(p, bundle)
    if kind == "mov":
        if "channel" in p:
            raise BadRequest("channel", "kind='mov' has no channel dimension")
        channel = None
    else:
        channel = _get_choice(p, "channel", ("mem", "sparseCell"), required=True)
    cmap = _get_choice(p, "cmap", ALLOWED_CMAPS, default="gray")
    lo_pct, hi_pct = _pct_pair(p)
    downsample = _downsample_param(p)
    return {
        "kind": kind, "frame": frame, "k": k, "channel": channel, "cmap": cmap,
        "lo_pct": lo_pct, "hi_pct": hi_pct, "downsample": downsample,
    }


def _render_plane(r, bundle):
    if r["kind"] == "mov":
        arr = bundle.mov_plane(r["frame"], r["k"])
    elif r["kind"] == "raw_moving":
        arr = bundle.raw_moving_plane(r["frame"], r["k"], r["channel"])
    else:
        arr = bundle.projected_plane(r["frame"], r["k"], r["channel"])
    lo, hi = _resolve_limits(r["cmap"], arr, r["lo_pct"], r["hi_pct"])
    rgb = render.render_scalar(arr, lo, hi, r["cmap"])
    return _downsample(rgb, r["downsample"])


# ---- /img/refspace --------------------------------------------------------------

_REFSPACE_PARAMS = {
    "frame", "z", "channel", "source", "overlay", "alpha", "cmap", "lo_pct",
    "hi_pct", "downsample",
}


def _parse_refspace(query, bundle, state):
    p = _params(query)
    _check_unknown(p, _REFSPACE_PARAMS)
    source = p.get("source", "precomputed")
    if source == "precomputed":
        frame = _frame(p, bundle)
        z = _zplane(p, bundle, "z", bundle.ref_shape_zyx[0])
        channel = _get_choice(p, "channel", ("mem", "sparseCell"), required=True)
        job_id = None
    else:
        job_id = source
        with state.jobs_lock:
            job = state.jobs.get(job_id)
        if job is None:
            raise BadRequest("source", "unknown job id %r" % job_id)
        # A job's array (planes: 20 slices; refspace: 220) is addressed by
        # 'z' regardless of mode; validated against that job's own extent so
        # an out-of-range index never silently reads the wrong slice.
        z = _zplane(p, bundle, "z", job["array"].shape[0])
        frame = _get_int(p, "frame", default=None)
        if frame is not None and frame not in bundle.frames:
            raise BadRequest("frame", "'frame'=%r not in available frames %s" % (frame, bundle.frames))
        channel = p.get("channel")  # unused for a job source; the reproject result has none
    overlay = _get_bool01(p, "overlay", default=0)
    alpha = _get_float(p, "alpha", default=0.6, lo=0.0, hi=1.0)
    cmap = _get_choice(p, "cmap", ALLOWED_CMAPS, default="gray")
    lo_pct, hi_pct = _pct_pair(p)
    downsample = _downsample_param(p)
    return {
        "source": source, "job_id": job_id, "frame": frame, "z": z, "channel": channel,
        "overlay": int(overlay), "alpha": alpha, "cmap": cmap, "lo_pct": lo_pct,
        "hi_pct": hi_pct, "downsample": downsample,
    }


def _render_refspace(r, bundle, state):
    if r["source"] == "precomputed":
        arr = bundle.refspace_plane(r["frame"], r["z"], r["channel"])
    else:
        with state.jobs_lock:
            job = state.jobs.get(r["job_id"])
        if job is None:
            raise BadRequest("source", "job %r was freed" % r["job_id"])
        arr = job["array"][r["z"]]
    lo, hi = _resolve_limits(r["cmap"], arr, r["lo_pct"], r["hi_pct"])
    over_rgb = render.render_scalar(arr, lo, hi, r["cmap"])
    if r["overlay"]:
        ref = bundle.refspace_reference_plane(r["z"])
        b_lo, b_hi = render.contrast_limits(ref, lo_pct=r["lo_pct"], hi_pct=r["hi_pct"])
        base_rgb = render.render_scalar(ref, b_lo, b_hi, "gray")
        rgb = render.composite(base_rgb, over_rgb, r["alpha"])
    else:
        rgb = over_rgb
    return _downsample(rgb, r["downsample"])


# ---- /img/coverage --------------------------------------------------------------

_COVERAGE_PARAMS = {"frame", "k", "downsample"}


def _parse_coverage(query, bundle):
    p = _params(query)
    _check_unknown(p, _COVERAGE_PARAMS)
    frame = _frame(p, bundle)
    k = _kplane(p, bundle)
    downsample = _downsample_param(p)
    return {"frame": frame, "k": k, "downsample": downsample}


def _render_coverage(r, bundle):
    arr = bundle.coverage_plane(r["frame"], r["k"]).astype(np.float32)
    rgb = render.render_scalar(arr, 0.0, 1.0, "gray")
    return _downsample(rgb, r["downsample"])


# ---- /img/montage ---------------------------------------------------------------

_MONTAGE_PARAMS = {"axis", "quantity", "k", "frame", "cols", "max_edge", "downsample"}


def _parse_montage(query, bundle):
    p = _params(query)
    _check_unknown(p, _MONTAGE_PARAMS)
    axis = _get_choice(p, "axis", ("time", "z"), default="time")
    quantity = _get_choice(p, "quantity", ("norm", "dx", "dy", "dz"), default="norm")
    cols = _get_int(p, "cols", default=None)
    if cols is not None and cols < 1:
        raise BadRequest("cols", "'cols'=%r must be >= 1" % cols)
    max_edge = _get_int(p, "max_edge", default=2048)
    if max_edge < 1:
        raise BadRequest("max_edge", "'max_edge'=%r must be >= 1" % max_edge)
    downsample = _downsample_param(p)
    if axis == "time":
        if "frame" in p:
            raise BadRequest("frame", "'frame' is not used when axis='time'; use 'k'")
        k = _kplane(p, bundle)
        frame = None
    else:
        if "k" in p:
            raise BadRequest("k", "'k' is not used when axis='z'; use 'frame'")
        frame = _frame(p, bundle)
        k = None
    return {
        "axis": axis, "quantity": quantity, "k": k, "frame": frame,
        "cols": cols, "max_edge": max_edge, "downsample": downsample,
    }


def _montage_default_cmap(quantity):
    return "diverging" if quantity in ("dx", "dy", "dz") else "magma"


def _quantity_component(plane, quantity):
    dx, dy, dz = plane[..., 0], plane[..., 1], plane[..., 2]
    if quantity == "dx":
        return dx
    if quantity == "dy":
        return dy
    if quantity == "dz":
        return dz
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _montage_grid(r, n):
    cols = r["cols"] if r["cols"] is not None else int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return cols, rows


def _montage_tile_scale(r, bundle):
    """Pure function of resolved params and the bundle's fixed plane shape:
    the per-tile scale factor is knowable before rendering, so the
    X-Tile-Scale header is correct on a cache hit too."""
    plane_h, plane_w = bundle.mov_shape_zyx[1], bundle.mov_shape_zyx[2]
    downsample = r["downsample"]
    tile_h = len(range(0, plane_h, downsample))
    tile_w = len(range(0, plane_w, downsample))
    n = len(bundle.frames) if r["axis"] == "time" else bundle.mov_shape_zyx[0]
    cols, rows = _montage_grid(r, n)
    edge = max(rows * tile_h, cols * tile_w)
    internal_factor = (r["max_edge"] / edge) if edge > r["max_edge"] else 1.0
    return (1.0 / downsample) * internal_factor


def _render_montage(r, bundle):
    cmap = _montage_default_cmap(r["quantity"])
    if r["axis"] == "time":
        source_planes = [bundle.motion_plane(f, r["k"]) for f in bundle.frames]
        labels = [str(f) for f in bundle.frames]
    else:
        nz = bundle.mov_shape_zyx[0]
        source_planes = [bundle.motion_plane(r["frame"], zk) for zk in range(nz)]
        labels = [str(zk) for zk in range(nz)]

    tiles = []
    for plane in source_planes:
        arr = _downsample(_quantity_component(plane, r["quantity"]), r["downsample"])
        lo, hi = _resolve_limits(cmap, arr, 1.0, 99.0)  # not exposed as montage params
        tiles.append(render.render_scalar(arr, lo, hi, cmap))

    return render.montage(tiles, cols=r["cols"], labels=labels, max_edge_px=r["max_edge"])


# ---- /api/manifest, /api/metrics, /api/summary --------------------------------


def _frames_for(bundle, kind):
    if hasattr(bundle, "frames_for"):
        return bundle.frames_for(kind)
    return list(bundle.frames)  # this qc_bundle.py has one frame set for every kind


def _cache_state(bundle):
    if hasattr(bundle, "cache_state"):
        return bundle.cache_state()
    warm = sorted(bundle._zmajor_cache.keys())
    return {
        "warm_kind_frame_pairs": [list(t) for t in warm],
        "note": "qc_bundle.QCBundle has no cache_state(); derived from _zmajor_cache",
    }


def api_manifest(bundle):
    m = bundle.manifest()
    kinds = {
        kind: _frames_for(bundle, kind)
        for kind in (
            "motion", "phase", "mov", "mask_mov", "coverage",
            "raw_moving", "projected", "refspace",
        )
    }
    return {
        "run_dir": m["run_dir"],
        "ref_shape_zyx": m["ref_shape_zyx"],
        "mov_shape_zyx": m["mov_shape_zyx"],
        "kinds": kinds,
        "projection_params": m["projection_params"],
        "fixed_target_z": m["fixed_target_z"],
        "cache_state": _cache_state(bundle),
    }


def _mask_support_frac(state, frame):
    with state.mask_frac_lock:
        if frame not in state.mask_frac_cache:
            bundle = state.bundle
            nz, h, w = bundle.mov_shape_zyx
            count = sum(int(bundle.mask_mov_plane(frame, k).sum()) for k in range(nz))
            state.mask_frac_cache[frame] = count / (nz * h * w)
        return state.mask_frac_cache[frame]


def api_metrics(state):
    bundle = state.bundle
    out = dict(bundle.metrics())
    out["mask_support_frac"] = {str(f): _mask_support_frac(state, f) for f in bundle.frames}
    return out


def _compute_field_summary(bundle):
    # SPEC.md Correction 3 assigns field_summary() to qc_bundle.py; the
    # delivered qc_bundle.py does not implement it (see module docstring).
    nz = bundle.mov_shape_zyx[0]
    frames = list(bundle.frames)
    mean = np.zeros((len(frames), nz))
    p95 = np.zeros((len(frames), nz))
    mx = np.zeros((len(frames), nz))
    for fi, frame in enumerate(frames):
        for k in range(nz):
            plane = bundle.motion_plane(frame, k)
            norm = np.linalg.norm(plane, axis=-1)
            finite = norm[np.isfinite(norm)]
            if finite.size == 0:
                continue
            mean[fi, k] = finite.mean()
            p95[fi, k] = np.percentile(finite, 95)
            mx[fi, k] = finite.max()
    return {
        "frames": frames, "mean": mean.tolist(), "p95": p95.tolist(), "max": mx.tolist(),
        "complete": True,
    }


def api_summary(state):
    with state.summary_lock:
        if state.summary_cache is None:
            state.summary_cache = _compute_field_summary(state.bundle)
        return state.summary_cache


def api_jobs(state):
    with state.jobs_lock:
        return [
            {
                "job_id": jid, "mode": j["mode"], "field_frame": j["field_frame"],
                "image_frame": j["image_frame"], "params": j["params"],
                "shape": j["shape"], "nbytes": j["nbytes"],
            }
            for jid, j in state.jobs.items()
        ]


def _import_reproject():
    import reproject  # sibling module owned by another agent; may not exist yet

    return reproject


# ---- PNG cache and shared server state -----------------------------------------


class PNGCache:
    """Bounded-bytes LRU cache of rendered PNG bytes, keyed by the full
    resolved-parameter tuple for one route."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.data = {}
        self.order = []  # most-recently-used at the end
        self.nbytes = 0

    def get(self, key):
        with self.lock:
            if key not in self.data:
                return None
            self.order.remove(key)
            self.order.append(key)
            return self.data[key]

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                self.nbytes -= len(self.data[key])
                self.order.remove(key)
            self.data[key] = value
            self.order.append(key)
            self.nbytes += len(value)
            while self.nbytes > self.max_bytes and self.order:
                oldest = self.order.pop(0)
                self.nbytes -= len(self.data.pop(oldest))


class AppState:
    def __init__(self, bundle):
        self.bundle = bundle
        self.png_cache = PNGCache(PNG_CACHE_MAX_BYTES)
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.summary_cache = None
        self.summary_lock = threading.Lock()
        self.mask_frac_cache = {}
        self.mask_frac_lock = threading.Lock()
        # scatter_to_refspace peaks at ~13.4 GB of the A4000's 16.08 GB (measured),
        # so two concurrent reprojections exhaust the card. Serialise every GPU call.
        self.gpu_lock = threading.Lock()


# ---- HTTP layer -----------------------------------------------------------------


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler_cls, state):
        self.state = state
        super().__init__(addr, handler_cls)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _png(self, status, body, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, name):
        safe = os.path.normpath("/" + name).lstrip("/")
        path = os.path.join(STATIC_DIR, safe)
        real_static = os.path.realpath(STATIC_DIR)
        if os.path.realpath(path) != path and not os.path.realpath(path).startswith(real_static + os.sep):
            return self._json(400, {"error": "invalid static path %r" % name})
        if not os.path.isfile(path):
            return self._json(404, {"error": "static asset not found: %s" % name})
        ctype, _ = mimetypes.guess_type(path)
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_png(self, name, resolved, build_fn, state, extra_headers=None):
        key = (name,) + tuple(sorted(resolved.items()))
        cached = state.png_cache.get(key)
        if cached is None:
            rgb = build_fn()
            cached = render.to_png(rgb)
            state.png_cache.put(key, cached)
        self._png(200, cached, extra_headers)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        state = self.server.state
        bundle = state.bundle
        parsed = urllib.parse.urlsplit(self.path)
        path, query = parsed.path, parsed.query
        try:
            if method == "GET" and path == "/":
                return self._serve_static("index.html")
            if method == "GET" and path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if method == "GET" and path == "/img/alignment_qc":
                return self._serve_alignment_qc(bundle, query)
            if method == "GET" and path == "/api/manifest":
                return self._json(200, api_manifest(bundle))
            if method == "GET" and path == "/api/metrics":
                return self._json(200, api_metrics(state))
            if method == "GET" and path == "/api/summary":
                return self._json(200, api_summary(state))
            if method == "GET" and path == "/img/motion":
                r = _parse_motion(query, bundle)
                return self._serve_png("motion", r, lambda: _render_motion(r, bundle), state)
            if method == "GET" and path == "/img/plane":
                r = _parse_plane(query, bundle)
                return self._serve_png("plane", r, lambda: _render_plane(r, bundle), state)
            if method == "GET" and path == "/img/refspace":
                r = _parse_refspace(query, bundle, state)
                return self._serve_png(
                    "refspace", r, lambda: _render_refspace(r, bundle, state), state
                )
            if method == "GET" and path == "/img/coverage":
                r = _parse_coverage(query, bundle)
                return self._serve_png("coverage", r, lambda: _render_coverage(r, bundle), state)
            if method == "GET" and path == "/img/montage":
                r = _parse_montage(query, bundle)
                scale = _montage_tile_scale(r, bundle)
                return self._serve_png(
                    "montage", r, lambda: _render_montage(r, bundle), state,
                    extra_headers={"X-Tile-Scale": "%.6f" % scale},
                )
            if method == "POST" and path == "/api/reproject":
                return self._api_reproject(state)
            if method == "GET" and path == "/api/jobs":
                return self._json(200, api_jobs(state))
            if method == "DELETE" and path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                with state.jobs_lock:
                    if job_id not in state.jobs:
                        return self._json(404, {"error": "unknown job id %r" % job_id})
                    del state.jobs[job_id]
                return self._json(200, {"deleted": job_id})
            return self._json(404, {"error": "no route for %s %s" % (method, path)})
        except BadRequest as e:
            self._json(400, e.to_json())
        except Exception as e:
            logging.exception("unhandled error on %s %s", method, self.path)
            self._json(500, {"error": str(e)})

    def _api_reproject(self, state):
        bundle = state.bundle
        try:
            body = self._read_json_body()
        except (ValueError, UnicodeDecodeError):
            return self._json(400, {"error": "request body is not valid JSON"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "request body must be a JSON object"})
        try:
            reproject = _import_reproject()
        except ImportError as e:
            return self._json(503, {"error": "reproject module unavailable: %s" % e})

        for name in ("field_frame", "image_frame"):
            if name not in body:
                return self._json(400, {"error": "missing required field '%s'" % name})
        field_frame, image_frame = body["field_frame"], body["image_frame"]
        if field_frame not in bundle.frames:
            return self._json(
                400, {"error": "'field_frame'=%r not in available frames %s" % (field_frame, bundle.frames)}
            )
        if image_frame not in bundle.frames:
            return self._json(
                400, {"error": "'image_frame'=%r not in available frames %s" % (image_frame, bundle.frames)}
            )
        mode = body.get("mode", "planes")
        if mode not in ("planes", "refspace"):
            return self._json(400, {"error": "'mode'=%r must be 'planes' or 'refspace'" % mode})
        params = body.get("params", {})
        if not isinstance(params, dict):
            return self._json(400, {"error": "'params' must be an object"})

        t0 = time.perf_counter()
        with state.gpu_lock:
            waited_s = time.perf_counter() - t0
            if mode == "planes":
                arr = reproject.project_to_planes(bundle, field_frame, image_frame, params)
            else:
                arr = reproject.scatter_to_refspace(bundle, field_frame, image_frame, params)
        elapsed_s = time.perf_counter() - t0

        job_id = uuid.uuid4().hex
        with state.jobs_lock:
            state.jobs[job_id] = {
                "array": arr, "mode": mode, "field_frame": field_frame, "image_frame": image_frame,
                "params": params, "shape": list(arr.shape), "nbytes": int(arr.nbytes),
                "created": time.time(),
            }
        # waited_s separates queueing behind another GPU job from compute time, so a slow
        # response is not misread as a slow kernel.
        self._json(200, {"job_id": job_id, "shape": list(arr.shape),
                         "elapsed_s": elapsed_s, "waited_s": waited_s})

    # The two alignment-QC images are PNGs the pipeline already wrote; they are served
    # verbatim rather than re-rendered. Only these two names are reachable, so the run
    # directory is not exposed as a general file tree.
    ALIGNMENT_QC_FILES = {
        "zinit_match_curve": "zinit_match_curve.png",
        "zinit_zncc_heatmap": "zinit_zncc_heatmap.png",
    }

    def _serve_alignment_qc(self, bundle, query):
        p = dict(urllib.parse.parse_qsl(query))
        unknown = set(p) - {"name"}
        if unknown:
            return self._json(400, {"error": "unknown parameter(s): %s" % ", ".join(sorted(unknown))})
        name = p.get("name", "zinit_match_curve")
        fname = self.ALIGNMENT_QC_FILES.get(name)
        if fname is None:
            return self._json(400, {"error": "unknown 'name' %r; expected one of %s"
                                    % (name, ", ".join(sorted(self.ALIGNMENT_QC_FILES)))})
        path = os.path.join(str(bundle.run_dir), "diagnostics", "alignment_qc", fname)
        if not os.path.isfile(path):
            return self._json(404, {"error": "alignment-QC image not present in this run: %s" % fname})
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logging.info("%s - %s", self.address_string(), fmt % args)


# ---- CLI / selftest ---------------------------------------------------------------


def _check(base, desc, path, expect_status, expect_ctype=None, method="GET", data=None, failures=None):
    if not isinstance(expect_status, (tuple, list, set)):
        expect_status = (expect_status,)
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        status, ctype, body = resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as e:
        status, ctype, body = e.code, e.headers.get("Content-Type", ""), e.read()
    ok = status in expect_status and (expect_ctype is None or expect_ctype in ctype)
    if ok:
        print("PASS %s (status=%s ctype=%s bytes=%d)" % (desc, status, ctype, len(body)))
    else:
        msg = "%s: got status=%s ctype=%s, want status in %s ctype=%s, body[:200]=%r" % (
            desc, status, ctype, expect_status, expect_ctype, body[:200],
        )
        print("FAIL " + msg)
        if failures is not None:
            failures.append(msg)
    return status, body


def _selftest(run_dir, cache_dir):
    logging.basicConfig(level=logging.WARNING)
    bundle = qc_bundle.QCBundle(run_dir, cache_dir)
    state = AppState(bundle)
    httpd = DashboardServer((_HOST, 0), Handler, state)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://%s:%d" % (_HOST, port)
    failures = []
    frame, k = bundle.frames[0], 0

    try:
        _check(base, "manifest", "/api/manifest", 200, "application/json", failures=failures)
        _check(base, "metrics", "/api/metrics", 200, "application/json", failures=failures)
        _check(base, "summary", "/api/summary", 200, "application/json", failures=failures)
        _check(base, "motion norm", "/img/motion?frame=%d&k=%d&quantity=norm" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "motion dx", "/img/motion?frame=%d&k=%d&quantity=dx" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "motion masked", "/img/motion?frame=%d&k=%d&mask=1" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "motion quiver", "/img/motion?frame=%d&k=%d&quiver=1" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "motion bad frame -> 400", "/img/motion?frame=9999&k=%d&quantity=norm" % k, 400, "application/json", failures=failures)
        _check(base, "motion unknown param -> 400", "/img/motion?frame=%d&k=%d&bogus=1" % (frame, k), 400, "application/json", failures=failures)
        _check(base, "motion asymmetric diverging -> 400", "/img/motion?frame=%d&k=%d&quantity=dx&lo_pct=5&hi_pct=99" % (frame, k), 400, "application/json", failures=failures)
        _check(base, "plane mov", "/img/plane?kind=mov&frame=%d&k=%d" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "plane raw_moving", "/img/plane?kind=raw_moving&frame=%d&k=%d&channel=mem" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "plane projected", "/img/plane?kind=projected&frame=%d&k=%d&channel=sparseCell" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "refspace", "/img/refspace?frame=%d&z=0&channel=mem&source=precomputed&overlay=1" % frame, 200, "image/png", failures=failures)
        _check(base, "refspace bad job -> 400", "/img/refspace?z=0&channel=mem&source=nosuchjob", 400, "application/json", failures=failures)
        _check(base, "coverage", "/img/coverage?frame=%d&k=%d" % (frame, k), 200, "image/png", failures=failures)
        _check(base, "montage time", "/img/montage?axis=time&quantity=norm&k=%d" % k, 200, "image/png", failures=failures)
        _check(base, "montage z", "/img/montage?axis=z&quantity=norm&frame=%d" % frame, 200, "image/png", failures=failures)
        _check(base, "jobs list", "/api/jobs", 200, "application/json", failures=failures)
        _check(base, "jobs delete missing -> 404", "/api/jobs/nosuchjob", 404, "application/json", method="DELETE", failures=failures)
        payload = json.dumps({"field_frame": frame, "image_frame": frame, "mode": "planes", "params": {}}).encode()
        _check(base, "reproject (200 if reproject.py present, else 503)", "/api/reproject", (200, 503), "application/json", method="POST", data=payload, failures=failures)
        # static/index.html is a sibling agent's deliverable, not owned by server.py:
        # accept 200 (delivered) or 404 (not yet delivered) as both correct for THIS file.
        _check(base, "static index", "/", (200, 404), failures=failures)
        _check(base, "unknown route -> 404", "/nope", 404, "application/json", failures=failures)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)

    if failures:
        print("SELFTEST FAILURES: %d" % len(failures))
        return 1
    print("SELFTEST OK")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="QC dashboard server")
    parser.add_argument("--run-dir", default=_DEFAULT_RUN_DIR)
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest(args.run_dir, args.cache_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bundle = qc_bundle.QCBundle(args.run_dir, args.cache_dir)
    state = AppState(bundle)
    httpd = DashboardServer((_HOST, args.port), Handler, state)
    print("QC dashboard serving on http://%s:%d (run_dir=%s)" % (_HOST, args.port, args.run_dir))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
