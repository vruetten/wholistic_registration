"""Data access for the QC dashboard: run-directory discovery, the z-major cache,
axis normalisation, mask/coverage unpacking, CSV loading.

Every accessor returns image planes as (Y, X) = (1500, 630) and vector fields as
(Y, X, 3) with the last axis ordered (x, y, z), per the canonical orientation rule
in dashboard/SPEC.md. No other module may transpose.

Imports restricted per SPEC.md #1 to numpy, tifffile, json, csv, os for the
QCBundle class itself. sys and time are added only for the --selftest CLI entry
point below QCBundle; QCBundle's own methods do not use them (see Spec gaps in
the implementer report).
"""
import csv
import json
import os
import sys
import threading
import time

import numpy as np
import tifffile

_CHANNELS = ("mem", "sparseCell")


class QCBundle:
    def __init__(self, run_dir, cache_dir):
        self.run_dir = run_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._build_lock = threading.Lock()
        self._frame_locks = {}
        self._diagnostics_dir = os.path.join(run_dir, "diagnostics")

        self.frames = self._discover_frames()

        self.ref_shape_zyx = tuple(
            int(v) for v in np.load(os.path.join(self._diagnostics_dir, "ref_shape.npy"))
        )
        mov_probe = np.load(
            os.path.join(self._diagnostics_dir, f"mov_mem_f{self.frames[0]}.npy"), mmap_mode="r"
        )
        self.mov_shape_zyx = tuple(int(v) for v in mov_probe.shape)
        # Every plane accessor shares one (Y, X); ref and mov grids must agree on it.
        assert self.ref_shape_zyx[1:] == self.mov_shape_zyx[1:]
        self._plane_shape = self.mov_shape_zyx[1:]

        with open(os.path.join(self._diagnostics_dir, "projection_params.json")) as f:
            self.projection_params = json.load(f)
        self.fixed_target_z = np.load(os.path.join(self._diagnostics_dir, "fixed_target_z.npy"))

        self._zmajor_cache = {}  # (kind, frame) -> memmap (Z, X, Y, 3) float32

    # ---- frame discovery -------------------------------------------------

    def _discover_frames(self):
        prefix, suffix = "phase_new_f", ".npy"
        found = []
        for name in os.listdir(self._diagnostics_dir):
            if name.startswith(prefix) and name.endswith(suffix):
                found.append(int(name[len(prefix) : -len(suffix)]))
        found.sort()
        assert found, f"no phase_new_f*.npy under {self._diagnostics_dir}"
        return found

    # ---- z-major cache for motion_current / phase_new ---------------------

    def _source_path(self, kind, frame):
        assert kind in ("motion_current", "phase_new")
        return os.path.join(self._diagnostics_dir, f"{kind}_f{frame}.npy")

    def _cache_path(self, kind, frame):
        src = self._source_path(kind, frame)
        st = os.stat(src)
        # mtime_ns + size in the filename: a changed source produces a different
        # path, so a stale cache is never read under an old name.
        key = f"{kind}_f{frame}_zmajor_m{st.st_mtime_ns}_s{st.st_size}.npy"
        return os.path.join(self.cache_dir, key)

    def _zmajor(self, kind, frame):
        cache_key = (kind, frame)
        if cache_key in self._zmajor_cache:
            return self._zmajor_cache[cache_key]
        # One lock per (kind, frame), so two threads asking for the same frame do not both
        # read the 216 MB source, while different frames still build concurrently. The
        # earlier version shared one fixed .tmp path across threads: the first os.replace
        # consumed it and the second raised FileNotFoundError. Observed in the server log
        # when a montage request fanned out over all frames.
        with self._build_lock:
            lock = self._frame_locks.setdefault(cache_key, threading.Lock())
        with lock:
            if cache_key in self._zmajor_cache:
                return self._zmajor_cache[cache_key]
            cache_path = self._cache_path(kind, frame)
            if not os.path.exists(cache_path):
                src = self._source_path(kind, frame)
                arr = np.load(src)  # native (x, y, z, c) = (630, 1500, 20, 3)
                zmajor = np.ascontiguousarray(np.transpose(arr, (2, 0, 1, 3)))  # (z, x, y, c)
                # Unique per process and thread: np.save appends .npy, so name the temp file
                # so that the appended suffix lands on the path we later rename.
                tmp_path = "%s.tmp%d_%d" % (cache_path, os.getpid(), threading.get_ident())
                np.save(tmp_path, zmajor)
                os.replace(tmp_path + ".npy", cache_path)  # atomic: no reader sees a partial file
            arr = np.load(cache_path, mmap_mode="r")
            self._zmajor_cache[cache_key] = arr
            return arr

    def _vector_plane(self, kind, frame, k):
        zmajor = self._zmajor(kind, frame)
        plane_xyc = zmajor[k]  # (x, y, c) = (630, 1500, 3)
        return np.ascontiguousarray(np.transpose(plane_xyc, (1, 0, 2))).astype(
            np.float32, copy=False
        )

    def motion_plane(self, frame, k):
        return self._vector_plane("motion_current", frame, k)

    def phase_plane(self, frame, k):
        return self._vector_plane("phase_new", frame, k)

    # ---- moving-grid scalar volume ----------------------------------------

    def mov_plane(self, frame, k):
        path = os.path.join(self._diagnostics_dir, f"mov_mem_f{frame}.npy")
        arr = np.load(path, mmap_mode="r")  # native (z, y, x); no cache needed
        return np.array(arr[k], dtype=np.float32)

    # ---- sparse coordinate masks --------------------------------------------

    @staticmethod
    def _sparse_plane_yx(npz, k, plane_shape):
        # Keys 'x', 'y', 'z' are explicit physical coordinate arrays; axis_order
        # only orders the accompanying 'shape' field, which this accessor does
        # not consume, so no branch on axis_order is needed here (read below
        # regardless, to honour "read the field where it exists").
        _ = str(npz["axis_order"])
        assert not bool(npz["dense"])  # sparse-coordinate-list format only
        z = npz["z"]
        sel = z == k
        y = npz["y"][sel]
        x = npz["x"][sel]
        out = np.zeros(plane_shape, dtype=bool)
        out[y, x] = True
        return out

    def mask_mov_plane(self, frame, k):
        path = os.path.join(self._diagnostics_dir, "masks_mov", f"mask_mov_{frame:06d}.npz")
        with np.load(path) as npz:
            return self._sparse_plane_yx(npz, k, self._plane_shape)

    def mask_ref_plane(self, z):
        path = os.path.join(self._diagnostics_dir, "mask_ref.npz")
        with np.load(path) as npz:
            return self._sparse_plane_yx(npz, z, self._plane_shape)

    # ---- bit-packed coverage -------------------------------------------------

    def coverage_plane(self, frame, k):
        path = os.path.join(self._diagnostics_dir, "coverage", f"no_coverage_{frame:06d}.npz")
        with np.load(path) as npz:
            packed = npz["packed"]
            shape = tuple(int(v) for v in npz["shape"])
            axis_order = str(npz["axis_order"])
        dense = np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape)
        z_axis = axis_order.index("z")
        plane = np.take(dense, k, axis=z_axis)
        remaining = axis_order.replace("z", "")
        if remaining == "xy":
            plane = plane.T
        else:
            assert remaining == "yx"
        return plane.astype(bool)

    # ---- TIFF volumes: raw / projected / refspace --------------------------

    @staticmethod
    def _read_tif_page(path, page):
        with tifffile.TiffFile(path) as tf:
            return np.array(tf.pages[page].asarray(), dtype=np.float32)

    def _tif_path(self, dir_prefix, file_token, channel, frame):
        assert channel in _CHANNELS
        dirname = f"{dir_prefix}_{channel}"
        # refspace_sparseCell/ holds files named ..._refspace_sparse_...; the
        # other two families use 'sparseCell' in both directory and filename.
        name_channel = "sparse" if (dir_prefix == "refspace" and channel == "sparseCell") else channel
        fname = f"vol_F260517_{file_token}_{name_channel}_{frame:06d}.tif"
        return os.path.join(self.run_dir, dirname, fname)

    def raw_moving_plane(self, frame, k, channel):
        path = self._tif_path("raw_moving", "raw", channel, frame)
        return self._read_tif_page(path, k)

    def projected_plane(self, frame, k, channel):
        path = self._tif_path("projected", "projected", channel, frame)
        return self._read_tif_page(path, k)

    def refspace_plane(self, frame, z, channel):
        path = self._tif_path("refspace", "refspace", channel, frame)
        return self._read_tif_page(path, z)

    def refspace_reference_plane(self, z):
        path = os.path.join(self.run_dir, "refspace_reference_mem.tif")
        return self._read_tif_page(path, z)

    # ---- CSV metrics and manifest -------------------------------------------

    @staticmethod
    def _parse_cell(s):
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s

    def metrics(self):
        out = {}
        for name in sorted(os.listdir(self._diagnostics_dir)):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(self._diagnostics_dir, name)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            columns, data_rows = rows[0], rows[1:]
            out[name[: -len(".csv")]] = {
                "columns": columns,
                "rows": [[self._parse_cell(c) for c in row] for row in data_rows],
            }
        return out

    def manifest(self):
        return {
            "run_dir": self.run_dir,
            "cache_dir": self.cache_dir,
            "frames": list(self.frames),
            "ref_shape_zyx": list(self.ref_shape_zyx),
            "mov_shape_zyx": list(self.mov_shape_zyx),
            "plane_shape_yx": list(self._plane_shape),
            "channels": list(_CHANNELS),
            "projection_params": self.projection_params,
            "fixed_target_z": self.fixed_target_z.tolist(),
        }


def _z_init(k):
    # base = (x_index, y_index, z_init[k]); verified identity, SPEC.md #"Measured
    # constants". Not derived from fixed_target_z.npy, which is a distinct,
    # nearby but not equal, per-frame optimisation target.
    return 20 + 10 * k


def _selftest(run_dir, cache_dir):
    bundle = QCBundle(run_dir, cache_dir)
    print("=== manifest ===")
    print(json.dumps(bundle.manifest(), indent=2))

    frames = bundle.frames[:2]
    mov_planes = [0, 5]
    ref_planes = [0, 110]

    def report(label, a):
        finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
        nan_count = int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0
        lo = finite.min() if finite.size else float("nan")
        hi = finite.max() if finite.size else float("nan")
        print(f"{label}: shape={a.shape} dtype={a.dtype} min={lo} max={hi} nan={nan_count}")

    print("=== accessors ===")
    for frame in frames:
        for k in mov_planes:
            report(f"motion_plane(f{frame},k{k})", bundle.motion_plane(frame, k))
            report(f"phase_plane(f{frame},k{k})", bundle.phase_plane(frame, k))
            report(f"mov_plane(f{frame},k{k})", bundle.mov_plane(frame, k))
            report(f"mask_mov_plane(f{frame},k{k})", bundle.mask_mov_plane(frame, k))
            report(f"coverage_plane(f{frame},k{k})", bundle.coverage_plane(frame, k))
            for channel in _CHANNELS:
                report(
                    f"raw_moving_plane(f{frame},k{k},{channel})",
                    bundle.raw_moving_plane(frame, k, channel),
                )
                report(
                    f"projected_plane(f{frame},k{k},{channel})",
                    bundle.projected_plane(frame, k, channel),
                )
        for z in ref_planes:
            report(f"mask_ref_plane(z{z})", bundle.mask_ref_plane(z))
            for channel in _CHANNELS:
                report(
                    f"refspace_plane(f{frame},z{z},{channel})",
                    bundle.refspace_plane(frame, z, channel),
                )
        report(f"refspace_reference_plane(z{ref_planes[0]})", bundle.refspace_reference_plane(ref_planes[0]))

    metrics = bundle.metrics()
    print("=== metrics ===")
    for name, table in metrics.items():
        print(f"{name}: columns={table['columns']} n_rows={len(table['rows'])}")

    print("=== timing: cold vs warm motion_plane ===")
    # A frame untouched by the accessor loop above (which only used
    # bundle.frames[:2]), so this motion_plane call is a genuine first access.
    timing_frame = bundle.frames[-1]
    assert timing_frame not in frames
    t0 = time.perf_counter()
    bundle.motion_plane(timing_frame, 3)
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    bundle.motion_plane(timing_frame, 7)
    warm_s = time.perf_counter() - t0
    print(f"cold (includes cache build) = {cold_s:.3f} s")
    print(f"warm (cache already built)  = {warm_s:.3f} s")

    print("=== identity: phase_plane == base + motion_plane ===")
    id_frame = frames[0]
    y_idx = np.arange(0, bundle._plane_shape[0], 17)
    x_idx = np.arange(0, bundle._plane_shape[1], 7)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    for k in range(bundle.mov_shape_zyx[0]):
        motion = bundle.motion_plane(id_frame, k)[yy, xx]
        phase = bundle.phase_plane(id_frame, k)[yy, xx]
        pred_x = xx.astype(np.float32) + motion[..., 0]
        pred_y = yy.astype(np.float32) + motion[..., 1]
        pred_z = _z_init(k) + motion[..., 2]
        resid_x = np.max(np.abs(pred_x - phase[..., 0]))
        resid_y = np.max(np.abs(pred_y - phase[..., 1]))
        resid_z = np.max(np.abs(pred_z - phase[..., 2]))
        print(f"k={k}: max resid x={resid_x:.6g} y={resid_y:.6g} z={resid_z:.6g}")
        assert resid_x < 1e-2 and resid_y < 1e-2 and resid_z < 1e-2

    print("=== coverage_plane mean vs hole_frac ===")
    hole_rows = {row[0]: row for row in metrics["errors_membrane"]["rows"]}
    hole_cols = metrics["errors_membrane"]["columns"]
    hole_frac_idx = hole_cols.index("hole_frac")
    for frame in bundle.frames:
        n_z = bundle.mov_shape_zyx[0]
        computed = float(
            np.mean([bundle.coverage_plane(frame, k).mean() for k in range(n_z)])
        )
        expected = float(hole_rows[frame][hole_frac_idx])
        print(f"frame={frame}: computed={computed:.6f} expected={expected:.6f}")
        assert round(computed, 6) == round(expected, 6)

    print("=== selftest OK ===")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _RUN_DIR = (
            "/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/"
            "registration_out/f260517_0625_qc_v4"
        )
        _CACHE_DIR = "/tmp/qc_bundle_cache_selftest"
        _selftest(_RUN_DIR, _CACHE_DIR)
