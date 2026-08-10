"""
Utility functions for saving and loading intermediate results in the
motion episode / motion mode / motion region / motion pattern pipeline.

This module is intentionally lightweight and does not depend on the
specific motion analysis module, except that objects must be pickleable.

Recommended directory layout:

cache_root/
    metadata.json
    01_patch_motion/
        arrays.npz
        metadata.json
    02_motion_units/
        objects.pkl
        metadata.json
    03_episodes/
        objects.pkl
        metadata.json
    04_modes/
        objects.pkl
        metadata.json
    05_regions/
        objects.pkl
        metadata.json
    06_patterns/
        objects.pkl
        metadata.json

Important:
    Pickled custom objects require that the class definitions are importable
    with the same module path when loading. If you move/rename the module
    defining MotionEpisode / MotionMode / MotionRegion / MotionPattern,
    old pickle files may fail to load.
"""

from __future__ import annotations

import os
import json
import time
import pickle
import hashlib
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# Basic helpers
# ============================================================

def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj):
    """JSON serializer for numpy / Path / simple non-JSON objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "summary": {
                "min": float(np.nanmin(obj)) if obj.size else None,
                "max": float(np.nanmax(obj)) if obj.size else None,
                "mean": float(np.nanmean(obj)) if obj.size else None,
            },
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def save_json(obj: Dict[str, Any], path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def md5_of_file(path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_pickle(obj: Any, path, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def load_pickle(path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def as_numpy(x):
    """
    Convert common array-like objects to numpy.
    Supports numpy and cupy arrays when cupy is installed.
    """
    if x is None:
        return None

    # cupy array support without hard dependency
    mod = type(x).__module__
    if mod.startswith("cupy"):
        return x.get()

    return np.asarray(x)


def save_npz(path, compressed: bool = False, **arrays) -> None:
    """
    Save arrays to npz. Set compressed=False for speed and large motion arrays.
    """
    path = Path(path)
    ensure_dir(path.parent)
    arrays_np = {k: as_numpy(v) for k, v in arrays.items() if v is not None}

    if compressed:
        np.savez_compressed(path, **arrays_np)
    else:
        np.savez(path, **arrays_np)


def load_npz(path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def stage_path(cache_root, stage_name: str) -> Path:
    return ensure_dir(Path(cache_root) / stage_name)


def write_cache_metadata(
    cache_root,
    project_name: str = "motion_pattern_validation",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    cache_root = ensure_dir(cache_root)
    metadata = {
        "project_name": project_name,
        "created_at": now_string(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    save_json(metadata, cache_root / "metadata.json")


# ============================================================
# Stage-specific save / load functions
# ============================================================

def save_patch_motion_stage(
    cache_root,
    motion_delta,
    motion_abs,
    mask_patched,
    params: Optional[Dict[str, Any]] = None,
    compressed: bool = False,
    stage_name: str = "01_patch_motion",
) -> Path:
    """
    Save patch-level motion arrays.

    Usually generated by:
        motion_delta, motion_abs, mask_patched = mcp.motions_obtain(..., return_abs=True)
    """
    out_dir = stage_path(cache_root, stage_name)
    save_npz(
        out_dir / "arrays.npz",
        compressed=compressed,
        motion_delta=motion_delta,
        motion_abs=motion_abs,
        mask_patched=mask_patched,
    )
    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "arrays": {
                "motion_delta_shape": list(np.asarray(motion_delta).shape),
                "motion_abs_shape": list(np.asarray(motion_abs).shape),
                "mask_patched_shape": list(np.asarray(mask_patched).shape),
            },
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_patch_motion_stage(cache_root, stage_name: str = "01_patch_motion") -> Dict[str, np.ndarray]:
    out_dir = Path(cache_root) / stage_name
    return load_npz(out_dir / "arrays.npz")


def save_motion_units_stage(
    cache_root,
    rest_motion,
    motion_units,
    active_mask,
    motion_units_filtered=None,
    params: Optional[Dict[str, Any]] = None,
    compressed_arrays: bool = False,
    stage_name: str = "02_motion_units",
) -> Path:
    """
    Save rest motion, active mask, and MotionUnit objects.
    """
    out_dir = stage_path(cache_root, stage_name)

    save_npz(
        out_dir / "arrays.npz",
        compressed=compressed_arrays,
        rest_motion=rest_motion,
        active_mask=active_mask,
    )

    save_pickle(
        {
            "motion_units": motion_units,
            "motion_units_filtered": motion_units_filtered,
        },
        out_dir / "objects.pkl",
    )

    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "arrays": {
                "rest_motion_shape": list(np.asarray(rest_motion).shape),
                "active_mask_shape": list(np.asarray(active_mask).shape),
            },
            "has_motion_units_filtered": motion_units_filtered is not None,
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_motion_units_stage(cache_root, stage_name: str = "02_motion_units") -> Dict[str, Any]:
    out_dir = Path(cache_root) / stage_name
    arrays = load_npz(out_dir / "arrays.npz")
    objects = load_pickle(out_dir / "objects.pkl")
    return {**arrays, **objects}


def save_episodes_stage(
    cache_root,
    episodes,
    params: Optional[Dict[str, Any]] = None,
    stage_name: str = "03_episodes",
) -> Path:
    """
    Save MotionEpisode objects.
    """
    out_dir = stage_path(cache_root, stage_name)
    save_pickle({"episodes": episodes}, out_dir / "objects.pkl")

    summary = summarize_episodes(episodes)
    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "summary": summary,
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_episodes_stage(cache_root, stage_name: str = "03_episodes"):
    out_dir = Path(cache_root) / stage_name
    return load_pickle(out_dir / "objects.pkl")["episodes"]


def save_modes_stage(
    cache_root,
    episodes,
    modes=None,
    params: Optional[Dict[str, Any]] = None,
    stage_name: str = "04_modes",
) -> Path:
    """
    Save episodes after MotionMode decomposition.
    Saving episodes is preferred because modes are attached to episodes.
    """
    if modes is None:
        modes = collect_attr_from_episodes(episodes, "modes")

    out_dir = stage_path(cache_root, stage_name)
    save_pickle(
        {
            "episodes": episodes,
            "modes": modes,
        },
        out_dir / "objects.pkl",
    )

    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "summary": {
                "n_episodes": len(episodes),
                "n_modes": len(modes),
                "modes_per_episode": [len(getattr(ep, "modes", []) or []) for ep in episodes],
            },
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_modes_stage(cache_root, stage_name: str = "04_modes") -> Dict[str, Any]:
    out_dir = Path(cache_root) / stage_name
    return load_pickle(out_dir / "objects.pkl")


def save_regions_stage(
    cache_root,
    episodes,
    regions=None,
    params: Optional[Dict[str, Any]] = None,
    stage_name: str = "05_regions",
) -> Path:
    """
    Save episodes after MotionRegion extraction.
    Saving episodes is preferred because regions are attached to episodes.
    """
    if regions is None:
        regions = collect_attr_from_episodes(episodes, "regions")

    out_dir = stage_path(cache_root, stage_name)
    save_pickle(
        {
            "episodes": episodes,
            "regions": regions,
        },
        out_dir / "objects.pkl",
    )

    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "summary": {
                "n_episodes": len(episodes),
                "n_regions": len(regions),
                "regions_per_episode": [len(getattr(ep, "regions", []) or []) for ep in episodes],
            },
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_regions_stage(cache_root, stage_name: str = "05_regions") -> Dict[str, Any]:
    out_dir = Path(cache_root) / stage_name
    return load_pickle(out_dir / "objects.pkl")


def save_patterns_stage(
    cache_root,
    patterns,
    kept_regions=None,
    groups=None,
    labels=None,
    info=None,
    params: Optional[Dict[str, Any]] = None,
    stage_name: str = "06_patterns",
) -> Path:
    """
    Save MotionPattern clustering outputs.
    """
    out_dir = stage_path(cache_root, stage_name)

    # Save the (large) distance matrix in its own npz and keep it OUT of the
    # pickle — pop from a copy before the pickle is written
    info_saved = dict(info) if isinstance(info, dict) else info
    if isinstance(info_saved, dict) and "distance_matrix" in info_saved:
        dist = info_saved.pop("distance_matrix")
        save_npz(out_dir / "distance_matrix.npz", compressed=False, distance_matrix=dist)

    obj = {
        "patterns": patterns,
        "kept_regions": kept_regions,
        "groups": groups,
        "labels": labels,
        "info": info_saved,
    }
    save_pickle(obj, out_dir / "objects.pkl")

    save_json(
        {
            "stage": stage_name,
            "saved_at": now_string(),
            "params": params or {},
            "summary": {
                "n_patterns": len(patterns) if patterns is not None else 0,
                "n_kept_regions": len(kept_regions) if kept_regions is not None else None,
                "pattern_sizes": [len(getattr(p, "regions", []) or []) for p in patterns] if patterns is not None else [],
            },
            "info_keys": list(info.keys()) if isinstance(info, dict) else None,
        },
        out_dir / "metadata.json",
    )

    return out_dir


def load_patterns_stage(cache_root, stage_name: str = "06_patterns") -> Dict[str, Any]:
    out_dir = Path(cache_root) / stage_name
    obj = load_pickle(out_dir / "objects.pkl")

    dist_path = out_dir / "distance_matrix.npz"
    if dist_path.exists():
        dist = load_npz(dist_path)["distance_matrix"]
        if obj.get("info") is not None and isinstance(obj["info"], dict):
            obj["info"]["distance_matrix"] = dist

    return obj


# ============================================================
# Summaries and collection helpers
# ============================================================

def collect_attr_from_episodes(episodes, attr_name: str) -> list:
    out = []
    for ep in episodes:
        vals = getattr(ep, attr_name, None)
        if vals is None:
            continue
        out.extend(list(vals))
    return out


def summarize_episodes(episodes) -> Dict[str, Any]:
    out = {
        "n_episodes": len(episodes),
        "time_ranges": [],
        "areas": [],
        "has_motion_delta": 0,
        "has_motion_abs": 0,
        "has_global_motion": 0,
    }

    for ep in episodes:
        tr = getattr(ep, "time_range", None)
        if tr is not None:
            out["time_ranges"].append(list(tr))

        mask = getattr(ep, "spatial_region", None)
        if mask is not None:
            out["areas"].append(int(np.asarray(mask).astype(bool).sum()))

        if getattr(ep, "motion_delta", None) is not None:
            out["has_motion_delta"] += 1
        if getattr(ep, "motion_abs", None) is not None:
            out["has_motion_abs"] += 1
        if getattr(ep, "global_motion", None) is not None:
            out["has_global_motion"] += 1

    if out["areas"]:
        out["area_min"] = int(np.min(out["areas"]))
        out["area_median"] = float(np.median(out["areas"]))
        out["area_max"] = int(np.max(out["areas"]))

    durations = []
    for tr in out["time_ranges"]:
        if len(tr) >= 2:
            durations.append(int(tr[1] - tr[0] + 1))
    if durations:
        out["duration_min"] = int(np.min(durations))
        out["duration_median"] = float(np.median(durations))
        out["duration_max"] = int(np.max(durations))

    return out


def print_stage_metadata(cache_root) -> None:
    """
    Print metadata.json files under cache_root.
    """
    cache_root = Path(cache_root)
    for meta_path in sorted(cache_root.glob("*/metadata.json")):
        print("\n" + "=" * 80)
        print(meta_path)
        try:
            meta = load_json(meta_path)
            print(json.dumps(meta, indent=2, ensure_ascii=False, default=json_default))
        except Exception as e:
            print(f"Failed to read: {e}")


# ============================================================
# Convenience: save/load entire checkpoint
# ============================================================

def save_checkpoint(
    cache_root,
    name: str,
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save arbitrary checkpoint payload as pickle.

    Example:
        save_checkpoint(cache_root, "debug_after_regions", {
            "episodes": episodes,
            "regions": regions,
        })
    """
    out_dir = stage_path(cache_root, f"checkpoint_{name}")
    save_pickle(payload, out_dir / "objects.pkl")
    save_json(
        {
            "stage": f"checkpoint_{name}",
            "saved_at": now_string(),
            "metadata": metadata or {},
            "keys": list(payload.keys()),
        },
        out_dir / "metadata.json",
    )
    return out_dir


def load_checkpoint(cache_root, name: str) -> Dict[str, Any]:
    out_dir = Path(cache_root) / f"checkpoint_{name}"
    return load_pickle(out_dir / "objects.pkl")



