"""Crash-safe state and numeric checkpoints for registration resume."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registration_fingerprint(config_path, input_path, code_paths=()):
    """Return a stable signature for the config, input file metadata, and code."""
    config_path = Path(config_path).resolve()
    input_path = Path(input_path).resolve()
    input_stat = input_path.stat()
    code = []
    for path in sorted((Path(p).resolve() for p in code_paths), key=str):
        code.append({"path": str(path), "sha256": _sha256_file(path)})
    record = {
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "input_path": str(input_path),
        "input_size": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "code": code,
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_json_state(path, state):
    """Atomically replace a small JSON run-state manifest."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _atomic_write(path, lambda stream: stream.write(payload))


def load_json_state(path):
    with open(path, "r", encoding="utf-8") as stream:
        state = json.load(stream)
    if not isinstance(state, dict):
        raise ValueError(f"Registration state must contain a JSON object: {path}")
    return state


def save_checkpoint(path, metadata, **arrays):
    """Atomically write JSON metadata plus pickle-free numeric NumPy arrays."""
    if "metadata" in arrays:
        raise ValueError("'metadata' is a reserved checkpoint array name")
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise TypeError(f"Checkpoint array {name!r} has object dtype")
    values = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))}
    values.update(arrays)
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **values))


def load_checkpoint(path):
    """Load checkpoint metadata and arrays without enabling pickle."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: np.array(archive[name]) for name in archive.files if name != "metadata"}
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must contain a JSON object: {path}")
    return metadata, arrays


def checkpoint_elapsed_seconds(path):
    """Read elapsed time from a checkpoint, returning zero if none is readable."""
    if not os.path.isfile(path):
        return 0.0
    try:
        metadata, _ = load_checkpoint(path)
        value = float(metadata.get("elapsed_seconds", 0.0))
        return value if math.isfinite(value) and value >= 0 else 0.0
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return 0.0


def max_logged_elapsed(log_path):
    """Return the largest valid elapsed value in an append-only JSONL log."""
    latest = 0.0
    try:
        with open(log_path, "r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = float(json.loads(line).get("elapsed_seconds", 0.0))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if math.isfinite(value):
                    latest = max(latest, value)
    except FileNotFoundError:
        pass
    return latest
