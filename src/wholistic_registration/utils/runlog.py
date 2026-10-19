"""Small append-only JSONL progress log for long registration runs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from typing import Any


def log_progress(log_path, started_monotonic, phase, **fields):
    """Append one timestamped progress record from the parent or a worker process."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "phase": str(phase),
        "pid": os.getpid(),
    }
    record.update(fields)
    payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"Short write to registration log {path}: {written}/{len(payload)} bytes")
    finally:
        os.close(fd)
    return record
