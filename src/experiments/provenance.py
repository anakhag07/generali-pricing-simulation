"""Small, deterministic provenance helpers shared by experiments and reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, str | int]:
    """Return the canonical path and content digest for an input or output file."""
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def array_sha256(values: Any) -> str:
    """Hash an array using explicit dtype, shape, and contiguous bytes."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_provenance(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a stable, human-readable provenance JSON document."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = ["array_sha256", "file_record", "file_sha256", "write_provenance"]
