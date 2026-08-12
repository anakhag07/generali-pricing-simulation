"""Sharded, lazy evaluation for the full-customer monotone spline cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


FULL_CACHE_SCHEMA_VERSION = 1
FULL_CACHE_FORMAT = "monotone_spline_xgb_full_customer_cache"
REPRESENTATION = "dense_grid_cubic_hermite"
DEFAULT_STORAGE_DTYPE = np.dtype("float32")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_directory(cache_dir: str | Path, shard_index: int) -> Path:
    """Return the stable directory for one numbered shard."""
    return Path(cache_dir) / "shards" / f"shard_{int(shard_index):05d}"


def _array_file_metadata(path: Path) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def write_cache_shard(
    cache_dir: str | Path,
    *,
    shard_index: int,
    start_position: int,
    stop_position: int,
    row_indices: Sequence[int],
    customer_ids: Sequence[str],
    action_grid: Sequence[float],
    churn_values: np.ndarray,
    churn_derivatives: np.ndarray,
    upper_slopes: Sequence[float],
    storage_dtype: str | np.dtype = DEFAULT_STORAGE_DTYPE,
    fit_seconds: float,
    started_at: str,
    finished_at: str,
    failure_rows: Sequence[int] = (),
) -> dict[str, Any]:
    """Atomically write one independently loadable cache shard."""
    output = shard_directory(cache_dir, shard_index)
    if output.exists():
        return validate_cache_shard(output)

    dtype = np.dtype(storage_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("storage_dtype must be float32 or float64.")
    rows = np.asarray(row_indices, dtype=np.int64)
    ids = np.asarray(customer_ids, dtype=str)
    grid = np.asarray(action_grid, dtype=np.float64)
    values = np.asarray(churn_values, dtype=dtype)
    derivatives = np.asarray(churn_derivatives, dtype=dtype)
    slopes = np.asarray(upper_slopes, dtype=dtype)
    failures = np.asarray(failure_rows, dtype=np.int64)
    n_rows = int(stop_position) - int(start_position)
    if rows.shape != (n_rows,) or ids.shape != (n_rows,):
        raise ValueError("row_indices and customer_ids must match the shard range.")
    if values.shape != (n_rows, grid.size) or derivatives.shape != values.shape:
        raise ValueError("curve arrays must have shape (shard rows, action-grid size).")
    if slopes.shape != (n_rows,):
        raise ValueError("upper_slopes must contain one value per shard row.")
    if failures.size:
        raise ValueError("A complete cache shard cannot contain failed rows.")
    if not np.isfinite(values).all() or not np.isfinite(derivatives).all():
        raise ValueError("curve arrays must be finite.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    arrays = {
        "row_indices.npy": rows,
        "customer_ids.npy": ids,
        "action_grid.npy": grid,
        "churn_values.npy": values,
        "churn_derivatives.npy": derivatives,
        "upper_slopes.npy": slopes,
    }
    try:
        for name, array in arrays.items():
            np.save(temporary / name, array, allow_pickle=False)
        files = {
            name: _array_file_metadata(temporary / name)
            for name in sorted(arrays)
        }
        metadata: dict[str, Any] = {
            "schema_version": FULL_CACHE_SCHEMA_VERSION,
            "format": FULL_CACHE_FORMAT,
            "representation": REPRESENTATION,
            "shard_index": int(shard_index),
            "start_position": int(start_position),
            "stop_position": int(stop_position),
            "row_count": n_rows,
            "row_index_min": int(rows.min()),
            "row_index_max": int(rows.max()),
            "storage_dtype": str(dtype),
            "failure_count": 0,
            "failure_rows": [],
            "fit_seconds": float(fit_seconds),
            "started_at": str(started_at),
            "finished_at": str(finished_at),
            "files": files,
        }
        (temporary / "shard.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output)
    except BaseException:
        # Leave the explicitly named temporary directory for forensic recovery.
        raise
    return validate_cache_shard(output)


def validate_cache_shard(
    path: str | Path,
    *,
    expected_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate one shard's schema, checksums, identities, and curve invariants."""
    root = Path(path)
    metadata_path = root / "shard.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing shard metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != FULL_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported full-cache shard schema in {root}.")
    if metadata.get("format") != FULL_CACHE_FORMAT:
        raise ValueError(f"Unexpected full-cache shard format in {root}.")
    if metadata.get("representation") != REPRESENTATION:
        raise ValueError(f"Unexpected curve representation in {root}.")
    required = {
        "row_indices.npy",
        "customer_ids.npy",
        "action_grid.npy",
        "churn_values.npy",
        "churn_derivatives.npy",
        "upper_slopes.npy",
    }
    files = metadata.get("files")
    if not isinstance(files, dict) or set(files) != required:
        raise ValueError(f"Shard {root} has an incomplete file inventory.")
    for name in sorted(required):
        file_path = root / name
        if not file_path.is_file():
            raise ValueError(f"Shard {root} is missing {name}.")
        if sha256_file(file_path) != str(files[name].get("sha256", "")):
            raise ValueError(f"Checksum mismatch for {file_path}.")

    rows = np.load(root / "row_indices.npy", allow_pickle=False, mmap_mode="r")
    ids = np.load(root / "customer_ids.npy", allow_pickle=False, mmap_mode="r")
    grid = np.load(root / "action_grid.npy", allow_pickle=False, mmap_mode="r")
    values = np.load(root / "churn_values.npy", allow_pickle=False, mmap_mode="r")
    derivatives = np.load(
        root / "churn_derivatives.npy", allow_pickle=False, mmap_mode="r"
    )
    slopes = np.load(root / "upper_slopes.npy", allow_pickle=False, mmap_mode="r")
    n_rows = int(metadata["row_count"])
    if rows.shape != (n_rows,) or ids.shape != (n_rows,):
        raise ValueError(f"Shard {root} identity arrays do not match row_count.")
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"Shard {root} has an invalid action grid.")
    if values.shape != (n_rows, grid.size) or derivatives.shape != values.shape:
        raise ValueError(f"Shard {root} has invalid curve-array shapes.")
    if slopes.shape != (n_rows,):
        raise ValueError(f"Shard {root} has invalid upper-slope shape.")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(derivatives)):
        raise ValueError(f"Shard {root} contains non-finite curves.")
    tolerance = 2e-6 if values.dtype == np.float32 else 1e-12
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError(f"Shard {root} contains out-of-bounds churn values.")
    if np.any(np.diff(values, axis=1) < -tolerance):
        raise ValueError(f"Shard {root} contains non-monotone churn values.")
    if np.any(derivatives < -tolerance):
        raise ValueError(f"Shard {root} contains negative churn derivatives.")
    if np.any(slopes < -tolerance):
        raise ValueError(f"Shard {root} contains negative upper-tail slopes.")
    if int(metadata.get("failure_count", -1)) != 0:
        raise ValueError(f"Shard {root} contains failed fits.")
    if expected_rows is not None and not np.array_equal(
        rows, np.asarray(expected_rows, dtype=np.int64)
    ):
        raise ValueError(f"Shard {root} does not contain its expected source rows.")
    interior = (
        grid[:-1, None]
        + np.diff(grid)[:, None] * np.asarray([0.25, 0.5, 0.75])
    ).ravel()
    for start in range(0, n_rows, 512):
        stop = min(start + 512, n_rows)
        evaluated, evaluated_derivative = _hermite_inside_grid(
            np.asarray(grid),
            np.asarray(values[start:stop], dtype=np.float64),
            np.asarray(derivatives[start:stop], dtype=np.float64),
            interior,
            clip_output=False,
        )
        if np.any(evaluated < -tolerance) or np.any(evaluated > 1.0 + tolerance):
            raise ValueError(f"Shard {root} leaves probability bounds within an interval.")
        if np.any(evaluated_derivative < -tolerance):
            raise ValueError(f"Shard {root} is non-monotone within an interval.")
    return metadata


@dataclass(frozen=True)
class _LocatedRows:
    requested_rows: np.ndarray
    positions: np.ndarray
    shard_indices: np.ndarray


class ShardedMonotoneSplineCache:
    """Lazy vectorized value/derivative evaluator for a collected full cache."""

    def __init__(self, cache_dir: str | Path, *, verify_checksums: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing full-cache manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if int(self.manifest.get("schema_version", -1)) != FULL_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported full monotone-spline cache schema version.")
        if self.manifest.get("format") != FULL_CACHE_FORMAT:
            raise ValueError("Manifest has an unexpected full-cache format.")
        if self.manifest.get("representation") != REPRESENTATION:
            raise ValueError("Manifest has an unexpected curve representation.")
        self.shards = sorted(
            self.manifest.get("shards", []), key=lambda item: int(item["start_position"])
        )
        if not self.shards:
            raise ValueError("Full-cache manifest contains no shards.")
        self._stops = np.asarray([item["stop_position"] for item in self.shards], dtype=int)
        expected_starts = np.r_[0, self._stops[:-1]]
        actual_starts = np.asarray([item["start_position"] for item in self.shards], dtype=int)
        if not np.array_equal(expected_starts, actual_starts):
            raise ValueError("Manifest shards are missing or overlapping positions.")
        index_info = self.manifest.get("eligible_row_indices", {})
        index_path = self.cache_dir / str(index_info.get("path", "eligible_row_indices.npy"))
        if verify_checksums and sha256_file(index_path) != index_info.get("sha256"):
            raise ValueError("Eligible-row index checksum does not match the manifest.")
        self.eligible_row_indices = np.load(
            index_path, allow_pickle=False, mmap_mode="r"
        )
        if self.eligible_row_indices.shape != (int(self.manifest["row_count"]),):
            raise ValueError("Eligible-row index length does not match the manifest.")
        if np.any(np.diff(self.eligible_row_indices) <= 0):
            raise ValueError("Eligible source row indices must be unique and ordered.")
        if verify_checksums:
            for shard in self.shards:
                validate_cache_shard(self.cache_dir / str(shard["path"]))

    def _locate(self, row_indices: Sequence[int]) -> _LocatedRows:
        requested = np.asarray(row_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("row_indices must be a 1D sequence.")
        positions = np.searchsorted(self.eligible_row_indices, requested)
        valid = positions < self.eligible_row_indices.size
        valid &= self.eligible_row_indices[np.minimum(positions, self.eligible_row_indices.size - 1)] == requested
        if not np.all(valid):
            missing = requested[~valid]
            raise KeyError(f"Source rows are absent from the cache: {missing[:10].tolist()}")
        shard_indices = np.searchsorted(self._stops, positions, side="right")
        return _LocatedRows(requested, positions.astype(int), shard_indices.astype(int))

    def customer_ids(self, row_indices: Sequence[int]) -> np.ndarray:
        """Return customer IDs aligned to arbitrary source-row selections."""
        located = self._locate(row_indices)
        output = np.empty(located.requested_rows.size, dtype=object)
        for shard_index in np.unique(located.shard_indices):
            selected = located.shard_indices == shard_index
            shard = self.shards[int(shard_index)]
            local = located.positions[selected] - int(shard["start_position"])
            ids = np.load(
                self.cache_dir / str(shard["path"]) / "customer_ids.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
            output[selected] = ids[local]
        return output.astype(str)

    def evaluate(
        self,
        row_indices: Sequence[int],
        u: float | Sequence[float] | np.ndarray,
        *,
        derivative: bool = False,
        pairwise: bool = False,
    ) -> np.ndarray:
        """Evaluate acceptance (or d-acceptance/du) without loading other shards.

        By default, a vector ``u`` is a shared grid and the result has shape
        ``(n_rows, n_u)``. A scalar returns ``(n_rows,)``. With ``pairwise=True``,
        ``u`` must contain one action per selected row and the result is 1D.
        """
        located = self._locate(row_indices)
        actions = np.asarray(u, dtype=np.float64)
        scalar = actions.ndim == 0
        if scalar:
            actions = actions.reshape(1)
        if actions.ndim != 1 or not np.isfinite(actions).all():
            raise ValueError("u must be a finite scalar or 1D array.")
        if pairwise and actions.shape != (located.requested_rows.size,):
            raise ValueError("pairwise u must contain one action per selected row.")
        output_shape = (
            (located.requested_rows.size,)
            if pairwise
            else (located.requested_rows.size, actions.size)
        )
        output = np.empty(output_shape, dtype=np.float64)
        for shard_index in np.unique(located.shard_indices):
            selected = located.shard_indices == shard_index
            shard = self.shards[int(shard_index)]
            root = self.cache_dir / str(shard["path"])
            local = located.positions[selected] - int(shard["start_position"])
            grid = np.load(root / "action_grid.npy", allow_pickle=False, mmap_mode="r")
            values = np.asarray(
                np.load(root / "churn_values.npy", allow_pickle=False, mmap_mode="r")[local],
                dtype=np.float64,
            )
            slopes = np.asarray(
                np.load(root / "upper_slopes.npy", allow_pickle=False, mmap_mode="r")[local],
                dtype=np.float64,
            )
            node_derivatives = np.asarray(
                np.load(
                    root / "churn_derivatives.npy", allow_pickle=False, mmap_mode="r"
                )[local],
                dtype=np.float64,
            )
            shard_actions = actions[selected] if pairwise else actions
            churn, d_churn = _evaluate_hermite(
                np.asarray(grid, dtype=np.float64),
                values,
                node_derivatives,
                slopes,
                shard_actions,
                pairwise=pairwise,
            )
            output[selected] = -d_churn if derivative else 1.0 - churn
        if scalar and not pairwise:
            return output[:, 0]
        return output

    def acceptance(
        self, row_indices: Sequence[int], u: float | Sequence[float] | np.ndarray, *, pairwise: bool = False
    ) -> np.ndarray:
        """Evaluate cached acceptance probabilities."""
        return self.evaluate(row_indices, u, pairwise=pairwise)

    def derivative(
        self, row_indices: Sequence[int], u: float | Sequence[float] | np.ndarray, *, pairwise: bool = False
    ) -> np.ndarray:
        """Evaluate analytical cached acceptance derivatives."""
        return self.evaluate(row_indices, u, derivative=True, pairwise=pairwise)


def _evaluate_hermite(
    grid: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
    upper_slopes: np.ndarray,
    actions: np.ndarray,
    *,
    pairwise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate shared-grid cubic Hermite curves with canonical tail semantics."""
    n_rows = values.shape[0]
    if pairwise:
        result = np.empty(n_rows, dtype=np.float64)
        d_result = np.empty(n_rows, dtype=np.float64)
        below = actions < grid[0]
        above = actions > grid[-1]
        inside = ~(below | above)
        result[below] = values[below, 0]
        d_result[below] = 0.0
        raw_upper = values[above, -1] + upper_slopes[above] * (actions[above] - grid[-1])
        result[above] = np.clip(raw_upper, 0.0, 1.0)
        d_result[above] = np.where(
            (raw_upper > 0.0) & (raw_upper < 1.0), upper_slopes[above], 0.0
        )
        if np.any(inside):
            row = np.flatnonzero(inside)
            result[inside], d_result[inside] = _hermite_inside_pairwise(
                grid, values[row], derivatives[row], actions[inside]
            )
        return result, d_result

    result = np.empty((n_rows, actions.size), dtype=np.float64)
    d_result = np.empty_like(result)
    below = actions < grid[0]
    above = actions > grid[-1]
    inside = ~(below | above)
    result[:, below] = values[:, [0]]
    d_result[:, below] = 0.0
    raw_upper = values[:, [-1]] + upper_slopes[:, None] * (actions[above] - grid[-1])
    result[:, above] = np.clip(raw_upper, 0.0, 1.0)
    d_result[:, above] = np.where(
        (raw_upper > 0.0) & (raw_upper < 1.0), upper_slopes[:, None], 0.0
    )
    if np.any(inside):
        result[:, inside], d_result[:, inside] = _hermite_inside_grid(
            grid, values, derivatives, actions[inside]
        )
    return result, d_result


def _hermite_inside_grid(
    grid: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
    actions: np.ndarray,
    *,
    clip_output: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    interval = np.minimum(np.searchsorted(grid, actions, side="right") - 1, grid.size - 2)
    h = grid[interval + 1] - grid[interval]
    t = (actions - grid[interval]) / h
    y0 = values[:, interval]
    y1 = values[:, interval + 1]
    d0 = derivatives[:, interval]
    d1 = derivatives[:, interval + 1]
    t2 = t * t
    t3 = t2 * t
    value = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * d1
    )
    derivative = (
        (6.0 * t2 - 6.0 * t) / h * y0
        + (3.0 * t2 - 4.0 * t + 1.0) * d0
        + (-6.0 * t2 + 6.0 * t) / h * y1
        + (3.0 * t2 - 2.0 * t) * d1
    )
    if not clip_output:
        return value, derivative
    bounded = np.clip(value, 0.0, 1.0)
    return bounded, np.where((value > 0.0) & (value < 1.0), derivative, 0.0)


def _hermite_inside_pairwise(
    grid: np.ndarray, values: np.ndarray, derivatives: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    interval = np.minimum(np.searchsorted(grid, actions, side="right") - 1, grid.size - 2)
    row = np.arange(actions.size)
    h = grid[interval + 1] - grid[interval]
    t = (actions - grid[interval]) / h
    y0 = values[row, interval]
    y1 = values[row, interval + 1]
    d0 = derivatives[row, interval]
    d1 = derivatives[row, interval + 1]
    t2 = t * t
    t3 = t2 * t
    value = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * d1
    )
    derivative = (
        (6.0 * t2 - 6.0 * t) / h * y0
        + (3.0 * t2 - 4.0 * t + 1.0) * d0
        + (-6.0 * t2 + 6.0 * t) / h * y1
        + (3.0 * t2 - 2.0 * t) * d1
    )
    bounded = np.clip(value, 0.0, 1.0)
    return bounded, np.where((value > 0.0) & (value < 1.0), derivative, 0.0)


def load_full_monotone_spline_cache(
    cache_dir: str | Path, *, verify_checksums: bool = False
) -> ShardedMonotoneSplineCache:
    """Load a collected full-customer cache."""
    return ShardedMonotoneSplineCache(cache_dir, verify_checksums=verify_checksums)


__all__ = [
    "DEFAULT_STORAGE_DTYPE",
    "FULL_CACHE_FORMAT",
    "FULL_CACHE_SCHEMA_VERSION",
    "REPRESENTATION",
    "ShardedMonotoneSplineCache",
    "load_full_monotone_spline_cache",
    "sha256_file",
    "shard_directory",
    "validate_cache_shard",
    "write_cache_shard",
]
