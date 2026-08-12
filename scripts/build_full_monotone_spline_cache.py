"""Build and validate a resumable full-customer monotone XGBoost spline cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import ACCEPTANCE_MODEL_ARTIFACTS, DATASET_PATH
from data.full_monotone_spline_cache import (
    FULL_CACHE_FORMAT,
    FULL_CACHE_SCHEMA_VERSION,
    REPRESENTATION,
    ShardedMonotoneSplineCache,
    sha256_file,
    shard_directory,
    validate_cache_shard,
    write_cache_shard,
)
from data.loader import load_model_artifacts, load_x_frame
from data.monotone_spline_xgb import FittedMonotoneChurnCurve, fit_monotone_churn_curve
from experiments.launch import (
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
)
from experiments.paths import results_root
from scripts import analyze_model_acceptance_features as acceptance_analysis


PROJECT_NAME = "monotone-spline-xgb-full-v1"
DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_ARRAY_MAX_PARALLEL = 8
DEFAULT_VALUE_ATOL = 2e-6
DEFAULT_DERIVATIVE_ATOL = 2e-4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--storage-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--validation-sample-size", type=int, default=128)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--value-atol", type=float, default=DEFAULT_VALUE_ATOL)
    parser.add_argument("--derivative-atol", type=float, default=DEFAULT_DERIVATIVE_ATOL)
    add_launch_args(parser, default_launch="auto", default_array=True)
    args = parser.parse_args(argv)
    for name in ("chunk_size", "n_jobs", "validation_sample_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in ("value_atol", "derivative_atol"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if args.array_max_parallel is None:
        args.array_max_parallel = DEFAULT_ARRAY_MAX_PARALLEL
    return args


def _fit_one(anchor_acceptance: np.ndarray, weights: np.ndarray) -> FittedMonotoneChurnCurve:
    return fit_monotone_churn_curve(
        acceptance_analysis.ANCHOR_U,
        1.0 - anchor_acceptance,
        weights=weights,
        dense_grid_size=500,
    )


def _shard_payload(path: Path, metadata: Mapping[str, Any], *, resumed: bool) -> dict[str, Any]:
    return {
        "kind": "curve_cache_shard",
        "shard_index": int(metadata["shard_index"]),
        "start_position": int(metadata["start_position"]),
        "stop_position": int(metadata["stop_position"]),
        "row_count": int(metadata["row_count"]),
        "failure_count": int(metadata["failure_count"]),
        "fit_seconds": float(metadata["fit_seconds"]),
        "started_at": str(metadata["started_at"]),
        "finished_at": str(metadata["finished_at"]),
        "shard_path": str(path),
        "resumed": bool(resumed),
    }


def _run_task(
    task_index: int,
    context: LaunchContext,
    *,
    args: argparse.Namespace,
    eligible: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    start = task_index * int(args.chunk_size)
    stop = min(start + int(args.chunk_size), eligible.size)
    expected_rows = eligible[start:stop]
    output = shard_directory(context.sweep_dir, task_index)
    if output.exists():
        metadata = validate_cache_shard(output, expected_rows=expected_rows)
        print(f"[resume] shard {task_index} already complete", flush=True)
        return _shard_payload(output, metadata, resumed=True)

    started_at = _utc_now()
    started = time.perf_counter()
    frame = load_x_frame("monotone_spline_xgb", row_indices=expected_rows)
    xgb_acceptance, _ = load_model_artifacts("xgb")
    anchors = acceptance_analysis._predict_acceptance_matrix(
        xgb_acceptance, frame, acceptance_analysis.ANCHOR_U
    )

    def fit_safely(row_position: int) -> tuple[FittedMonotoneChurnCurve | None, str | None]:
        try:
            return _fit_one(anchors[row_position], weights), None
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    fitted = Parallel(n_jobs=int(args.n_jobs), prefer="threads")(
        delayed(fit_safely)(row_position) for row_position in range(expected_rows.size)
    )
    failure_positions = np.asarray(
        [index for index, (curve, _) in enumerate(fitted) if curve is None], dtype=int
    )
    if failure_positions.size:
        failure_rows = expected_rows[failure_positions]
        details = [fitted[index][1] for index in failure_positions[:10]]
        raise RuntimeError(
            f"Shard {task_index} had {failure_rows.size} spline-fit failures; "
            f"source rows={failure_rows[:10].tolist()}, errors={details}"
        )

    curves = [curve for curve, _ in fitted if curve is not None]
    action_grid = curves[0].action_grid
    values = np.stack([curve.curve(action_grid) for curve in curves])
    derivatives = np.stack([curve.curve.derivative()(action_grid) for curve in curves])
    upper_slopes = np.asarray([curve.upper_slope for curve in curves])
    fit_seconds = time.perf_counter() - started
    metadata = write_cache_shard(
        context.sweep_dir,
        shard_index=task_index,
        start_position=start,
        stop_position=stop,
        row_indices=expected_rows,
        customer_ids=frame["id"].astype("string").to_numpy(dtype=str),
        action_grid=action_grid,
        churn_values=values,
        churn_derivatives=derivatives,
        upper_slopes=upper_slopes,
        storage_dtype=args.storage_dtype,
        fit_seconds=fit_seconds,
        started_at=started_at,
        finished_at=_utc_now(),
    )
    print(
        f"[cache] shard={task_index} rows={expected_rows.size:,} "
        f"fit_seconds={fit_seconds:.1f} path={output}",
        flush=True,
    )
    return _shard_payload(output, metadata, resumed=False)


def validate_task_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    eligible: np.ndarray,
    chunk_size: int,
    cache_dir: str | Path,
) -> list[dict[str, Any]]:
    """Reject absent, failed, duplicate, overlapping, or misplaced shard records."""
    expected_count = (eligible.size + int(chunk_size) - 1) // int(chunk_size)
    by_task: dict[int, Mapping[str, Any]] = {}
    payloads: list[Mapping[str, Any]] = []
    for record in records:
        task_index = int(record.get("task_index", -1))
        if task_index in by_task:
            raise ValueError(f"Duplicate task record for index {task_index}.")
        by_task[task_index] = record
        if record.get("status") != "success":
            raise ValueError(
                f"Task {task_index} did not succeed: {record.get('error', 'unknown error')}"
            )
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Task {task_index} has no shard payload.")
        payloads.append(payload)
    expected_tasks = set(range(expected_count))
    actual_tasks = set(by_task)
    if actual_tasks != expected_tasks:
        missing = sorted(expected_tasks - actual_tasks)
        extra = sorted(actual_tasks - expected_tasks)
        raise ValueError(f"Task records are incomplete or out of range; missing={missing}, extra={extra}.")

    shard_indices = [int(payload.get("shard_index", -1)) for payload in payloads]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError("Duplicate shard indices were reported by successful tasks.")
    if set(shard_indices) != expected_tasks:
        raise ValueError("Shard indices are missing or out of range.")

    root = Path(cache_dir).resolve()
    validated: list[dict[str, Any]] = []
    covered: list[np.ndarray] = []
    previous_stop = 0
    for payload in sorted(payloads, key=lambda item: int(item["start_position"])):
        shard_index = int(payload["shard_index"])
        start = int(payload["start_position"])
        stop = int(payload["stop_position"])
        expected_start = shard_index * int(chunk_size)
        expected_stop = min(expected_start + int(chunk_size), eligible.size)
        if start != expected_start or stop != expected_stop:
            raise ValueError(f"Shard {shard_index} reports an out-of-range row span.")
        if start != previous_stop:
            raise ValueError("Shard row-position spans are missing or overlapping.")
        path = Path(str(payload["shard_path"])).resolve()
        if path != shard_directory(root, shard_index).resolve():
            raise ValueError(f"Shard {shard_index} points outside its expected cache path.")
        metadata = validate_cache_shard(path, expected_rows=eligible[start:stop])
        rows = np.load(path / "row_indices.npy", allow_pickle=False, mmap_mode="r")
        covered.append(np.asarray(rows))
        validated.append({**metadata, "path": str(path.relative_to(root))})
        previous_stop = stop
    concatenated = np.concatenate(covered)
    if concatenated.size != eligible.size:
        raise ValueError("Collected shards do not contain exactly the eligible row count.")
    if np.unique(concatenated).size != concatenated.size:
        raise ValueError("Collected shards contain duplicate source rows.")
    if not np.array_equal(concatenated, eligible):
        raise ValueError("Collected shards are missing, reordered, or contain out-of-range rows.")
    return validated


def _canonical_curve_values(
    curve: FittedMonotoneChurnCurve, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    below = actions < curve.action_grid[0]
    above = actions > curve.action_grid[-1]
    inside = ~(below | above)
    churn = np.empty(actions.size, dtype=float)
    derivative = np.empty(actions.size, dtype=float)
    churn[below] = curve.churn_min
    derivative[below] = 0.0
    raw_upper = curve.churn_max + curve.upper_slope * (
        actions[above] - curve.action_grid[-1]
    )
    churn[above] = np.clip(raw_upper, 0.0, 1.0)
    derivative[above] = np.where(
        (raw_upper > 0.0) & (raw_upper < 1.0), curve.upper_slope, 0.0
    )
    raw_inside = curve.curve(actions[inside])
    churn[inside] = np.clip(raw_inside, 0.0, 1.0)
    derivative[inside] = np.where(
        (raw_inside > 0.0) & (raw_inside < 1.0),
        curve.curve.derivative()(actions[inside]),
        0.0,
    )
    return 1.0 - churn, -derivative


def validate_canonical_parity(
    cache: ShardedMonotoneSplineCache,
    *,
    eligible: np.ndarray,
    weights: np.ndarray,
    sample_size: int,
    seed: int,
    value_atol: float,
    derivative_atol: float,
) -> dict[str, Any]:
    """Compare serialized Hermite curves to fresh canonical helper fits."""
    count = min(int(sample_size), eligible.size)
    rng = np.random.default_rng(int(seed))
    sample_rows = np.sort(rng.choice(eligible, size=count, replace=False))
    frame = load_x_frame("xgb", row_indices=sample_rows)
    xgb_acceptance, _ = load_model_artifacts("xgb")
    anchors = acceptance_analysis._predict_acceptance_matrix(
        xgb_acceptance, frame, acceptance_analysis.ANCHOR_U
    )
    action_grid = np.linspace(-0.02, 0.24, 521)
    cached_values = cache.acceptance(sample_rows, action_grid)
    cached_derivatives = cache.derivative(sample_rows, action_grid)
    value_error = 0.0
    derivative_error = 0.0
    for index, anchor_row in enumerate(anchors):
        curve = _fit_one(anchor_row, weights)
        canonical_values, canonical_derivatives = _canonical_curve_values(curve, action_grid)
        value_error = max(
            value_error,
            float(np.max(np.abs(cached_values[index] - canonical_values))),
        )
        derivative_error = max(
            derivative_error,
            float(np.max(np.abs(cached_derivatives[index] - canonical_derivatives))),
        )
    if value_error > float(value_atol):
        raise ValueError(
            f"Canonical value parity failed: max error {value_error:.6g} > {value_atol:.6g}."
        )
    if derivative_error > float(derivative_atol):
        raise ValueError(
            "Canonical derivative parity failed: "
            f"max error {derivative_error:.6g} > {derivative_atol:.6g}."
        )
    if np.any(np.diff(cached_values, axis=1) > float(value_atol)):
        raise ValueError("Validation sample contains non-monotone acceptance curves.")
    if np.any(cached_values < -float(value_atol)) or np.any(
        cached_values > 1.0 + float(value_atol)
    ):
        raise ValueError("Validation sample contains out-of-bounds acceptance values.")
    return {
        "sample_size": count,
        "sample_seed": int(seed),
        "evaluation_grid": {"min": -0.02, "max": 0.24, "count": 521},
        "max_acceptance_abs_error": value_error,
        "max_derivative_abs_error": derivative_error,
        "acceptance_atol": float(value_atol),
        "derivative_atol": float(derivative_atol),
        "monotonicity_and_bounds": "passed",
        "tail_semantics": "passed_below_constant_above_clipped_tangent",
    }


def _git_provenance() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    try:
        return {
            "commit": output("git", "rev-parse", "HEAD"),
            "branch": output("git", "branch", "--show-current"),
            "dirty": bool(output("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _write_eligible_index(cache_dir: Path, eligible: np.ndarray) -> dict[str, Any]:
    path = cache_dir / "eligible_row_indices.npy"
    if path.exists():
        loaded = np.load(path, allow_pickle=False, mmap_mode="r")
        if not np.array_equal(loaded, eligible):
            raise ValueError("Existing eligible-row index does not match this collection.")
    else:
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, np.asarray(eligible, dtype=np.int64), allow_pickle=False)
        temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "dtype": "int64",
        "shape": [int(eligible.size)],
    }


def _manifest(
    *,
    context: LaunchContext,
    args: argparse.Namespace,
    eligible: np.ndarray,
    weights: np.ndarray,
    shards: Sequence[Mapping[str, Any]],
    eligible_index: Mapping[str, Any],
) -> dict[str, Any]:
    started = [datetime.fromisoformat(str(shard["started_at"])) for shard in shards]
    finished = [datetime.fromisoformat(str(shard["finished_at"])) for shard in shards]
    payload_bytes = int(eligible_index["bytes"]) + sum(
        int(file_info["bytes"])
        for shard in shards
        for file_info in shard["files"].values()
    )
    model_path = Path(ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"])
    bytes_per_float32 = eligible.size * 500 * 2 * np.dtype("float32").itemsize
    bytes_per_float64 = eligible.size * 500 * 2 * np.dtype("float64").itemsize
    return {
        "schema_version": FULL_CACHE_SCHEMA_VERSION,
        "format": FULL_CACHE_FORMAT,
        "representation": REPRESENTATION,
        "cache_version": "v1",
        "created_at": _utc_now(),
        "cache_path": str(context.sweep_dir),
        "row_count": int(eligible.size),
        "row_index_min": int(eligible.min()),
        "row_index_max": int(eligible.max()),
        "row_order": "ascending canonical dataset CSV row position",
        "row_identity": ["dataset_row_index", "customer_id"],
        "eligible_row_indices": dict(eligible_index),
        "failure_count": 0,
        "failure_rows": [],
        "source": {
            "dataset_path": str(DATASET_PATH),
            "dataset_sha256": sha256_file(DATASET_PATH),
            "xgb_acceptance_path": str(model_path),
            "xgb_acceptance_sha256": sha256_file(model_path),
            "git": _git_provenance(),
        },
        "config": {
            "chunk_size": int(args.chunk_size),
            "shard_count": len(shards),
            "array_max_parallel": int(args.array_max_parallel),
            "n_jobs_per_task": int(args.n_jobs),
            "storage_dtype": str(args.storage_dtype),
            "anchor_u": acceptance_analysis.ANCHOR_U.tolist(),
            "dense_grid_size": 500,
            "tail_semantics": "constant below support; clipped nonnegative tangent above support",
        },
        "weighting": {
            "provenance": (
                "all eligible customers' historical U values, restricted to [0, 0.16], "
                "rounded to 0.01, normalized frequencies reindexed to the 17 anchors; "
                "zero weights become 1e-9 inside fit_monotone_churn_curve"
            ),
            "weights_before_zero_floor": weights.tolist(),
        },
        "recipe": {
            "canonical_helper": "data.monotone_spline_xgb.fit_monotone_churn_curve",
            "all_customer_path": "scripts.analyze_model_acceptance_features",
            "steps": [
                "17 raw-XGB acceptance anchor predictions",
                "weighted scipy make_smoothing_spline",
                "clip to [0, 1] on 500 points",
                "increasing isotonic regression",
                "PCHIP interpolation",
            ],
        },
        "dtype_assessment": {
            "chosen": str(args.storage_dtype),
            "float32_curve_payload_bytes": int(bytes_per_float32),
            "float64_curve_payload_bytes": int(bytes_per_float64),
            "rationale": (
                "float32 stores both PCHIP knot values and analytical knot derivatives, "
                "halving the curve payload versus float64 while avoiding derivative "
                "reconstruction from quantized values; canonical parity is enforced below"
            ),
        },
        "timing": {
            "sum_shard_fit_seconds": float(sum(float(shard["fit_seconds"]) for shard in shards)),
            "first_task_started_at": min(started).isoformat(),
            "last_task_finished_at": max(finished).isoformat(),
            "fitting_wall_span_seconds": float((max(finished) - min(started)).total_seconds()),
        },
        "payload_disk_bytes": payload_bytes,
        "shards": [dict(shard) for shard in shards],
        "validation": {"status": "pending"},
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _record_exact_disk_size(cache_dir: Path, manifest: dict[str, Any]) -> None:
    """Record total cache bytes, including the final manifest itself."""
    manifest_path = cache_dir / "manifest.json"
    other_bytes = sum(
        path.stat().st_size
        for path in cache_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    for _ in range(10):
        encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        total = other_bytes + len(encoded)
        if manifest.get("disk_size_bytes") == total:
            break
        manifest["disk_size_bytes"] = total


def _collect(
    context: LaunchContext,
    *,
    args: argparse.Namespace,
    eligible: np.ndarray,
    weights: np.ndarray,
) -> None:
    collection_started = time.perf_counter()
    records = read_task_records(context)
    shards = validate_task_coverage(
        records,
        eligible=eligible,
        chunk_size=args.chunk_size,
        cache_dir=context.sweep_dir,
    )
    eligible_index = _write_eligible_index(context.sweep_dir, eligible)
    manifest = _manifest(
        context=context,
        args=args,
        eligible=eligible,
        weights=weights,
        shards=shards,
        eligible_index=eligible_index,
    )
    manifest_path = context.sweep_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    cache = ShardedMonotoneSplineCache(context.sweep_dir, verify_checksums=False)
    validation = validate_canonical_parity(
        cache,
        eligible=eligible,
        weights=weights,
        sample_size=args.validation_sample_size,
        seed=args.validation_seed,
        value_atol=args.value_atol,
        derivative_atol=args.derivative_atol,
    )
    validation["status"] = "passed"
    validation["all_shard_checksums"] = "passed"
    validation["exact_coverage"] = "passed"
    validation["collection_seconds"] = time.perf_counter() - collection_started
    manifest["validation"] = validation
    _record_exact_disk_size(context.sweep_dir, manifest)
    _write_manifest(manifest_path, manifest)
    print(
        f"Collected {eligible.size:,} rows in {len(shards)} shards; failures=0; "
        f"payload={int(manifest['payload_disk_bytes']) / 2**30:.3f} GiB; "
        f"max value error={validation['max_acceptance_abs_error']:.3g}; "
        f"max derivative error={validation['max_derivative_abs_error']:.3g}; "
        f"cache={context.sweep_dir}",
        flush=True,
    )


def _build_launch_plan(
    args: argparse.Namespace,
    eligible: np.ndarray,
    weights: np.ndarray,
) -> LaunchPlan:
    task_count = (eligible.size + int(args.chunk_size) - 1) // int(args.chunk_size)
    return LaunchPlan(
        name=PROJECT_NAME,
        task_count=task_count,
        requires_jax=False,
        run_task=lambda index, context: _run_task(
            index, context, args=args, eligible=eligible, weights=weights
        ),
        collect=lambda context: _collect(
            context, args=args, eligible=eligible, weights=weights
        ),
        runs_root=str(results_root() / "cache"),
        default_launch="auto",
        default_array=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    eligible = acceptance_analysis._eligible_rows()
    weights = acceptance_analysis._spline_weights(eligible)
    task_count = (eligible.size + int(args.chunk_size) - 1) // int(args.chunk_size)
    print(
        f"Prepared {task_count} cache shards for {eligible.size:,} eligible rows; "
        f"chunk_size={args.chunk_size:,}, dtype={args.storage_dtype}, "
        f"array_max_parallel={args.array_max_parallel}.",
        flush=True,
    )
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(
        _build_launch_plan(args, eligible, weights),
        args=args,
        argv=original_argv,
    )


if __name__ == "__main__":
    main()
