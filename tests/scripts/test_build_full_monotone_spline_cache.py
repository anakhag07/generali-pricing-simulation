"""Collector and resume tests for the full-customer cache builder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data.full_monotone_spline_cache import shard_directory, write_cache_shard
from experiments.launch import LaunchContext
from scripts import build_full_monotone_spline_cache as builder


def _context(tmp_path: Path) -> LaunchContext:
    sweep_dir = tmp_path / "cache" / "sweeps" / "run"
    return LaunchContext(
        plan_name=builder.PROJECT_NAME,
        runs_root=tmp_path / "cache",
        sweep_id="run",
        sweep_dir=sweep_dir,
        tasks_dir=sweep_dir / "tasks",
        launch_mode="local",
        array=True,
    )


def _write_shard(
    context: LaunchContext,
    eligible: np.ndarray,
    *,
    shard_index: int,
    start: int,
    stop: int,
) -> dict[str, object]:
    grid = np.linspace(0.0, 0.16, 5)
    values = np.stack(
        [0.1 + row_position * 0.01 + grid for row_position in range(start, stop)]
    )
    derivatives = np.ones_like(values)
    metadata = write_cache_shard(
        context.sweep_dir,
        shard_index=shard_index,
        start_position=start,
        stop_position=stop,
        row_indices=eligible[start:stop],
        customer_ids=[f"id-{row}" for row in eligible[start:stop]],
        action_grid=grid,
        churn_values=values,
        churn_derivatives=derivatives,
        upper_slopes=np.ones(stop - start),
        fit_seconds=0.5,
        started_at="2026-08-12T00:00:00+00:00",
        finished_at="2026-08-12T00:00:01+00:00",
    )
    return {
        "kind": "curve_cache_shard",
        "shard_index": shard_index,
        "start_position": start,
        "stop_position": stop,
        "shard_path": str(shard_directory(context.sweep_dir, shard_index)),
        "row_count": stop - start,
        "failure_count": 0,
        "fit_seconds": metadata["fit_seconds"],
        "started_at": metadata["started_at"],
        "finished_at": metadata["finished_at"],
    }


def _records(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"task_index": index, "status": "success", "payload": payload}
        for index, payload in enumerate(payloads)
    ]


def test_exact_coverage_accepts_one_ordered_shard_per_task(tmp_path: Path) -> None:
    context = _context(tmp_path)
    eligible = np.asarray([2, 5, 9, 12, 18])
    payloads = [
        _write_shard(context, eligible, shard_index=0, start=0, stop=2),
        _write_shard(context, eligible, shard_index=1, start=2, stop=4),
        _write_shard(context, eligible, shard_index=2, start=4, stop=5),
    ]

    validated = builder.validate_task_coverage(
        _records(payloads), eligible=eligible, chunk_size=2, cache_dir=context.sweep_dir
    )

    assert [shard["row_count"] for shard in validated] == [2, 2, 1]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "overlap", "out_of_range"])
def test_missing_duplicate_overlapping_or_out_of_range_shards_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    context = _context(tmp_path)
    eligible = np.asarray([2, 5, 9, 12])
    payloads = [
        _write_shard(context, eligible, shard_index=0, start=0, stop=2),
        _write_shard(context, eligible, shard_index=1, start=2, stop=4),
    ]
    records = _records(payloads)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        duplicate = dict(records[1])
        duplicate["task_index"] = 0
        records.append(duplicate)
    elif mutation == "overlap":
        records[1]["payload"] = {**payloads[1], "start_position": 1}
    else:
        records[1]["payload"] = {**payloads[1], "stop_position": 5}

    with pytest.raises(ValueError):
        builder.validate_task_coverage(
            records, eligible=eligible, chunk_size=2, cache_dir=context.sweep_dir
        )


def test_existing_valid_shard_resumes_without_loading_models(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    eligible = np.asarray([2, 5])
    payload = _write_shard(context, eligible, shard_index=0, start=0, stop=2)
    monkeypatch.setattr(
        builder,
        "load_x_frame",
        lambda *args, **kwargs: pytest.fail("resume should not load source data"),
    )
    args = SimpleNamespace(chunk_size=2, n_jobs=1, storage_dtype="float32")

    resumed = builder._run_task(
        0,
        context,
        args=args,
        eligible=eligible,
        weights=np.ones(17),
    )

    assert resumed["resumed"] is True
    assert resumed["shard_path"] == payload["shard_path"]


def test_resume_rejects_existing_shard_with_wrong_rows(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_shard(context, np.asarray([2, 6]), shard_index=0, start=0, stop=2)
    args = SimpleNamespace(chunk_size=2, n_jobs=1, storage_dtype="float32")

    with pytest.raises(ValueError, match="expected source rows"):
        builder._run_task(
            0,
            context,
            args=args,
            eligible=np.asarray([2, 5]),
            weights=np.ones(17),
        )


def test_parse_args_defaults_to_ten_thousand_rows_and_parallel_eight() -> None:
    args = builder._parse_args(["--launch", "local"])
    assert args.chunk_size == 10_000
    assert args.array is True
    assert args.array_max_parallel == 8


def test_main_builds_72_shard_plan_for_expected_customer_count(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(builder.acceptance_analysis, "_eligible_rows", lambda: np.arange(715_023))
    monkeypatch.setattr(builder.acceptance_analysis, "_spline_weights", lambda rows: np.ones(17))
    monkeypatch.setattr(
        builder,
        "run_launch_plan",
        lambda plan, **kwargs: captured.update(plan=plan, kwargs=kwargs),
    )

    builder.main(["--launch", "local"])

    assert captured["plan"].task_count == 72
    assert captured["plan"].default_array is True
    assert captured["kwargs"]["args"].array_max_parallel == 8
