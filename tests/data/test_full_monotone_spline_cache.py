"""Tests for the sharded full-customer monotone spline representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.full_monotone_spline_cache import (
    FULL_CACHE_FORMAT,
    FULL_CACHE_SCHEMA_VERSION,
    REPRESENTATION,
    ShardedMonotoneSplineCache,
    sha256_file,
    validate_cache_shard,
    write_cache_shard,
)
from data.monotone_spline_xgb import FittedMonotoneChurnCurve, fit_monotone_churn_curve


def _curves() -> list[FittedMonotoneChurnCurve]:
    anchors = np.linspace(0.0, 0.16, 17)
    weights = np.linspace(0.5, 1.5, anchors.size)
    return [
        fit_monotone_churn_curve(
            anchors,
            np.clip(base + slope * anchors + 0.008 * np.sin(60.0 * anchors), 0.0, 1.0),
            weights=weights,
        )
        for base, slope in ((0.08, 0.8), (0.12, 1.0), (0.18, 0.5), (0.03, 1.4))
    ]


def _write_test_cache(tmp_path: Path) -> tuple[Path, np.ndarray, list[FittedMonotoneChurnCurve]]:
    cache_dir = tmp_path / "cache"
    rows = np.asarray([2, 5, 9, 14], dtype=np.int64)
    curves = _curves()
    shard_metadata = []
    for shard_index, (start, stop) in enumerate(((0, 2), (2, 4))):
        selected = curves[start:stop]
        grid = selected[0].action_grid
        metadata = write_cache_shard(
            cache_dir,
            shard_index=shard_index,
            start_position=start,
            stop_position=stop,
            row_indices=rows[start:stop],
            customer_ids=[f"customer-{row}" for row in rows[start:stop]],
            action_grid=grid,
            churn_values=np.stack([curve.curve(grid) for curve in selected]),
            churn_derivatives=np.stack([curve.curve.derivative()(grid) for curve in selected]),
            upper_slopes=[curve.upper_slope for curve in selected],
            storage_dtype="float32",
            fit_seconds=1.0,
            started_at="2026-08-12T00:00:00+00:00",
            finished_at="2026-08-12T00:00:01+00:00",
        )
        shard_metadata.append({**metadata, "path": f"shards/shard_{shard_index:05d}"})
    np.save(cache_dir / "eligible_row_indices.npy", rows, allow_pickle=False)
    index_path = cache_dir / "eligible_row_indices.npy"
    manifest = {
        "schema_version": FULL_CACHE_SCHEMA_VERSION,
        "format": FULL_CACHE_FORMAT,
        "representation": REPRESENTATION,
        "row_count": rows.size,
        "eligible_row_indices": {
            "path": index_path.name,
            "sha256": sha256_file(index_path),
        },
        "shards": shard_metadata,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache_dir, rows, curves


def _canonical(
    curve: FittedMonotoneChurnCurve, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    below = actions < curve.action_grid[0]
    above = actions > curve.action_grid[-1]
    inside = ~(below | above)
    churn = np.empty(actions.size)
    derivative = np.empty(actions.size)
    churn[below] = curve.churn_min
    derivative[below] = 0.0
    raw_upper = curve.churn_max + curve.upper_slope * (actions[above] - curve.action_grid[-1])
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


def test_round_trip_parity_monotonicity_bounds_derivatives_and_tails(tmp_path: Path) -> None:
    cache_dir, rows, curves = _write_test_cache(tmp_path)
    cache = ShardedMonotoneSplineCache(cache_dir, verify_checksums=True)
    selected = rows[[3, 0, 2]]
    actions = np.linspace(-0.03, 0.30, 997)

    acceptance = cache.acceptance(selected, actions)
    derivative = cache.derivative(selected, actions)

    for output_index, curve_index in enumerate((3, 0, 2)):
        expected, expected_derivative = _canonical(curves[curve_index], actions)
        np.testing.assert_allclose(acceptance[output_index], expected, atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(
            derivative[output_index], expected_derivative, atol=2e-4, rtol=0.0
        )
    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))
    assert np.all(np.diff(acceptance, axis=1) <= 2e-6)
    assert np.all(derivative[:, actions < 0.0] == 0.0)
    np.testing.assert_array_equal(
        cache.customer_ids(selected), ["customer-14", "customer-2", "customer-9"]
    )


def test_scalar_grid_and_pairwise_evaluation_preserve_selection_order(tmp_path: Path) -> None:
    cache_dir, rows, _ = _write_test_cache(tmp_path)
    cache = ShardedMonotoneSplineCache(cache_dir)
    selected = rows[[2, 0, 3]]
    actions = np.asarray([0.02, 0.08, 0.19])

    scalar = cache.acceptance(selected, 0.08)
    grid = cache.acceptance(selected, actions)
    pairwise = cache.acceptance(selected, actions, pairwise=True)

    assert scalar.shape == (3,)
    assert grid.shape == (3, 3)
    assert pairwise.shape == (3,)
    np.testing.assert_allclose(pairwise, np.diag(grid))


def test_shard_checksum_corruption_is_rejected(tmp_path: Path) -> None:
    cache_dir, _, _ = _write_test_cache(tmp_path)
    shard = cache_dir / "shards" / "shard_00000"
    with (shard / "upper_slopes.npy").open("ab") as handle:
        handle.write(b"corrupt")

    with np.testing.assert_raises_regex(ValueError, "Checksum mismatch"):
        validate_cache_shard(shard)


def test_missing_source_row_is_rejected(tmp_path: Path) -> None:
    cache_dir, _, _ = _write_test_cache(tmp_path)
    cache = ShardedMonotoneSplineCache(cache_dir)
    with np.testing.assert_raises_regex(KeyError, "absent from the cache"):
        cache.acceptance([999], 0.08)
