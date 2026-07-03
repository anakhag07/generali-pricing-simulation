from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from experiments.sweep_reporting import (
    aggregate_seed_grid_rows,
    collect_config_sweep_final_rows,
    collect_seed_grid_final_rows,
    write_rows_csv,
)
from experiments.sweep_utils import SweepRunResult


@dataclass(frozen=True)
class _EstimatorResult:
    u: float
    value: float
    mean_acceptance: float | None
    constraint_violation: float | None = None


def _sweep_result() -> SweepRunResult:
    result = SimpleNamespace(
        config=SimpleNamespace(acceptance_floor=0.9),
        results={
            "first_order": _EstimatorResult(
                u=0.1,
                value=-2.0,
                mean_acceptance=0.92,
                constraint_violation=0.0,
            ),
            "spsa": _EstimatorResult(u=0.2, value=-1.5, mean_acceptance=None),
        },
    )
    return SweepRunResult(
        run_name="floor-0.9",
        config=result.config,
        overrides={"acceptance_floor": 0.9},
        result=result,
        run_context=SimpleNamespace(run_dir="outputs/run"),
    )


def test_collect_config_sweep_final_rows_skips_missing_acceptance() -> None:
    rows = collect_config_sweep_final_rows(
        [_sweep_result()],
        config_attr="acceptance_floor",
        sweep_key="c",
        include_constraint_violation=True,
    )

    assert rows == [
        {
            "run_name": "floor-0.9",
            "estimator": "first_order",
            "c": 0.9,
            "u": 0.1,
            "mean_acceptance": 0.92,
            "value": -2.0,
            "constraint_violation": 0.0,
        }
    ]


@dataclass(frozen=True)
class _SeedEstimatorResult:
    u: float
    value: float
    time: float
    mean_acceptance: float | None = None
    constraint_violation: float | None = None


def _seed_sweep_result(run_seed: int, value: float) -> SweepRunResult:
    result = SimpleNamespace(
        results={"first_order": _SeedEstimatorResult(u=0.1 * run_seed, value=value, time=0.5)},
        train_metrics={},
        test_metrics={},
    )
    return SweepRunResult(
        run_name="variant-a",
        config=SimpleNamespace(),
        overrides={},
        result=result,
        run_context=SimpleNamespace(run_dir="outputs/run"),
        run_seed=run_seed,
    )


def test_collect_and_aggregate_seed_grid_rows() -> None:
    finals = collect_seed_grid_final_rows(
        [_seed_sweep_result(1, value=1.0), _seed_sweep_result(2, value=3.0)]
    )
    assert len(finals) == 2
    assert {row["run_seed"] for row in finals} == {1, 2}

    summary = aggregate_seed_grid_rows(finals)
    assert len(summary) == 1
    row = summary[0]
    assert row["variant"] == "variant-a"
    assert row["estimator"] == "first_order"
    assert row["n_seeds"] == 2
    assert row["final_value_mean"] == 2.0
    assert row["final_value_std"] == 1.0
    # metrics absent from all rows aggregate to blank, not a crash
    assert row["mean_acceptance_mean"] == ""


def test_write_rows_csv_fills_missing_fields(tmp_path) -> None:
    output = tmp_path / "rows.csv"

    write_rows_csv(output, [{"a": 1}], fieldnames=["a", "b"])

    assert output.read_text(encoding="utf-8") == "a,b\n1,\n"
