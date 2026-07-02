from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from experiments.sweep_reporting import collect_config_sweep_final_rows, write_rows_csv
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


def test_write_rows_csv_fills_missing_fields(tmp_path) -> None:
    output = tmp_path / "rows.csv"

    write_rows_csv(output, [{"a": 1}], fieldnames=["a", "b"])

    assert output.read_text(encoding="utf-8") == "a,b\n1,\n"
