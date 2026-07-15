"""Run repeated experiments with explicit seed-stream variation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.paths import results_root
from experiments.results import ExperimentResult, PolicyEvaluation
from experiments.seeds import SeedSetup, SeedStream, replicate_seed_setup


@dataclass(frozen=True)
class SeedRepeatSpec:
    """Configuration for repeated runs over one or more seed streams."""

    base_preset: str
    run_seeds: tuple[int, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)
    vary: tuple[SeedStream, ...] = ("optimizer",)
    fixed_data_seed: int | None = None
    fixed_split_seed: int | None = None
    fixed_theta_seed: int | None = None
    fixed_noise_seed: int | None = None
    fixed_optimizer_seed: int | None = None
    output_root: str | Path | None = None
    project_name: str = "seed-repeats"

    def __post_init__(self) -> None:
        if not self.run_seeds:
            raise ValueError("run_seeds must contain at least one seed.")
        run_seeds = tuple(int(seed) for seed in self.run_seeds)
        object.__setattr__(self, "run_seeds", run_seeds)
        allowed = {"data", "split", "theta", "noise", "optimizer", "all"}
        unknown = sorted(set(self.vary) - allowed)
        if unknown:
            raise ValueError(f"Unknown seed streams: {', '.join(unknown)}.")
        if "all" in self.vary and len(self.vary) > 1:
            raise ValueError("vary=('all',) cannot be combined with other seed streams.")


@dataclass(frozen=True)
class SeedRepeatOutput:
    """Output paths and rows from a completed seed-repeat run."""

    output_dir: Path
    final_rows: list[dict[str, object]]
    summary_rows: list[dict[str, object]]
    results: list[tuple[int, str, ExperimentResult]]


def seed_setup_for_repeat(spec: SeedRepeatSpec, run_seed: int) -> SeedSetup:
    """Build the concrete ``SeedSetup`` for one repeated run.

    Thin wrapper over ``experiments.seeds.replicate_seed_setup`` that maps the
    spec's per-stream ``fixed_*_seed`` overrides onto the shared replication policy.
    """
    fixed = {
        "data": spec.fixed_data_seed,
        "split": spec.fixed_split_seed,
        "theta": spec.fixed_theta_seed,
        "noise": spec.fixed_noise_seed,
        "optimizer": spec.fixed_optimizer_seed,
    }
    return replicate_seed_setup(
        int(run_seed),
        int(spec.run_seeds[0]),
        vary=spec.vary,
        fixed=fixed,
    )


def run_seed_repeats(spec: SeedRepeatSpec) -> SeedRepeatOutput:
    """Run a preset repeatedly and write aggregate seed-repeat CSVs."""
    output_dir = _seed_repeat_output_dir(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, str, ExperimentResult]] = []
    final_rows: list[dict[str, object]] = []
    for run_seed in spec.run_seeds:
        seed_setup = seed_setup_for_repeat(spec, run_seed)
        overrides = {**dict(spec.overrides), "seed_setup": seed_setup}
        config = get_config(spec.base_preset, overrides=overrides)
        run_name = f"seed-{int(run_seed)}"
        executed = execute_experiment_run(
            run_name,
            config,
            runs_root=output_dir / "runs",
            run_metadata={
                "preset_name": spec.base_preset,
                "variant_name": run_name,
                "overrides": dict(spec.overrides),
                "run_seed": int(run_seed),
            },
        )
        run_context = executed.run_context
        result = executed.result
        results.append((int(run_seed), run_name, result))
        final_rows.extend(_final_rows(int(run_seed), run_name, str(run_context.run_dir), result))

    summary_rows = _summary_rows(final_rows)
    _write_rows(output_dir / "seed_repeats.csv", final_rows, _FINAL_FIELDNAMES)
    _write_rows(output_dir / "seed_repeats_summary.csv", summary_rows, _SUMMARY_FIELDNAMES)
    return SeedRepeatOutput(
        output_dir=output_dir,
        final_rows=final_rows,
        summary_rows=summary_rows,
        results=results,
    )


def _final_rows(
    run_seed: int,
    run_name: str,
    run_dir: str,
    result: ExperimentResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for estimator, estimator_result in result.results.items():
        row: dict[str, object] = {
            "run_seed": int(run_seed),
            "run_name": run_name,
            "run_dir": run_dir,
            "estimator": estimator,
            "final_u": _optional_float(estimator_result.u),
            "final_value": float(estimator_result.value),
            "runtime_sec": float(estimator_result.time),
            "mean_acceptance": _optional_float(estimator_result.mean_acceptance),
            "constraint_violation": _optional_float(estimator_result.constraint_violation),
        }
        row.update(_evaluation_fields("train", result.train_metrics.get(estimator)))
        row.update(_evaluation_fields("test", result.test_metrics.get(estimator)))
        rows.append(row)
    return rows


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    estimators = sorted({str(row.get("estimator")) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for estimator in estimators:
        estimator_rows = [row for row in rows if str(row.get("estimator")) == estimator]
        row: dict[str, object] = {
            "estimator": estimator,
            "n_runs": len(estimator_rows),
        }
        for metric in _SUMMARY_METRICS:
            values = [_row_float(item.get(metric)) for item in estimator_rows]
            finite = np.asarray([value for value in values if value is not None], dtype=float)
            if finite.size == 0:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_min"] = ""
                row[f"{metric}_max"] = ""
                continue
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_std"] = float(np.std(finite, ddof=0))
            row[f"{metric}_min"] = float(np.min(finite))
            row[f"{metric}_max"] = float(np.max(finite))
        summary_rows.append(row)
    return summary_rows


def _evaluation_fields(prefix: str, evaluation: PolicyEvaluation | None) -> dict[str, object]:
    if evaluation is None:
        return {
            f"{prefix}_objective_value": "",
            f"{prefix}_objective_sum": "",
            f"{prefix}_mean_u": "",
            f"{prefix}_mean_acceptance": "",
        }
    return {
        f"{prefix}_objective_value": float(evaluation.objective_value),
        f"{prefix}_objective_sum": float(evaluation.objective_sum),
        f"{prefix}_mean_u": _optional_float(evaluation.mean_u),
        f"{prefix}_mean_acceptance": _optional_float(evaluation.mean_acceptance),
    }


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _seed_repeat_output_dir(spec: SeedRepeatSpec) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = results_root() if spec.output_root is None else Path(spec.output_root)
    return root / spec.project_name / f"seed_repeats_{timestamp}"


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _row_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


_FINAL_FIELDNAMES = [
    "run_seed",
    "run_name",
    "run_dir",
    "estimator",
    "final_u",
    "final_value",
    "runtime_sec",
    "mean_acceptance",
    "constraint_violation",
    "train_objective_value",
    "train_objective_sum",
    "train_mean_u",
    "train_mean_acceptance",
    "test_objective_value",
    "test_objective_sum",
    "test_mean_u",
    "test_mean_acceptance",
]

_SUMMARY_METRICS = (
    "final_value",
    "final_u",
    "runtime_sec",
    "mean_acceptance",
    "train_objective_value",
    "test_objective_value",
)

_SUMMARY_FIELDNAMES = ["estimator", "n_runs"] + [
    f"{metric}_{stat}"
    for metric in _SUMMARY_METRICS
    for stat in ("mean", "std", "min", "max")
]


__all__ = [
    "SeedRepeatOutput",
    "SeedRepeatSpec",
    "run_seed_repeats",
    "seed_setup_for_repeat",
]
