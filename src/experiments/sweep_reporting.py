"""Aggregate CSV and frontier-plot helpers for completed sweeps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.results import PolicyEvaluation
from experiments.sweep_utils import SweepRunResult
from reporting.visualization import plot_sweep_pareto_frontier, plot_sweep_tradeoffs


@dataclass(frozen=True)
class SweepFrontierMetric:
    """One y-axis metric for a sweep Pareto frontier plot."""

    y_key: str
    y_label: str
    filename: str


DEFAULT_FRONTIER_METRICS: tuple[SweepFrontierMetric, ...] = (
    SweepFrontierMetric(
        y_key="value",
        y_label="Final objective value",
        filename="pareto_objective_acceptance.png",
    ),
    SweepFrontierMetric(
        y_key="u",
        y_label="Final u",
        filename="pareto_u_acceptance.png",
    ),
)


def timestamped_sweep_output_dir(
    *,
    project_name: str,
    dirname_prefix: str,
    runs_root: str = "outputs",
) -> Path:
    """Return and create a timestamped aggregate sweep output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(runs_root) / _path_part(project_name) / f"{dirname_prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_config_sweep_final_rows(
    sweep_results: Sequence[SweepRunResult],
    *,
    config_attr: str,
    sweep_key: str,
    include_constraint_violation: bool = False,
) -> list[dict[str, object]]:
    """Collect final estimator rows keyed by a scalar ExperimentConfig field."""
    rows: list[dict[str, object]] = []
    for sweep_result in sweep_results:
        sweep_value = getattr(sweep_result.result.config, config_attr)
        if sweep_value is None:
            continue
        for estimator, estimator_result in sweep_result.result.results.items():
            if estimator_result.mean_acceptance is None:
                continue
            row: dict[str, object] = {
                "run_name": sweep_result.run_name,
                "estimator": estimator,
                sweep_key: float(sweep_value),
                "u": float(estimator_result.u),
                "mean_acceptance": float(estimator_result.mean_acceptance),
                "value": float(estimator_result.value),
            }
            if include_constraint_violation:
                row["constraint_violation"] = _optional_float(
                    estimator_result.constraint_violation
                )
            rows.append(row)
    return rows


def write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    """Write row dictionaries to CSV, filling absent fields with blanks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_sweep_frontier_plots(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    sweep_key: str,
    sweep_label: str,
    tradeoff_filename: str,
    frontier_metrics: Sequence[SweepFrontierMetric] = DEFAULT_FRONTIER_METRICS,
) -> None:
    """Write standard action/acceptance and Pareto frontier plots for a sweep."""
    plot_dir = str(output_dir)
    plot_sweep_tradeoffs(
        rows,
        plot_dir,
        sweep_key=sweep_key,
        sweep_label=sweep_label,
        filename=tradeoff_filename,
    )
    for metric in frontier_metrics:
        plot_sweep_pareto_frontier(
            rows,
            plot_dir,
            sweep_key=sweep_key,
            sweep_label=sweep_label,
            y_key=metric.y_key,
            y_label=metric.y_label,
            filename=metric.filename,
        )


# --- Cross-seed aggregation (per variant x estimator over seed replicates) ---

_SEED_SUMMARY_METRICS: tuple[str, ...] = (
    "final_value",
    "final_u",
    "runtime_sec",
    "mean_acceptance",
    "train_objective_value",
    "test_objective_value",
)

SEED_GRID_FINAL_FIELDNAMES: tuple[str, ...] = (
    "variant",
    "run_seed",
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
)

SEED_GRID_SUMMARY_FIELDNAMES: tuple[str, ...] = ("variant", "estimator", "n_seeds") + tuple(
    f"{metric}_{stat}"
    for metric in _SEED_SUMMARY_METRICS
    for stat in ("mean", "std", "min", "max")
)


def collect_seed_grid_final_rows(
    sweep_results: Sequence[SweepRunResult],
) -> list[dict[str, object]]:
    """Collect per-(variant, seed, estimator) final rows from seed-replicated runs."""
    rows: list[dict[str, object]] = []
    for sweep_result in sweep_results:
        result = sweep_result.result
        run_dir = str(sweep_result.run_context.run_dir)
        for estimator, estimator_result in result.results.items():
            row: dict[str, object] = {
                "variant": sweep_result.run_name,
                "run_seed": "" if sweep_result.run_seed is None else int(sweep_result.run_seed),
                "run_dir": run_dir,
                "estimator": estimator,
                "final_u": float(estimator_result.u),
                "final_value": float(estimator_result.value),
                "runtime_sec": float(estimator_result.time),
                "mean_acceptance": _optional_float(estimator_result.mean_acceptance),
                "constraint_violation": _optional_float(estimator_result.constraint_violation),
            }
            row.update(_evaluation_fields("train", result.train_metrics.get(estimator)))
            row.update(_evaluation_fields("test", result.test_metrics.get(estimator)))
            rows.append(row)
    return rows


def aggregate_seed_grid_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate final rows to mean/std/min/max per (variant, estimator) over seeds."""
    keys = sorted({(str(row.get("variant")), str(row.get("estimator"))) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for variant, estimator in keys:
        group = [
            row
            for row in rows
            if str(row.get("variant")) == variant and str(row.get("estimator")) == estimator
        ]
        summary: dict[str, object] = {
            "variant": variant,
            "estimator": estimator,
            "n_seeds": len(group),
        }
        for metric in _SEED_SUMMARY_METRICS:
            values = [_row_float(item.get(metric)) for item in group]
            finite = np.asarray([value for value in values if value is not None], dtype=float)
            if finite.size == 0:
                for stat in ("mean", "std", "min", "max"):
                    summary[f"{metric}_{stat}"] = ""
                continue
            summary[f"{metric}_mean"] = float(np.mean(finite))
            summary[f"{metric}_std"] = float(np.std(finite, ddof=0))
            summary[f"{metric}_min"] = float(np.min(finite))
            summary[f"{metric}_max"] = float(np.max(finite))
        summary_rows.append(summary)
    return summary_rows


def write_seed_grid_csvs(
    output_dir: Path,
    final_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write the per-seed final rows and the cross-seed aggregate summary CSVs."""
    write_rows_csv(output_dir / "seed_grid_finals.csv", final_rows, SEED_GRID_FINAL_FIELDNAMES)
    write_rows_csv(output_dir / "seed_grid_summary.csv", summary_rows, SEED_GRID_SUMMARY_FIELDNAMES)


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
        f"{prefix}_mean_u": float(evaluation.mean_u),
        f"{prefix}_mean_acceptance": _optional_float(evaluation.mean_acceptance),
    }


def _row_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _path_part(value: object) -> str:
    text = str(value)
    return text.replace(" ", "").replace("/", "-")


__all__ = [
    "DEFAULT_FRONTIER_METRICS",
    "SEED_GRID_FINAL_FIELDNAMES",
    "SEED_GRID_SUMMARY_FIELDNAMES",
    "SweepFrontierMetric",
    "aggregate_seed_grid_rows",
    "collect_config_sweep_final_rows",
    "collect_seed_grid_final_rows",
    "timestamped_sweep_output_dir",
    "write_rows_csv",
    "write_seed_grid_csvs",
    "write_sweep_frontier_plots",
]
