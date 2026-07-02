"""Aggregate CSV and frontier-plot helpers for completed sweeps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

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


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _path_part(value: object) -> str:
    text = str(value)
    return text.replace(" ", "").replace("/", "-")


__all__ = [
    "DEFAULT_FRONTIER_METRICS",
    "SweepFrontierMetric",
    "collect_config_sweep_final_rows",
    "timestamped_sweep_output_dir",
    "write_rows_csv",
    "write_sweep_frontier_plots",
]
