"""Utilities for running named preset comparisons with aggregate outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.config import ExperimentConfig
from experiments.configs import get_config
from experiments.reporters import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    PolicyArtifactReporter,
    ReporterStack,
    WandbReporter,
    create_run_context,
)
from experiments.results import ExperimentResult
from experiments.run import run_experiment
from reporting.visualization import (
    plot_comparison_final_metric,
    plot_comparison_objective_curves,
    plot_comparison_u_curves,
)


@dataclass(frozen=True)
class ComparisonSpec:
    """One named comparison variant backed by a registered config preset."""

    name: str
    preset: str
    overrides: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonRun:
    """Resolved config for one comparison variant before execution."""

    name: str
    preset: str
    config: ExperimentConfig
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class ComparisonResult:
    """Completed result for one named comparison variant."""

    name: str
    preset: str
    result: ExperimentResult


def generate_comparison_runs(
    *,
    specs: Sequence[ComparisonSpec],
    common_overrides: Mapping[str, Any] | None = None,
) -> list[ComparisonRun]:
    """Resolve comparison specs into runnable configs with merged overrides."""
    if not specs:
        raise ValueError("At least one comparison spec is required.")
    _validate_unique_names(specs)

    common = dict(common_overrides or {})
    runs: list[ComparisonRun] = []
    for spec in specs:
        if not spec.name:
            raise ValueError("Comparison spec names must be non-empty.")
        overrides = {**common, **dict(spec.overrides)}
        config = get_config(spec.preset, overrides=overrides)
        runs.append(
            ComparisonRun(
                name=spec.name,
                preset=spec.preset,
                config=config,
                overrides=overrides,
            )
        )
    return runs


def run_preset_comparison(
    *,
    specs: Sequence[ComparisonSpec],
    common_overrides: Mapping[str, Any] | None = None,
    project_name: str,
    runs_root: str = "outputs",
    validate_shared_estimators: bool = True,
    validate_shared_x: bool = False,
    write_aggregate_outputs: bool = True,
) -> list[ComparisonResult]:
    """Execute named preset variants and optionally write aggregate comparison outputs."""
    comparison_runs = generate_comparison_runs(
        specs=specs,
        common_overrides=common_overrides,
    )
    results: list[ComparisonResult] = []
    runs_root_path = _project_runs_root(runs_root, project_name)

    for run in comparison_runs:
        run_context = create_run_context(run.name, runs_root=runs_root_path)
        reporter_list = [
            ConsoleReporter(verbose=run.config.verbose),
            FileStepLogger(),
            PolicyArtifactReporter(),
            JsonReporter(),
            PlotReporter(),
        ]
        if run.config.wandb_enabled:
            reporter_list.append(WandbReporter())
        reporters = ReporterStack(reporter_list)
        reporters.on_start(run_context, run.config)
        result = run_experiment(run.config, step_reporter=reporters)
        reporters.on_end(run_context, result)
        results.append(ComparisonResult(name=run.name, preset=run.preset, result=result))

    if validate_shared_estimators:
        validate_comparison_estimators(results)
    if validate_shared_x:
        validate_comparison_x_samples(results)
    if write_aggregate_outputs:
        output_dir = _comparison_output_dir(runs_root_path)
        write_comparison_outputs(results, output_dir)

    return results


def collect_comparison_trace_rows(
    results: Sequence[ComparisonResult],
) -> list[dict[str, object]]:
    """Flatten per-step traces into CSV/plot friendly comparison rows."""
    rows: list[dict[str, object]] = []
    for comparison_result in results:
        for estimator, trace in comparison_result.result.traces.items():
            for index, step in enumerate(trace.steps):
                rows.append(
                    {
                        "comparison": comparison_result.name,
                        "preset": comparison_result.preset,
                        "estimator": estimator,
                        "step": int(step),
                        "u": _sequence_value(trace.u_values, index),
                        "objective": _sequence_value(trace.objective_values, index),
                        "theta_grad_norm": _optional_sequence_value(trace.theta_grad_norms, index),
                        "true_theta_grad_norm": _optional_sequence_value(
                            trace.true_theta_grad_norms,
                            index,
                        ),
                        "step_size": _optional_sequence_value(trace.step_sizes, index),
                    }
                )
    return rows


def collect_comparison_final_rows(
    results: Sequence[ComparisonResult],
) -> list[dict[str, object]]:
    """Flatten final estimator results into CSV/plot friendly comparison rows."""
    rows: list[dict[str, object]] = []
    for comparison_result in results:
        n_objective_terms = int(comparison_result.result.x_samples.shape[0])
        for estimator, estimator_result in comparison_result.result.results.items():
            rows.append(
                {
                    "comparison": comparison_result.name,
                    "preset": comparison_result.preset,
                    "estimator": estimator,
                    "final_u": float(estimator_result.u),
                    "final_value": float(estimator_result.value),
                    "final_objective_sum": n_objective_terms * float(estimator_result.value),
                    "runtime_sec": float(estimator_result.time),
                    "mean_acceptance": _optional_float(estimator_result.mean_acceptance),
                    "constraint_violation": _optional_float(estimator_result.constraint_violation),
                    "acceptance_multiplier": _optional_float(estimator_result.acceptance_multiplier),
                    "constraint_penalty": _optional_float(estimator_result.constraint_penalty),
                }
            )
    return rows


def write_comparison_outputs(
    results: Sequence[ComparisonResult],
    output_dir: Path,
) -> None:
    """Write aggregate comparison CSVs and plots under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = collect_comparison_trace_rows(results)
    final_rows = collect_comparison_final_rows(results)

    _write_rows(
        output_dir / "comparison_traces.csv",
        trace_rows,
        fieldnames=[
            "comparison",
            "preset",
            "estimator",
            "step",
            "u",
            "objective",
            "theta_grad_norm",
            "true_theta_grad_norm",
            "step_size",
        ],
    )
    _write_rows(
        output_dir / "comparison_finals.csv",
        final_rows,
        fieldnames=[
            "comparison",
            "preset",
            "estimator",
            "final_u",
            "final_value",
            "final_objective_sum",
            "runtime_sec",
            "mean_acceptance",
            "constraint_violation",
            "acceptance_multiplier",
            "constraint_penalty",
        ],
    )

    plot_dir = str(output_dir)
    plot_comparison_objective_curves(trace_rows, plot_dir)
    plot_comparison_u_curves(trace_rows, plot_dir)
    plot_comparison_final_metric(
        final_rows,
        plot_dir,
        metric_key="final_value",
        metric_label="Final objective value",
        filename="final_objective.png",
    )
    plot_comparison_final_metric(
        final_rows,
        plot_dir,
        metric_key="final_objective_sum",
        metric_label="Final summed objective value",
        filename="final_objective_sum.png",
    )
    plot_comparison_final_metric(
        final_rows,
        plot_dir,
        metric_key="final_u",
        metric_label="Final u",
        filename="final_u.png",
    )
    if any(row.get("mean_acceptance") != "" for row in final_rows):
        plot_comparison_final_metric(
            final_rows,
            plot_dir,
            metric_key="mean_acceptance",
            metric_label="Mean acceptance",
            filename="mean_acceptance.png",
        )


def validate_comparison_estimators(results: Sequence[ComparisonResult]) -> None:
    """Raise when comparison variants do not share the same estimator set."""
    if not results:
        return
    expected = tuple(results[0].result.results.keys())
    expected_set = set(expected)
    for comparison_result in results[1:]:
        estimator_set = set(comparison_result.result.results.keys())
        if estimator_set != expected_set:
            raise ValueError(
                "Comparison variants must share estimator sets when "
                "validate_shared_estimators=True. "
                f"Expected {sorted(expected_set)}, got {sorted(estimator_set)} "
                f"for '{comparison_result.name}'."
            )


def validate_comparison_x_samples(results: Sequence[ComparisonResult]) -> None:
    """Raise when comparison variants do not share the same state samples."""
    if not results:
        return
    expected = np.asarray(results[0].result.x_samples, dtype=float)
    for comparison_result in results[1:]:
        x_samples = np.asarray(comparison_result.result.x_samples, dtype=float)
        if x_samples.shape != expected.shape or not np.allclose(x_samples, expected):
            raise ValueError(
                "Comparison variants must share x_samples when validate_shared_x=True. "
                f"Variant '{comparison_result.name}' has shape {x_samples.shape}; "
                f"expected {expected.shape}."
            )


def _validate_unique_names(specs: Sequence[ComparisonSpec]) -> None:
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise ValueError(f"Comparison spec names must be unique: {duplicate_text}.")


def _project_runs_root(runs_root: str, project_name: str) -> str:
    return str(Path(runs_root) / _stringify_path_part(project_name))


def _comparison_output_dir(runs_root_path: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(runs_root_path) / f"comparison_{timestamp}"


def _stringify_path_part(value: object) -> str:
    text = str(value)
    return text.replace(" ", "").replace("/", "-")


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sequence_value(values: Sequence[float], index: int) -> float:
    return float(values[index])


def _optional_sequence_value(values: Sequence[float] | None, index: int) -> float | str:
    if values is None:
        return ""
    if index >= len(values):
        return ""
    return float(values[index])


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


__all__ = [
    "ComparisonResult",
    "ComparisonRun",
    "ComparisonSpec",
    "collect_comparison_final_rows",
    "collect_comparison_trace_rows",
    "generate_comparison_runs",
    "run_preset_comparison",
    "validate_comparison_estimators",
    "validate_comparison_x_samples",
    "write_comparison_outputs",
]
