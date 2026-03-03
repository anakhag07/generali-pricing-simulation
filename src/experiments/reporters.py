"""Reporting utilities for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Protocol, runtime_checkable, Sequence

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from experiments.logging import log_grad, log_step, log_summary
from experiments.results import ExperimentResult
from experiments.visualization import (
    ESTIMATOR_STYLES,
    plot_gradient_norms,
    plot_loss_curves,
    plot_objective_u_slice,
    plot_step_sizes,
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
)
from optimization.steps import STEP_RULE_CONSTANT


@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    run_id: str
    run_dir: Path
    plots_dir: Path
    started_at: datetime


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", cleaned)
    return cleaned or "run"


def create_run_context(
    experiment_name: str,
    runs_root: str = "runs",
    started_at: datetime | None = None,
) -> RunContext:
    timestamp = started_at or datetime.now()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_name(experiment_name)
    run_dir = Path(runs_root) / safe_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    return RunContext(
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        plots_dir=plots_dir,
        started_at=timestamp,
    )


@runtime_checkable
class StepReporter(Protocol):
    def log_step(self, method: str, step: int, u: float, value: float) -> None:
        ...

    def log_grad(self, method: str, step: int, grad: float) -> None:
        ...


@runtime_checkable
class Reporter(Protocol):
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        ...

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        ...


class ReporterStack:
    def __init__(self, reporters: Sequence[Reporter]) -> None:
        self._reporters = list(reporters)

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        for reporter in self._reporters:
            reporter.on_start(run_context, config)

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        for reporter in self._reporters:
            reporter.on_end(run_context, result)

    def log_step(self, method: str, step: int, u: float, value: float) -> None:
        for reporter in self._reporters:
            if isinstance(reporter, StepReporter):
                reporter.log_step(method, step, u, value)

    def log_grad(self, method: str, step: int, grad: float) -> None:
        for reporter in self._reporters:
            if isinstance(reporter, StepReporter):
                reporter.log_grad(method, step, grad)


class ConsoleReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        print(f"\n=== Running experiment: {run_context.experiment_name} ===")

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        log_summary(result)

    def log_step(self, method: str, step: int, u: float, value: float) -> None:
        log_step(method, step, u, value)

    def log_grad(self, method: str, step: int, grad: float) -> None:
        log_grad(method, step, grad)


class JsonReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        payload = _build_summary_payload(run_context, result)
        summary_path = run_context.run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)


class PlotReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        config = result.config
        if not config.plot or not result.traces:
            return
        run_context.plots_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = str(run_context.plots_dir)
        objective_model = config.objective_model
        policy_spec = config.policy_spec
        traces = result.traces
        u_lbfgs = float(result.results["lbfgs"].u) if "lbfgs" in result.results else None
        u_star_plot = _u_star_for_plot(objective_model, result.u_star, u_lbfgs)
        plot_loss_curves(
            traces,
            plot_dir,
            u_star=u_star_plot,
        )
        plot_gradient_norms(traces, plot_dir)
        plot_objective_u_slice(
            result.x_samples,
            objective_model,
            traces,
            plot_dir,
            u_star=u_star_plot,
        )
        if config.step_rule != STEP_RULE_CONSTANT:
            plot_step_sizes(traces, plot_dir)
        if policy_spec.theta.size >= 2:
            axis_indices = (0, 1)
            axis_labels = None
            theta_path_points = [policy_spec.theta]
            for trace in traces.values():
                if trace.theta_values:
                    theta_path_points.extend(trace.theta_values)
            if policy_spec.theta.size > 2 and theta_path_points:
                axis_indices = select_theta_axes_max_variance(theta_path_points)
                axis_labels = (
                    f"theta[{axis_indices[0]}] (max-var axis)",
                    f"theta[{axis_indices[1]}] (max-var axis)",
                )
            ordered_results = [
                (name, result.results[name])
                for name in config.enabled_estimators
                if name in result.results
            ]
            theta_refs = [policy_spec.theta]
            theta_points = [(policy_spec.theta, "initial", "#636363", "o")]
            for name, estimator_result in ordered_results:
                theta_refs.append(estimator_result.theta)
                style = ESTIMATOR_STYLES[name]
                theta_points.append(
                    (
                        estimator_result.theta,
                        style["label"],
                        style["color"],
                        style["marker"],
                    )
                )
            plot_theta_objective_contours(
                result.x_samples,
                objective_model,
                policy_spec,
                policy_spec.theta,
                plot_dir,
                axis_indices=axis_indices,
                axis_labels=axis_labels,
                theta_refs=theta_refs,
                theta_points=theta_points,
                traces=traces,
            )


def _u_star_for_plot(
    objective_model: object,
    u_star: float | None,
    u_lbfgs: float | None = None,
) -> float | None:
    if isinstance(objective_model, FixedRegressionObjective):
        return None
    if u_star is not None:
        return u_star
    if u_lbfgs is None:
        return None
    return u_lbfgs


def _build_summary_payload(run_context: RunContext, result: ExperimentResult) -> dict:
    estimators: dict[str, dict] = {}
    for name, estimator_result in result.results.items():
        estimators[name] = {
            "final_u": float(estimator_result.u),
            "final_value": float(estimator_result.value),
            "runtime_sec": float(estimator_result.time),
            "theta": _as_list(estimator_result.theta),
        }

    trace_summary: dict[str, dict] = {}
    for name, trace in result.traces.items():
        if trace.objective_values:
            trace_summary[name] = {
                "steps": len(trace.steps),
                "final_objective": float(trace.objective_values[-1]),
                "min_objective": float(np.min(trace.objective_values)),
            }

    return {
        "run": {
            "experiment_name": run_context.experiment_name,
            "run_id": run_context.run_id,
            "started_at": run_context.started_at.isoformat(),
            "run_dir": str(run_context.run_dir),
        },
        "config": result.config.to_dict(),
        "initial_value": float(result.initial_value),
        "u_star": float(result.u_star) if result.u_star is not None else None,
        "value_at_u_star": float(result.value_at_u_star)
        if result.value_at_u_star is not None
        else None,
        "estimators": estimators,
        "trace_summary": trace_summary,
    }


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


__all__ = [
    "ConsoleReporter",
    "JsonReporter",
    "PlotReporter",
    "Reporter",
    "ReporterStack",
    "RunContext",
    "StepReporter",
    "create_run_context",
]
