"""Plot reporter and experiment-specific plot orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from data.loader import FEATURE_COLS_GLM, FEATURE_COLS_XGB, _load_observed_u_array
from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult
from objective.objectives import FixedRegressionObjective
from reporting.visualization import (
    _plot_policy_acceptance_histograms,
    _plot_policy_delta_u_by_elasticity,
    _plot_policy_delta_u_histograms,
    _plot_policy_final_summary_metrics,
    _plot_policy_objective_contribution_summary,
    _plot_policy_u_acceptance_histograms,
    _plot_policy_u_histograms,
    plot_gradient_norms,
    plot_loss_curves,
    plot_step_sizes,
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
)


class PlotReporter:
    """Writes standard optimization and policy diagnostic plots."""

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        del run_context, config

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        config = result.config
        if not config.plot or not result.traces:
            return
        run_context.plots_dir.mkdir(parents=True, exist_ok=True)
        optimization_dir = run_context.plots_dir / "optimization"
        policy_train_dir = run_context.plots_dir / "policy_train"
        policy_test_dir = run_context.plots_dir / "policy_test"
        optimization_dir.mkdir(parents=True, exist_ok=True)
        policy_train_dir.mkdir(parents=True, exist_ok=True)
        policy_test_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = str(optimization_dir)
        objective = config.objective
        traces = result.traces
        timings: dict[str, float] = {}

        def timed(name: str, fn: object, *args: object, **kwargs: object) -> None:
            start = time.perf_counter()
            try:
                fn(*args, **kwargs)  # type: ignore[misc]
            finally:
                timings[name] = time.perf_counter() - start

        timed(
            "loss_curves",
            plot_loss_curves,
            traces,
            plot_dir,
            u_star=_u_star_for_plot(objective, result.u_star),
            constant_u_baselines=result.constant_u_baselines,
        )
        timed("gradient_norms", plot_gradient_norms, traces, plot_dir)
        theta_by_estimator = {name: item.theta for name, item in result.results.items()}
        runtime_by_estimator = {name: item.time for name, item in result.results.items()}
        _plot_policy_diagnostics(
            result,
            result.x_samples,
            result.train_row_indices,
            str(policy_train_dir),
            "policy_train",
            theta_by_estimator,
            runtime_by_estimator,
            timed,
        )
        if result.x_test is not None:
            _plot_policy_diagnostics(
                result,
                result.x_test,
                result.test_row_indices,
                str(policy_test_dir),
                "policy_test",
                theta_by_estimator,
                runtime_by_estimator,
                timed,
            )
        if any(trace.step_sizes is not None for trace in traces.values()):
            timed("step_sizes", plot_step_sizes, traces, plot_dir)
        if config.theta0.size >= 2:
            _write_theta_contours(config, result, traces, objective, plot_dir, timed)
        with (run_context.plots_dir / "plot_timings.json").open("w", encoding="utf-8") as handle:
            json.dump({key: float(value) for key, value in timings.items()}, handle, indent=2, sort_keys=True)


def _write_theta_contours(
    config: ExperimentConfig,
    result: ExperimentResult,
    traces: object,
    objective: object,
    plot_dir: str,
    timed: object,
) -> None:
    axis_indices = (0, 1)
    axis_labels = None
    theta_path_points = [config.theta0]
    for trace in result.traces.values():
        if trace.theta_values:
            theta_path_points.extend(theta for theta in trace.theta_values if theta.size == config.theta0.size)
    if config.theta0.size > 2 and theta_path_points:
        axis_indices = select_theta_axes_max_variance(theta_path_points)
        axis_labels = (
            f"theta[{axis_indices[0]}] (max-var axis)",
            f"theta[{axis_indices[1]}] (max-var axis)",
        )
    theta_refs = list(theta_path_points)
    theta_points = [(config.theta0, "initial")]
    for name in config.enabled_estimators:
        estimator_result = result.results.get(name)
        if estimator_result is not None and estimator_result.theta.size == config.theta0.size:
            theta_refs.append(estimator_result.theta)
    first_order_result = result.results.get("first_order")
    if first_order_result is not None:
        theta_points.append((first_order_result.theta, "first-order final point"))
    timed(
        "theta_objective_contours",
        plot_theta_objective_contours,
        _contour_x_samples(result.x_samples, objective),
        objective,
        config.theta0,
        plot_dir,
        axis_indices=axis_indices,
        axis_labels=axis_labels,
        theta_refs=theta_refs,
        theta_points=theta_points,
        traces=traces,
        grid_size=_contour_grid_size(objective),
    )


def _contour_x_samples(x_samples: object, objective: object, max_rows: int = 200) -> object:
    """Return deterministic contour-evaluation rows, subsampling costly model objectives."""
    if hasattr(x_samples, "iloc"):
        x_arr = x_samples.reset_index(drop=True)
    else:
        x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array/DataFrame.")
    is_model_based = hasattr(objective, "acceptance_model") and hasattr(objective, "loss_model")
    if not is_model_based or x_arr.shape[0] <= max_rows:
        return x_arr
    indices = np.linspace(0, x_arr.shape[0] - 1, max_rows, dtype=int)
    if hasattr(x_arr, "iloc"):
        return x_arr.iloc[indices].reset_index(drop=True)
    return x_arr[indices]


def _contour_grid_size(objective: object, default_grid_size: int = 60, model_based_grid_size: int = 20) -> int:
    """Return contour grid resolution, lowering only costly model-based diagnostics."""
    is_model_based = hasattr(objective, "acceptance_model") and hasattr(objective, "loss_model")
    return model_based_grid_size if is_model_based else default_grid_size


def _u_star_for_plot(action_objective: object, u_star: float | None) -> float | None:
    if isinstance(action_objective, FixedRegressionObjective):
        return None
    return u_star


def _observed_u_reference(
    result: ExperimentResult,
    x_samples: object,
    row_indices: np.ndarray | None,
) -> np.ndarray | None:
    objective = result.config.objective
    if result.config.x_fixed is None:
        return None
    if not hasattr(objective, "acceptance_model") or not hasattr(objective, "loss_model"):
        return None
    state_dim = int(result.config.state_dim)
    if state_dim == len(FEATURE_COLS_GLM):
        model_type = "glm"
    elif state_dim == len(FEATURE_COLS_XGB):
        model_type = "xgb"
    else:
        return None
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=int)
        if row_indices.shape != (_x_sample_count(x_samples),):
            return None
    return _load_observed_u_array(
        model_type,
        n_rows=_x_sample_count(x_samples),
        row_indices=row_indices,
        seed=result.config.seed,
    )


def _x_sample_count(x_samples: object) -> int:
    return int(x_samples.shape[0])


def _plot_policy_diagnostics(
    result: ExperimentResult,
    x_samples: object,
    row_indices: np.ndarray | None,
    plot_dir: str,
    timing_prefix: str,
    theta_by_estimator: dict[str, np.ndarray],
    runtime_by_estimator: dict[str, float],
    timed: object,
) -> None:
    observed_u = _observed_u_reference(result, x_samples, row_indices)
    if observed_u is None:
        return
    objective = result.config.objective
    timed(f"{timing_prefix}_policy_u_histograms", _plot_policy_u_histograms, observed_u, x_samples, objective, theta_by_estimator, plot_dir)
    timed(f"{timing_prefix}_policy_acceptance_histograms", _plot_policy_acceptance_histograms, observed_u, x_samples, objective, theta_by_estimator, plot_dir)
    timed(f"{timing_prefix}_policy_final_summary_metrics", _plot_policy_final_summary_metrics, x_samples, objective, theta_by_estimator, runtime_by_estimator, plot_dir)
    timed(f"{timing_prefix}_policy_delta_u_histograms", _plot_policy_delta_u_histograms, observed_u, x_samples, objective, theta_by_estimator, plot_dir)
    timed(f"{timing_prefix}_policy_delta_u_by_elasticity", _plot_policy_delta_u_by_elasticity, observed_u, x_samples, objective, theta_by_estimator, plot_dir)
    timed(f"{timing_prefix}_policy_objective_contribution_summary", _plot_policy_objective_contribution_summary, x_samples, objective, theta_by_estimator, plot_dir)
    timed(
        f"{timing_prefix}_policy_u_acceptance_histograms",
        _plot_policy_u_acceptance_histograms,
        x_samples,
        objective,
        theta_by_estimator,
        str(Path(plot_dir) / "u_acceptance"),
        acceptance_floor=result.config.acceptance_floor,
    )


__all__ = [
    "PlotReporter",
    "_contour_grid_size",
    "_contour_x_samples",
    "_observed_u_reference",
    "_u_star_for_plot",
]
