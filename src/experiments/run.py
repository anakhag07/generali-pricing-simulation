"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from data.models import Customer, default_rng
from experiments.config import ExperimentConfig
from experiments.helpers import run_first_order, run_lbfgs_theta, run_zeroth_order
from experiments.logging import log_summary
from experiments.visualization import (
    ESTIMATOR_STYLES,
    plot_gradient_norms,
    plot_loss_curves,
    plot_objective_u_slice,
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
)
from optimization.policy import policy_u


@dataclass(frozen=True)
class EstimatorResult:
    theta: np.ndarray
    u: float
    value: float
    time: float


def _objective_u_star(objective_model: object) -> float | None:
    objective = cast(Any, objective_model)
    optimal_u = getattr(objective, "optimal_u", None)
    if callable(optimal_u):
        return float(cast(Any, optimal_u)())
    u_star = getattr(objective, "u_star", None)
    if u_star is not None:
        return float(cast(Any, u_star))
    return None


def _u_star_for_plot(
    objective_model: object,
    u_star: float | None,
    u_lbfgs: float | None,
) -> float | None:
    if u_star is not None:
        return u_star
    if u_lbfgs is None:
        return None
    if isinstance(objective_model, FixedRegressionObjective):
        return None
    return u_lbfgs


def run_experiment(
    config: ExperimentConfig,
) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    objective_model = config.objective_model
    policy_spec = config.policy_spec
    enabled_estimators = tuple(config.enabled_estimators)

    rng = default_rng(config.seed)
    customers = [Customer.sample(rng, state_dim=config.state_dim) for _ in range(config.n_samples)]
    x_samples = [customer.x for customer in customers]

    theta_initial = policy_spec.theta
    u_initials = [policy_u(theta_initial, x, kind=policy_spec.kind) for x in x_samples]
    initial_value = float(
        sum(objective_model.value(x, u) for x, u in zip(x_samples, u_initials)) / len(x_samples)
    )
    u_star = _objective_u_star(objective_model)
    value_at_u_star = None
    if u_star is not None:
        value_at_u_star = float(
            sum(objective_model.value(x, u_star) for x in x_samples) / len(x_samples)
        )
    results: dict[str, EstimatorResult] = {}
    traces = {}

    if "first_order" in enabled_estimators:
        start_first = time.perf_counter()
        theta_first, trace_first = run_first_order(
            theta_initial,
            policy_spec.kind,
            x_samples,
            objective_model,
            rng,
            config.t_steps,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            log_steps=config.log_steps,
        )
        time_first = time.perf_counter() - start_first
        u_first_values = [policy_u(theta_first, x, kind=policy_spec.kind) for x in x_samples]
        u_first = float(sum(u_first_values) / len(u_first_values))
        value_first = float(
            sum(objective_model.value(x, u) for x, u in zip(x_samples, u_first_values))
            / len(x_samples)
        )
        results["first_order"] = EstimatorResult(
            theta=theta_first,
            u=u_first,
            value=value_first,
            time=time_first,
        )
        traces["first_order"] = trace_first

    if "zeroth_order" in enabled_estimators:
        start_zero = time.perf_counter()
        theta_zero, trace_zero = run_zeroth_order(
            theta_initial,
            policy_spec.kind,
            x_samples,
            objective_model,
            rng,
            config.t_steps,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            log_steps=config.log_steps,
        )
        time_zero = time.perf_counter() - start_zero
        u_zero_values = [policy_u(theta_zero, x, kind=policy_spec.kind) for x in x_samples]
        u_zero = float(sum(u_zero_values) / len(u_zero_values))
        value_zero = float(
            sum(objective_model.value(x, u) for x, u in zip(x_samples, u_zero_values))
            / len(x_samples)
        )
        results["zeroth_order"] = EstimatorResult(
            theta=theta_zero,
            u=u_zero,
            value=value_zero,
            time=time_zero,
        )
        traces["zeroth_order"] = trace_zero

    if "lbfgs" in enabled_estimators:
        start_lbfgs = time.perf_counter()
        theta_lbfgs, value_lbfgs, trace_lbfgs = run_lbfgs_theta(
            theta_initial,
            policy_spec.kind,
            x_samples,
            objective_model,
            config.lbfgs_maxiter,
        )
        time_lbfgs = time.perf_counter() - start_lbfgs
        u_lbfgs_values = [policy_u(theta_lbfgs, x, kind=policy_spec.kind) for x in x_samples]
        u_lbfgs = float(sum(u_lbfgs_values) / len(u_lbfgs_values))
        results["lbfgs"] = EstimatorResult(
            theta=theta_lbfgs,
            u=u_lbfgs,
            value=float(value_lbfgs),
            time=time_lbfgs,
        )
        traces["lbfgs"] = trace_lbfgs

    log_summary(
        initial_value,
        results,
        objective_model,
        policy_spec,
        u_star,
        value_at_u_star,
        config.t_steps,
        config.n_samples,
        config.step_size,
    )
    if config.plot and traces:
        u_lbfgs = float(results["lbfgs"].u) if "lbfgs" in results else None
        u_star_plot = _u_star_for_plot(objective_model, u_star, u_lbfgs)
        plot_loss_curves(
            traces,
            config.plot_dir,
            u_star=u_star_plot,
        )
        plot_gradient_norms(traces, config.plot_dir)
        plot_objective_u_slice(
            x_samples,
            objective_model,
            traces,
            config.plot_dir,
            u_star=u_star_plot,
        )
        if policy_spec.theta.size >= 2:
            axis_indices = (0, 1)
            axis_labels = None
            theta_path_points = [theta_initial]
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
                (name, results[name]) for name in enabled_estimators if name in results
            ]
            theta_refs = [theta_initial]
            theta_points = [(theta_initial, "initial", "#636363", "o")]
            for name, result in ordered_results:
                theta_refs.append(result.theta)
                style = ESTIMATOR_STYLES[name]
                theta_points.append(
                    (
                        result.theta,
                        style["label"],
                        style["color"],
                        style["marker"],
                    )
                )
            plot_theta_objective_contours(
                x_samples,
                objective_model,
                policy_spec,
                theta_initial,
                config.plot_dir,
                axis_indices=axis_indices,
                axis_labels=axis_labels,
                theta_refs=theta_refs,
                theta_points=theta_points,
                traces=traces,
            )
    u_first = float(results["first_order"].u) if "first_order" in results else None
    u_zero = float(results["zeroth_order"].u) if "zeroth_order" in results else None
    u_lbfgs = float(results["lbfgs"].u) if "lbfgs" in results else None
    return initial_value, u_first, u_zero, u_lbfgs
