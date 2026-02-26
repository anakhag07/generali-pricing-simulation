"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

import time
from typing import Tuple

from data.fixed_objective import FixedRegressionObjective
from data.models import Customer, default_rng
from experiments.config import ExperimentConfig
from experiments.helpers import run_first_order, run_lbfgs_theta, run_zeroth_order
from experiments.logging import log_summary
from experiments.visualization import (
    plot_gradient_norms,
    plot_loss_curves,
    plot_objective_u_slice,
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
)
from optimization.policy import policy_u


def _u_star_for_plot(objective_model: object, u_lbfgs: float) -> float | None:
    if isinstance(objective_model, FixedRegressionObjective):
        return None
    return u_lbfgs


def run_experiment(config: ExperimentConfig) -> Tuple[float, float, float, float]:
    objective_model = config.objective_model
    policy_spec = config.policy_spec

    rng = default_rng(config.seed)
    customers = [Customer.sample(rng, state_dim=config.state_dim) for _ in range(config.n_samples)]
    x_samples = [customer.x for customer in customers]

    theta_initial = policy_spec.theta
    u_initials = [policy_u(theta_initial, x, kind=policy_spec.kind) for x in x_samples]
    initial_value = float(
        sum(objective_model.value(x, u) for x, u in zip(x_samples, u_initials)) / len(x_samples)
    )
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
    u_first_values = [policy_u(theta_first, x, kind=policy_spec.kind) for x in x_samples]
    u_zero_values = [policy_u(theta_zero, x, kind=policy_spec.kind) for x in x_samples]
    u_first = float(sum(u_first_values) / len(u_first_values))
    u_zero = float(sum(u_zero_values) / len(u_zero_values))
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
    value_first = float(
        sum(objective_model.value(x, u) for x, u in zip(x_samples, u_first_values)) / len(x_samples)
    )
    value_zero = float(
        sum(objective_model.value(x, u) for x, u in zip(x_samples, u_zero_values)) / len(x_samples)
    )

    log_summary(
        initial_value,
        u_first,
        value_first,
        u_zero,
        value_zero,
        u_lbfgs,
        value_lbfgs,
        time_first,
        time_zero,
        time_lbfgs,
        objective_model,
        policy_spec,
        theta_first,
        theta_zero,
        theta_lbfgs,
        config.t_steps,
        config.n_samples,
        config.step_size,
    )
    if config.plot:
        plot_loss_curves(
            trace_first,
            trace_zero,
            trace_lbfgs,
            config.plot_dir,
            u_star=_u_star_for_plot(objective_model, u_lbfgs),
        )
        plot_gradient_norms(trace_first, trace_zero, trace_lbfgs, config.plot_dir)
        plot_objective_u_slice(
            x_samples,
            objective_model,
            trace_first,
            trace_zero,
            trace_lbfgs,
            config.plot_dir,
            u_lbfgs=u_lbfgs,
        )
        if policy_spec.theta.size >= 2:
            axis_indices = (0, 1)
            axis_labels = None
            if policy_spec.theta.size > 2:
                theta_points = [theta_initial]
                if trace_first.theta_values:
                    theta_points.extend(trace_first.theta_values)
                if trace_zero.theta_values:
                    theta_points.extend(trace_zero.theta_values)
                if trace_lbfgs is not None and trace_lbfgs.theta_values:
                    theta_points.extend(trace_lbfgs.theta_values)
                axis_indices = select_theta_axes_max_variance(theta_points)
                axis_labels = (
                    f"theta[{axis_indices[0]}] (max-var axis)",
                    f"theta[{axis_indices[1]}] (max-var axis)",
                )
            plot_theta_objective_contours(
                x_samples,
                objective_model,
                policy_spec,
                theta_initial,
                config.plot_dir,
                axis_indices=axis_indices,
                axis_labels=axis_labels,
                theta_refs=[theta_initial, theta_first, theta_zero, theta_lbfgs],
                theta_points=[
                    (theta_initial, "initial", "#636363", "o"),
                    (theta_first, "first-order", "#1f77b4", "o"),
                    (theta_zero, "zeroth-order", "#ff7f0e", "o"),
                    (theta_lbfgs, "L-BFGS", "#2ca02c", "x"),
                ],
                trace_first=trace_first,
                trace_zero=trace_zero,
                trace_lbfgs=trace_lbfgs,
            )
    return initial_value, u_first, u_zero, u_lbfgs
