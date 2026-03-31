"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

import time

import numpy as np

from objective.base import default_rng, sample_states
from objective.utils import _action_value_at_u, _mean_action, optimal_u
from experiments.config import ExperimentConfig
from experiments.helpers import (
    resolve_true_grad_theta_fn,
    run_finite_difference,
    run_first_order,
    run_gauss_stein,
    run_spsa,
    run_stein_difference,
)
from experiments.reporters import StepReporter
from experiments.results import EstimatorResult, ExperimentResult


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    """Run optimization with all enabled estimators; returns traces and final values."""
    objective = config.objective
    enabled_estimators = tuple(config.enabled_estimators)

    rng = default_rng(config.seed)
    x_samples = sample_states(rng, config.n_samples, config.state_dim)
    true_grad_theta_fn = resolve_true_grad_theta_fn(objective, config.correctness)

    theta_initial = np.asarray(config.theta0, dtype=float)
    initial_value = float(objective.value(theta_initial, x_samples))

    # Get optimal u if available
    u_star = optimal_u(objective)

    # Compute value at u* if available
    value_at_u_star = None
    if u_star is not None:
        try:
            value_at_u_star = _action_value_at_u(objective, x_samples, u_star)
        except ValueError:
            pass

    # Get policy from objective for mean_action computation
    policy = getattr(objective, "policy", None)

    results: dict[str, EstimatorResult] = {}
    traces = {}

    if "first_order" in enabled_estimators:
        start_first = time.perf_counter()
        theta_first, trace_first = run_first_order(
            theta_initial,
            x_samples,
            objective,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=config.grad_norm_tol,
            ftol=config.ftol,
            step_reporter=step_reporter,
        )
        time_first = time.perf_counter() - start_first
        u_first = _mean_action(policy, theta_first, x_samples) if policy is not None else float("nan")
        value_first = float(objective.value(theta_first, x_samples))
        results["first_order"] = EstimatorResult(theta=theta_first, u=u_first, value=value_first, time=time_first)
        traces["first_order"] = trace_first

    if "finite_difference" in enabled_estimators:
        start_fd = time.perf_counter()
        theta_fd, trace_fd = run_finite_difference(
            theta_initial,
            x_samples,
            objective,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=config.grad_norm_tol,
            ftol=config.ftol,
            step_reporter=step_reporter,
        )
        time_fd = time.perf_counter() - start_fd
        u_fd = _mean_action(policy, theta_fd, x_samples) if policy is not None else float("nan")
        value_fd = float(objective.value(theta_fd, x_samples))
        results["finite_difference"] = EstimatorResult(theta=theta_fd, u=u_fd, value=value_fd, time=time_fd)
        traces["finite_difference"] = trace_fd

    if "gauss_stein" in enabled_estimators:
        start_zero = time.perf_counter()
        theta_zero, trace_zero = run_gauss_stein(
            theta_initial,
            x_samples,
            objective,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=config.grad_norm_tol,
            ftol=config.ftol,
            step_reporter=step_reporter,
        )
        time_zero = time.perf_counter() - start_zero
        u_zero = _mean_action(policy, theta_zero, x_samples) if policy is not None else float("nan")
        value_zero = float(objective.value(theta_zero, x_samples))
        results["gauss_stein"] = EstimatorResult(theta=theta_zero, u=u_zero, value=value_zero, time=time_zero)
        traces["gauss_stein"] = trace_zero

    if "spsa" in enabled_estimators:
        start_spsa = time.perf_counter()
        theta_spsa, trace_spsa = run_spsa(
            theta_initial,
            x_samples,
            objective,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=config.grad_norm_tol,
            ftol=config.ftol,
            step_reporter=step_reporter,
        )
        time_spsa = time.perf_counter() - start_spsa
        u_spsa = _mean_action(policy, theta_spsa, x_samples) if policy is not None else float("nan")
        value_spsa = float(objective.value(theta_spsa, x_samples))
        results["spsa"] = EstimatorResult(theta=theta_spsa, u=u_spsa, value=value_spsa, time=time_spsa)
        traces["spsa"] = trace_spsa

    if "stein_difference" in enabled_estimators:
        start_stein = time.perf_counter()
        theta_stein, trace_stein = run_stein_difference(
            theta_initial,
            x_samples,
            objective,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=config.grad_norm_tol,
            ftol=config.ftol,
            step_reporter=step_reporter,
        )
        time_stein = time.perf_counter() - start_stein
        u_stein = _mean_action(policy, theta_stein, x_samples) if policy is not None else float("nan")
        value_stein = float(objective.value(theta_stein, x_samples))
        results["stein_difference"] = EstimatorResult(
            theta=theta_stein,
            u=u_stein,
            value=value_stein,
            time=time_stein,
        )
        traces["stein_difference"] = trace_stein

    return ExperimentResult(
        config=config,
        x_samples=x_samples,
        initial_value=initial_value,
        results=results,
        traces=traces,
        u_star=u_star,
        value_at_u_star=value_at_u_star,
    )
