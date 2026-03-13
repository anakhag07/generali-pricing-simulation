"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from objective.base import StateVector, default_rng
from experiments.config import ExperimentConfig
from experiments.helpers import (
    resolve_true_grad_theta_fn,
    run_first_order,
    run_gauss_stein,
    run_spsa,
)
from experiments.reporters import StepReporter
from experiments.results import EstimatorResult, ExperimentResult


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    objective_raw = config.objective
    if objective_raw is None:
        raise ValueError("config.objective must be initialized.")
    objective = cast(Any, objective_raw)
    enabled_estimators = tuple(config.enabled_estimators)

    rng = default_rng(config.seed)
    x_samples = [StateVector.sample(rng, dim=config.state_dim) for _ in range(config.n_samples)]
    x_array = np.stack([np.asarray(x, dtype=float) for x in x_samples], axis=0).astype(float)
    true_grad_theta_fn = resolve_true_grad_theta_fn(objective, config.correctness)

    theta_initial = np.asarray(config.theta0, dtype=float)
    initial_value = float(objective.value(theta_initial, x_array))

    u_star = None
    optimal_u = getattr(objective, "optimal_u", None)
    if callable(optimal_u):
        u_star_candidate = optimal_u()
        if u_star_candidate is not None:
            u_star = float(u_star_candidate)

    value_at_u_star = None
    action_value = getattr(objective, "action_value", None)
    if u_star is not None and callable(action_value):
        value_at_u_star = float(np.mean([action_value(x, u_star) for x in x_samples]))

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
        mean_action = getattr(objective, "mean_action", None)
        if not callable(mean_action):
            raise ValueError("objective must provide mean_action(theta, x_batch).")
        u_first = float(mean_action(theta_first, x_array))
        value_first = float(objective.value(theta_first, x_array))
        results["first_order"] = EstimatorResult(theta=theta_first, u=u_first, value=value_first, time=time_first)
        traces["first_order"] = trace_first

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
        mean_action = getattr(objective, "mean_action", None)
        if not callable(mean_action):
            raise ValueError("objective must provide mean_action(theta, x_batch).")
        u_zero = float(mean_action(theta_zero, x_array))
        value_zero = float(objective.value(theta_zero, x_array))
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
        mean_action = getattr(objective, "mean_action", None)
        if not callable(mean_action):
            raise ValueError("objective must provide mean_action(theta, x_batch).")
        u_spsa = float(mean_action(theta_spsa, x_array))
        value_spsa = float(objective.value(theta_spsa, x_array))
        results["spsa"] = EstimatorResult(theta=theta_spsa, u=u_spsa, value=value_spsa, time=time_spsa)
        traces["spsa"] = trace_spsa

    return ExperimentResult(
        config=config,
        x_samples=x_samples,
        initial_value=initial_value,
        results=results,
        traces=traces,
        u_star=u_star,
        value_at_u_star=value_at_u_star,
    )
