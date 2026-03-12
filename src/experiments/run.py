"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np

from objective.base import Customer, default_rng
from experiments.config import ExperimentConfig
from experiments.helpers import (
    resolve_true_grad_u_fn,
    run_first_order,
    run_gauss_stein,
    run_spsa,
)
from experiments.reporters import StepReporter
from experiments.results import EstimatorResult, ExperimentResult
from model.policy import policy_u


def _objective_u_star(objective_model: object) -> float | None:
    objective = cast(Any, objective_model)
    optimal_u = getattr(objective, "optimal_u", None)
    if callable(optimal_u):
        return float(cast(Any, optimal_u)())
    u_star = getattr(objective, "u_star", None)
    if u_star is not None:
        return float(cast(Any, u_star))
    return None


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    objective_model = config.objective_model
    policy_spec = config.policy_spec
    enabled_estimators = tuple(config.enabled_estimators)

    rng = default_rng(config.seed)
    customers = [Customer.sample(rng, state_dim=config.state_dim) for _ in range(config.n_samples)]
    x_samples = [customer.x for customer in customers]
    true_grad_u_fn = resolve_true_grad_u_fn(objective_model, config.correctness)

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
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_u_fn=true_grad_u_fn,
            grad_norm_tol=config.grad_norm_tol,
            step_reporter=step_reporter,
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

    if "gauss_stein" in enabled_estimators:
        start_zero = time.perf_counter()
        theta_zero, trace_zero = run_gauss_stein(
            theta_initial,
            policy_spec.kind,
            x_samples,
            objective_model,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_u_fn=true_grad_u_fn,
            grad_norm_tol=config.grad_norm_tol,
            step_reporter=step_reporter,
        )
        time_zero = time.perf_counter() - start_zero
        u_zero_values = [policy_u(theta_zero, x, kind=policy_spec.kind) for x in x_samples]
        u_zero = float(sum(u_zero_values) / len(u_zero_values))
        value_zero = float(
            sum(objective_model.value(x, u) for x, u in zip(x_samples, u_zero_values))
            / len(x_samples)
        )
        results["gauss_stein"] = EstimatorResult(
            theta=theta_zero,
            u=u_zero,
            value=value_zero,
            time=time_zero,
        )
        traces["gauss_stein"] = trace_zero

    if "spsa" in enabled_estimators:
        start_spsa = time.perf_counter()
        theta_spsa, trace_spsa = run_spsa(
            theta_initial,
            policy_spec.kind,
            x_samples,
            objective_model,
            rng,
            config.t_steps,
            config.step_rule,
            config.step_size,
            config.n_grad_samples,
            config.sigma,
            config.batch_size,
            true_grad_u_fn=true_grad_u_fn,
            grad_norm_tol=config.grad_norm_tol,
            step_reporter=step_reporter,
        )
        time_spsa = time.perf_counter() - start_spsa
        u_spsa_values = [policy_u(theta_spsa, x, kind=policy_spec.kind) for x in x_samples]
        u_spsa = float(sum(u_spsa_values) / len(u_spsa_values))
        value_spsa = float(
            sum(objective_model.value(x, u) for x, u in zip(x_samples, u_spsa_values))
            / len(x_samples)
        )
        results["spsa"] = EstimatorResult(
            theta=theta_spsa,
            u=u_spsa,
            value=value_spsa,
            time=time_spsa,
        )
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
