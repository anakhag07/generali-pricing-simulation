"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from numpy.random import SeedSequence

from objective.base import default_rng, sample_states
from experiments.defaults import random_theta0
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


def _maybe_apply_acceptance_floor(config: ExperimentConfig) -> ExperimentConfig:
    """Inject a config-level acceptance floor into model-based objectives."""
    if config.acceptance_floor is None:
        return config
    objective = config.objective
    if not hasattr(objective, "acceptance_floor"):
        return config
    objective_with_floor = replace(
        objective,
        acceptance_floor=float(config.acceptance_floor),
        acceptance_penalty_weight=(
            float(config.acceptance_penalty_weight)
            if config.acceptance_penalty_weight is not None
            else None
        ),
        acceptance_penalty_temperature=float(config.acceptance_penalty_temperature),
    )
    return replace(config, objective=objective_with_floor)


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    """Run optimization with all enabled estimators; returns traces and final values."""
    effective_config = _maybe_apply_acceptance_floor(config)
    objective = effective_config.objective
    enabled_estimators = tuple(effective_config.enabled_estimators)

    # When theta0 is None (random init), split the seed so theta0 generation
    # doesn't alter the state-sampling RNG stream. Explicit theta0 preserves
    # the original RNG path for backward compatibility.
    if effective_config.theta0 is None:
        ss = SeedSequence(effective_config.seed)
        theta0_child, main_child = ss.spawn(2)
        theta0_rng = default_rng(theta0_child)
        rng = default_rng(main_child)
        policy = getattr(objective, "policy", None)
        theta_initial = random_theta0(effective_config.state_dim, policy, theta0_rng)
        # Persist the resolved theta0 so reporters/plots can access it
        effective_config = replace(effective_config, theta0=theta_initial)
    else:
        rng = default_rng(effective_config.seed)
        theta_initial = np.asarray(effective_config.theta0, dtype=float)

    if effective_config.x_fixed is not None:
        x_samples = np.asarray(effective_config.x_fixed, dtype=float)
    else:
        x_samples = sample_states(rng, effective_config.n_samples, effective_config.state_dim)
    true_grad_theta_fn = resolve_true_grad_theta_fn(objective, effective_config.correctness)
    initial_value = float(objective.value(theta_initial, x_samples))
    mean_acceptance_fn = getattr(objective, "mean_acceptance", None)
    initial_mean_acceptance = (
        float(mean_acceptance_fn(theta_initial, x_samples)) if callable(mean_acceptance_fn) else None
    )

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
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_first = time.perf_counter() - start_first
        u_first = _mean_action(objective, theta_first, x_samples) if policy is not None else float("nan")
        value_first = float(objective.value(theta_first, x_samples))
        acceptance_first = float(mean_acceptance_fn(theta_first, x_samples)) if callable(mean_acceptance_fn) else None
        results["first_order"] = EstimatorResult(
            theta=theta_first,
            u=u_first,
            value=value_first,
            time=time_first,
            mean_acceptance=acceptance_first,
            constraint_violation=trace_first.constraint_violation,
            acceptance_multiplier=trace_first.acceptance_multiplier,
            constraint_penalty=trace_first.constraint_penalty,
        )
        traces["first_order"] = trace_first

    if "finite_difference" in enabled_estimators:
        start_fd = time.perf_counter()
        theta_fd, trace_fd = run_finite_difference(
            theta_initial,
            x_samples,
            objective,
            rng,
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_fd = time.perf_counter() - start_fd
        u_fd = _mean_action(objective, theta_fd, x_samples) if policy is not None else float("nan")
        value_fd = float(objective.value(theta_fd, x_samples))
        acceptance_fd = float(mean_acceptance_fn(theta_fd, x_samples)) if callable(mean_acceptance_fn) else None
        results["finite_difference"] = EstimatorResult(
            theta=theta_fd,
            u=u_fd,
            value=value_fd,
            time=time_fd,
            mean_acceptance=acceptance_fd,
            constraint_violation=trace_fd.constraint_violation,
            acceptance_multiplier=trace_fd.acceptance_multiplier,
            constraint_penalty=trace_fd.constraint_penalty,
        )
        traces["finite_difference"] = trace_fd

    if "gauss_stein" in enabled_estimators:
        start_zero = time.perf_counter()
        theta_zero, trace_zero = run_gauss_stein(
            theta_initial,
            x_samples,
            objective,
            rng,
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_zero = time.perf_counter() - start_zero
        u_zero = _mean_action(objective, theta_zero, x_samples) if policy is not None else float("nan")
        value_zero = float(objective.value(theta_zero, x_samples))
        acceptance_zero = float(mean_acceptance_fn(theta_zero, x_samples)) if callable(mean_acceptance_fn) else None
        results["gauss_stein"] = EstimatorResult(
            theta=theta_zero,
            u=u_zero,
            value=value_zero,
            time=time_zero,
            mean_acceptance=acceptance_zero,
            constraint_violation=trace_zero.constraint_violation,
            acceptance_multiplier=trace_zero.acceptance_multiplier,
            constraint_penalty=trace_zero.constraint_penalty,
        )
        traces["gauss_stein"] = trace_zero

    if "spsa" in enabled_estimators:
        start_spsa = time.perf_counter()
        theta_spsa, trace_spsa = run_spsa(
            theta_initial,
            x_samples,
            objective,
            rng,
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_spsa = time.perf_counter() - start_spsa
        u_spsa = _mean_action(objective, theta_spsa, x_samples) if policy is not None else float("nan")
        value_spsa = float(objective.value(theta_spsa, x_samples))
        acceptance_spsa = float(mean_acceptance_fn(theta_spsa, x_samples)) if callable(mean_acceptance_fn) else None
        results["spsa"] = EstimatorResult(
            theta=theta_spsa,
            u=u_spsa,
            value=value_spsa,
            time=time_spsa,
            mean_acceptance=acceptance_spsa,
            constraint_violation=trace_spsa.constraint_violation,
            acceptance_multiplier=trace_spsa.acceptance_multiplier,
            constraint_penalty=trace_spsa.constraint_penalty,
        )
        traces["spsa"] = trace_spsa

    if "stein_difference" in enabled_estimators:
        start_stein = time.perf_counter()
        theta_stein, trace_stein = run_stein_difference(
            theta_initial,
            x_samples,
            objective,
            rng,
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_theta_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_stein = time.perf_counter() - start_stein
        u_stein = _mean_action(objective, theta_stein, x_samples) if policy is not None else float("nan")
        value_stein = float(objective.value(theta_stein, x_samples))
        acceptance_stein = float(mean_acceptance_fn(theta_stein, x_samples)) if callable(mean_acceptance_fn) else None
        results["stein_difference"] = EstimatorResult(
            theta=theta_stein,
            u=u_stein,
            value=value_stein,
            time=time_stein,
            mean_acceptance=acceptance_stein,
            constraint_violation=trace_stein.constraint_violation,
            acceptance_multiplier=trace_stein.acceptance_multiplier,
            constraint_penalty=trace_stein.constraint_penalty,
        )
        traces["stein_difference"] = trace_stein

    return ExperimentResult(
        config=effective_config,
        x_samples=x_samples,
        initial_value=initial_value,
        results=results,
        traces=traces,
        u_star=u_star,
        value_at_u_star=value_at_u_star,
        initial_mean_acceptance=initial_mean_acceptance,
    )
