"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from numpy.random import SeedSequence

from objective.base import default_rng, sample_states
from experiments.defaults import random_theta0
from objective.policy import ConstantPolicy
from objective.utils import (
    mean_acceptance_at_constant_u,
    _mean_action,
    _policy_value,
    optimal_u,
    value_at_constant_u,
    value_for_reporting,
)
from experiments.config import ExperimentConfig
from experiments.helpers import (
    resolve_true_grad_theta_fn,
    run_constant,
    run_finite_difference,
    run_first_order,
    run_gauss_stein,
    run_spsa,
    run_stein_difference,
)
from experiments.reporters import StepReporter
from experiments.results import ConstantBaselineResult, EstimatorResult, ExperimentResult, PolicyEvaluation


def _maybe_apply_acceptance_controls(config: ExperimentConfig) -> ExperimentConfig:
    """Inject config-level acceptance controls into model-based objectives."""
    if config.acceptance_floor is None and config.lagrangian_lambda is None:
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
        lagrangian_lambda=(
            float(config.lagrangian_lambda)
            if config.lagrangian_lambda is not None
            else None
        ),
    )
    return replace(config, objective=objective_with_floor)


def _constant_policy_objective(objective: object) -> object:
    """Return an objective copy with a one-parameter constant policy."""
    if not hasattr(objective, "policy"):
        raise ValueError("enabled estimator 'constant' requires an objective with a policy.")
    return replace(objective, policy=ConstantPolicy())


def _constant_theta_start(objective: object, theta_initial: np.ndarray, x_samples: np.ndarray) -> np.ndarray:
    """Initialize the constant baseline at the configured policy's mean action."""
    try:
        start_u = _mean_action(objective, theta_initial, x_samples)
    except Exception:  # noqa: BLE001 - fall back for objectives with unusual policy hooks.
        start_u = 0.0
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        start_u = float(np.asarray(clip_fn(np.asarray([start_u], dtype=float)), dtype=float)[0])
    return np.asarray([start_u], dtype=float)


def _row_count(x_samples: object) -> int:
    return int(x_samples.shape[0])


def _take_rows(x_samples: object, indices: np.ndarray) -> object:
    if indices.size == _row_count(x_samples) and np.array_equal(indices, np.arange(indices.size, dtype=int)):
        return x_samples
    if hasattr(x_samples, "iloc"):
        return x_samples.iloc[indices].reset_index(drop=True)
    return np.asarray(x_samples, dtype=float)[indices]


def _split_samples(config: ExperimentConfig, x_samples: object) -> tuple[object, object | None, np.ndarray, np.ndarray]:
    n_total = _row_count(x_samples)
    full_indices = np.arange(n_total, dtype=int)
    if config.test_fraction == 0.0:
        return x_samples, None, full_indices, np.asarray([], dtype=int)
    if n_total < 2:
        raise ValueError("train/test split requires at least two samples.")
    rng = np.random.default_rng(int(config.seed))
    shuffled = rng.permutation(n_total).astype(int)
    n_test = int(round(float(config.test_fraction) * n_total))
    n_test = min(max(n_test, 1), n_total - 1)
    test_indices = shuffled[:n_test]
    train_indices = shuffled[n_test:]
    return (
        _take_rows(x_samples, train_indices),
        _take_rows(x_samples, test_indices),
        train_indices,
        test_indices,
    )


def _source_row_indices(config: ExperimentConfig, indices: np.ndarray) -> np.ndarray | None:
    if config.x_fixed_row_indices is None:
        return None
    row_indices = np.asarray(config.x_fixed_row_indices, dtype=int)
    return row_indices[indices].copy()


def _policy_u_values(objective: object, theta: np.ndarray, x_samples: object) -> np.ndarray:
    u_values = np.asarray(_policy_value(objective, theta, x_samples), dtype=float).reshape(-1)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float).reshape(-1)
    if u_values.shape != (_row_count(x_samples),):
        raise ValueError("policy.value(theta, x_batch) must return one value per row.")
    return u_values


def _evaluate_final_policy(objective: object, theta: np.ndarray, x_samples: object) -> PolicyEvaluation:
    theta_arr = np.asarray(theta, dtype=float)
    n_samples = _row_count(x_samples)
    objective_value = value_for_reporting(objective, theta_arr, x_samples)
    u_values = _policy_u_values(objective, theta_arr, x_samples)
    mean_acceptance_fn = getattr(objective, "mean_acceptance", None)
    mean_acceptance = (
        float(mean_acceptance_fn(theta_arr, x_samples)) if callable(mean_acceptance_fn) else None
    )
    projected_loss = None
    projected_revenue = None
    step_metrics_fn = getattr(objective, "_step_metrics", None)
    if callable(step_metrics_fn):
        step_metrics = step_metrics_fn(theta_arr, x_samples)
        if "projected_loss" in step_metrics:
            projected_loss = float(step_metrics["projected_loss"])
        if "projected_revenue" in step_metrics:
            projected_revenue = float(step_metrics["projected_revenue"])
        if mean_acceptance is None and "mean_acceptance" in step_metrics:
            mean_acceptance = float(step_metrics["mean_acceptance"])
    q25, q75 = np.quantile(u_values, [0.25, 0.75])
    return PolicyEvaluation(
        n_samples=n_samples,
        objective_value=objective_value,
        objective_sum=n_samples * objective_value,
        mean_u=float(np.mean(u_values)),
        u_q25=float(q25),
        u_q75=float(q75),
        mean_acceptance=mean_acceptance,
        projected_loss=projected_loss,
        projected_revenue=projected_revenue,
    )


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    """Run optimization with all enabled estimators; returns traces and final values."""
    effective_config = _maybe_apply_acceptance_controls(config)
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
        policy_input_dim = getattr(objective, "policy_input_dim", None)
        theta_state_dim = (
            int(policy_input_dim()) if callable(policy_input_dim) else effective_config.state_dim
        )
        theta_initial = random_theta0(theta_state_dim, policy, theta0_rng)
        # Persist the resolved theta0 so reporters/plots can access it
        effective_config = replace(effective_config, theta0=theta_initial)
    else:
        rng = default_rng(effective_config.seed)
        theta_initial = np.asarray(effective_config.theta0, dtype=float)

    if effective_config.x_fixed is not None:
        if hasattr(effective_config.x_fixed, "iloc") and hasattr(effective_config.x_fixed, "columns"):
            x_all = effective_config.x_fixed.reset_index(drop=True).copy()
        else:
            x_all = np.asarray(effective_config.x_fixed, dtype=float)
    else:
        x_all = sample_states(rng, effective_config.n_samples, effective_config.state_dim)
    x_samples, x_test, train_indices, test_indices = _split_samples(effective_config, x_all)
    train_row_indices = _source_row_indices(effective_config, train_indices)
    test_row_indices = _source_row_indices(effective_config, test_indices)
    true_grad_theta_fn = resolve_true_grad_theta_fn(objective, effective_config.correctness)
    initial_value = value_for_reporting(objective, theta_initial, x_samples)
    mean_acceptance_fn = getattr(objective, "mean_acceptance", None)
    initial_mean_acceptance = (
        float(mean_acceptance_fn(theta_initial, x_samples)) if callable(mean_acceptance_fn) else None
    )

    constant_u_baselines = tuple(
        ConstantBaselineResult(
            u=float(u),
            value=value_at_constant_u(objective, x_samples, float(u)),
            mean_acceptance=mean_acceptance_at_constant_u(objective, x_samples, float(u)),
        )
        for u in effective_config.constant_u_baselines
    )

    # Get optimal u if available
    u_star = optimal_u(objective)

    # Compute value at u* if available
    value_at_u_star = None
    if u_star is not None:
        try:
            value_at_u_star = value_at_constant_u(objective, x_samples, u_star)
        except ValueError:
            pass

    # Get policy from objective for mean_action computation
    policy = getattr(objective, "policy", None)

    results: dict[str, EstimatorResult] = {}
    traces = {}

    if "constant" in enabled_estimators:
        constant_objective = _constant_policy_objective(objective)
        theta_constant_initial = _constant_theta_start(objective, theta_initial, x_samples)
        true_grad_constant_fn = resolve_true_grad_theta_fn(
            constant_objective,
            effective_config.correctness,
        )
        mean_acceptance_constant_fn = getattr(constant_objective, "mean_acceptance", None)
        start_constant = time.perf_counter()
        theta_constant, trace_constant = run_constant(
            theta_constant_initial,
            x_samples,
            constant_objective,
            rng,
            effective_config.t_steps,
            effective_config.step_rule,
            effective_config.step_size,
            effective_config.n_grad_samples,
            effective_config.sigma,
            effective_config.batch_size,
            perturbation_space=effective_config.perturbation_space,
            true_grad_theta_fn=true_grad_constant_fn,
            grad_norm_tol=effective_config.grad_norm_tol,
            ftol=effective_config.ftol,
            initial_constr_penalty=effective_config.initial_constr_penalty,
            step_reporter=step_reporter,
        )
        time_constant = time.perf_counter() - start_constant
        u_constant = _mean_action(constant_objective, theta_constant, x_samples)
        value_constant = value_for_reporting(constant_objective, theta_constant, x_samples)
        acceptance_constant = (
            float(mean_acceptance_constant_fn(theta_constant, x_samples))
            if callable(mean_acceptance_constant_fn)
            else None
        )
        results["constant"] = EstimatorResult(
            theta=theta_constant,
            u=u_constant,
            value=value_constant,
            time=time_constant,
            mean_acceptance=acceptance_constant,
            constraint_violation=trace_constant.constraint_violation,
            acceptance_multiplier=trace_constant.acceptance_multiplier,
            constraint_penalty=trace_constant.constraint_penalty,
        )
        traces["constant"] = trace_constant

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
        value_first = value_for_reporting(objective, theta_first, x_samples)
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
        value_fd = value_for_reporting(objective, theta_fd, x_samples)
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
        value_zero = value_for_reporting(objective, theta_zero, x_samples)
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
        value_spsa = value_for_reporting(objective, theta_spsa, x_samples)
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
        value_stein = value_for_reporting(objective, theta_stein, x_samples)
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

    train_metrics: dict[str, PolicyEvaluation] = {}
    test_metrics: dict[str, PolicyEvaluation] = {}
    for name, estimator_result in results.items():
        eval_objective = _constant_policy_objective(objective) if name == "constant" else objective
        train_metrics[name] = _evaluate_final_policy(eval_objective, estimator_result.theta, x_samples)
        if x_test is not None:
            test_metrics[name] = _evaluate_final_policy(eval_objective, estimator_result.theta, x_test)

    return ExperimentResult(
        config=effective_config,
        x_samples=x_samples,
        initial_value=initial_value,
        results=results,
        traces=traces,
        u_star=u_star,
        value_at_u_star=value_at_u_star,
        initial_mean_acceptance=initial_mean_acceptance,
        constant_u_baselines=constant_u_baselines,
        x_test=x_test,
        train_indices=train_indices,
        test_indices=test_indices,
        train_row_indices=train_row_indices,
        test_row_indices=test_row_indices,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )
