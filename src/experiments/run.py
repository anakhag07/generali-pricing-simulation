"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import time

import numpy as np

from objective.base import sample_states
from objective.objectives import ModelBasedObjective, prepare_jax_glm_objective
from objective.policy import ConstantPolicy
from objective.utils import (
    mean_acceptance_at_constant_u,
    _mean_action,
    optimal_u,
    value_at_constant_u,
    value_for_reporting,
)
from experiments.config import ExperimentConfig
from experiments.correctness import TrueThetaGradFn, resolve_true_grad_theta_fn
from experiments.initialization import random_theta0
from experiments.policy_validation import evaluate_policy
from experiments.reporting.base import StepReporter
from experiments.results import (
    ConstantBaselineResult,
    EstimatorResult,
    ExperimentResult,
    OptimizationTrace,
    PolicyEvaluation,
)
from experiments.seeding import ResolvedSeedSetup, optimizer_rngs, resolve_seed_setup, rng_from_seed
from optimization.solvers import (
    run_constant_minimize,
    run_finite_difference_minimize,
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_spsa_minimize,
    run_stein_difference_minimize,
)


_SolverFn = Callable[..., tuple[np.ndarray, OptimizationTrace]]


@dataclass(frozen=True)
class _EstimatorSpec:
    solver: _SolverFn
    requires_rng_arg: bool = False


_ESTIMATOR_ORDER = (
    "constant",
    "first_order",
    "finite_difference",
    "gauss_stein",
    "spsa",
    "stein_difference",
)


_ESTIMATOR_SPECS = {
    "constant": _EstimatorSpec(run_constant_minimize),
    "first_order": _EstimatorSpec(run_first_order_minimize),
    "finite_difference": _EstimatorSpec(run_finite_difference_minimize),
    "gauss_stein": _EstimatorSpec(run_gauss_stein_minimize, requires_rng_arg=True),
    "spsa": _EstimatorSpec(run_spsa_minimize, requires_rng_arg=True),
    "stein_difference": _EstimatorSpec(run_stein_difference_minimize, requires_rng_arg=True),
}


_JAX_BACKEND_ESTIMATORS = {
    "first_order",
    "finite_difference",
    "gauss_stein",
    "spsa",
    "stein_difference",
}


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


def _maybe_apply_noise_seed(config: ExperimentConfig, noise_seed: int) -> ExperimentConfig:
    """Inject the resolved experiment noise seed into noisy objectives."""
    with_noise_seed = getattr(config.objective, "with_noise_seed", None)
    if not callable(with_noise_seed):
        return config
    return replace(config, objective=with_noise_seed(int(noise_seed)))


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


def _split_samples(
    config: ExperimentConfig,
    x_samples: object,
    split_seed: int,
) -> tuple[object, object | None, np.ndarray, np.ndarray]:
    n_total = _row_count(x_samples)
    full_indices = np.arange(n_total, dtype=int)
    if config.test_fraction == 0.0:
        return x_samples, None, full_indices, np.asarray([], dtype=int)
    if n_total < 2:
        raise ValueError("train/test split requires at least two samples.")
    rng = np.random.default_rng(int(split_seed))
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


def _optimizer_backend_objective(
    config: ExperimentConfig,
    objective: object,
    x_samples: object,
    theta_initial: np.ndarray,
    row_indices: np.ndarray | None,
) -> tuple[object, object]:
    """Return optimizer-facing objective/x samples for the configured compute backend."""
    if config.compute_backend == "numpy":
        return objective, x_samples
    if config.compute_backend != "jax":
        raise ValueError(f"Unsupported compute_backend '{config.compute_backend}'.")
    if config.step_rule != "trust-constr":
        raise ValueError("compute_backend='jax' is currently supported only with step_rule='trust-constr'.")
    if config.batch_size is not None:
        raise ValueError("compute_backend='jax' requires batch_size=None because it uses a fixed full batch.")
    unsupported = set(config.enabled_estimators) - _JAX_BACKEND_ESTIMATORS - {"constant"}
    if unsupported:
        supported = ", ".join(sorted(_JAX_BACKEND_ESTIMATORS | {"constant"}))
        unsupported_names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"compute_backend='jax' does not support enabled estimator(s): {unsupported_names}. "
            f"Supported estimators: {supported}."
        )
    if not isinstance(objective, ModelBasedObjective):
        raise ValueError("compute_backend='jax' currently supports GLM ModelBasedObjective runs only.")
    model_type = getattr(objective.acceptance_model, "model_type", None)
    if model_type != "glm":
        raise ValueError("compute_backend='jax' currently supports only GLM real-data artifacts.")
    jax_objective, batch = prepare_jax_glm_objective(
        objective,
        x_samples,
        row_indices=row_indices,
    )
    jax_objective.warmup(theta_initial)
    return jax_objective, batch.x_array


def _solve_estimator(
    *,
    estimator_name: str,
    spec: _EstimatorSpec,
    theta_start: np.ndarray,
    x_samples: object,
    objective: object,
    config: ExperimentConfig,
    seeds: ResolvedSeedSetup,
    true_grad_theta_fn: TrueThetaGradFn | None,
    step_reporter: StepReporter | None,
) -> tuple[np.ndarray, OptimizationTrace, float]:
    batch_rng, gradient_rng = optimizer_rngs(seeds, estimator_name)
    kwargs = {
        "theta_start": theta_start,
        "x_samples": x_samples,
        "objective": objective,
        "t_steps": config.t_steps,
        "n_grad_samples": config.n_grad_samples,
        "sigma": config.sigma,
        "perturbation_space": config.perturbation_space,
        "algorithm": config.step_rule,
        "step_size": config.step_size,
        "batch_size": config.batch_size,
        "true_grad_theta_fn": true_grad_theta_fn,
        "grad_norm_tol": config.grad_norm_tol,
        "ftol": config.ftol,
        "initial_constr_penalty": config.initial_constr_penalty,
        "step_reporter": step_reporter,
        "batch_rng": batch_rng,
        "gradient_rng": gradient_rng,
    }
    if spec.requires_rng_arg:
        kwargs["rng"] = gradient_rng
    start = time.perf_counter()
    theta_final, trace = spec.solver(**kwargs)
    return theta_final, trace, time.perf_counter() - start


def _estimator_result(
    objective: object,
    theta: np.ndarray,
    trace: OptimizationTrace,
    elapsed: float,
    x_samples: object,
) -> EstimatorResult:
    policy = getattr(objective, "policy", None)
    mean_acceptance_fn = getattr(objective, "mean_acceptance", None)
    return EstimatorResult(
        theta=theta,
        u=_mean_action(objective, theta, x_samples) if policy is not None else float("nan"),
        value=value_for_reporting(objective, theta, x_samples),
        time=elapsed,
        mean_acceptance=(
            float(mean_acceptance_fn(theta, x_samples)) if callable(mean_acceptance_fn) else None
        ),
        constraint_violation=trace.constraint_violation,
        acceptance_multiplier=trace.acceptance_multiplier,
        constraint_penalty=trace.constraint_penalty,
    )


def run_experiment(
    config: ExperimentConfig,
    step_reporter: StepReporter | None = None,
) -> ExperimentResult:
    """Run optimization with all enabled estimators; returns traces and final values."""
    effective_config = _maybe_apply_acceptance_controls(config)
    resolved_seeds = resolve_seed_setup(effective_config.seed_setup, effective_config.seed)
    effective_config = _maybe_apply_noise_seed(effective_config, resolved_seeds.noise_seed)
    objective = effective_config.objective
    enabled_estimators = tuple(effective_config.enabled_estimators)

    data_rng = rng_from_seed(resolved_seeds.data_seed)
    if effective_config.theta0 is None:
        theta0_rng = rng_from_seed(resolved_seeds.theta_seed)
        policy = getattr(objective, "policy", None)
        policy_input_dim = getattr(objective, "policy_input_dim", None)
        theta_state_dim = (
            int(policy_input_dim()) if callable(policy_input_dim) else effective_config.state_dim
        )
        theta_initial = random_theta0(theta_state_dim, policy, theta0_rng)
        # Persist the resolved theta0 so reporters/plots can access it
        effective_config = replace(effective_config, theta0=theta_initial)
    else:
        theta_initial = np.asarray(effective_config.theta0, dtype=float)

    if effective_config.x_fixed is not None:
        if hasattr(effective_config.x_fixed, "iloc") and hasattr(effective_config.x_fixed, "columns"):
            x_all = effective_config.x_fixed.reset_index(drop=True).copy()
        else:
            x_all = np.asarray(effective_config.x_fixed, dtype=float)
    else:
        x_all = sample_states(data_rng, effective_config.n_samples, effective_config.state_dim)
    x_samples, x_test, train_indices, test_indices = _split_samples(
        effective_config,
        x_all,
        resolved_seeds.split_seed,
    )
    train_row_indices = _source_row_indices(effective_config, train_indices)
    test_row_indices = _source_row_indices(effective_config, test_indices)
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

    optimizer_objective, optimizer_x_samples = _optimizer_backend_objective(
        effective_config,
        objective,
        x_samples,
        theta_initial,
        train_row_indices,
    )
    optimizer_true_grad_theta_fn = resolve_true_grad_theta_fn(
        optimizer_objective,
        effective_config.correctness,
    )

    results: dict[str, EstimatorResult] = {}
    traces: dict[str, OptimizationTrace] = {}

    for estimator_name in _ESTIMATOR_ORDER:
        if estimator_name not in enabled_estimators:
            continue
        spec = _ESTIMATOR_SPECS[estimator_name]
        if estimator_name == "constant":
            estimator_objective = _constant_policy_objective(objective)
            estimator_x_samples = x_samples
            estimator_theta_start = _constant_theta_start(objective, theta_initial, x_samples)
            true_grad_fn = resolve_true_grad_theta_fn(
                estimator_objective,
                effective_config.correctness,
            )
        else:
            estimator_objective = optimizer_objective
            estimator_x_samples = optimizer_x_samples
            estimator_theta_start = theta_initial
            true_grad_fn = optimizer_true_grad_theta_fn

        theta_final, trace, elapsed = _solve_estimator(
            estimator_name=estimator_name,
            spec=spec,
            theta_start=estimator_theta_start,
            x_samples=estimator_x_samples,
            objective=estimator_objective,
            config=effective_config,
            seeds=resolved_seeds,
            true_grad_theta_fn=true_grad_fn,
            step_reporter=step_reporter,
        )
        result_objective = _constant_policy_objective(objective) if estimator_name == "constant" else objective
        results[estimator_name] = _estimator_result(
            result_objective,
            theta_final,
            trace,
            elapsed,
            x_samples,
        )
        traces[estimator_name] = trace

    train_metrics: dict[str, PolicyEvaluation] = {}
    test_metrics: dict[str, PolicyEvaluation] = {}
    for name, estimator_result in results.items():
        eval_objective = _constant_policy_objective(objective) if name == "constant" else objective
        train_metrics[name] = evaluate_policy(eval_objective, estimator_result.theta, x_samples)
        if x_test is not None:
            test_metrics[name] = evaluate_policy(eval_objective, estimator_result.theta, x_test)

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
