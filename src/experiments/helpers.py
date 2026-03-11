"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from objective.base import ObjectiveModel, StateVector
from experiments.config import CorrectnessSpec
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from optimization.solvers import (
    run_first_order_minimize,
    run_spsa_minimize,
    run_zeroth_order_minimize,
)


ObjectiveFn = Callable[[float], float]
TrueGradFn = Callable[[StateVector, float], float]


def _clamp_u(value: float, bounds: tuple[float, float] | None) -> float:
    """Clamp u to bounds when provided."""
    if bounds is None:
        return float(value)
    lower, upper = bounds
    return float(min(max(value, lower), upper))


def _numdiff_grad(
    value_fn: ObjectiveFn,
    u: float,
    method: str,
    step: float,
    bounds: tuple[float, float] | None,
) -> float:
    """Approximate d/du using finite differences around u."""
    u_base = float(u)
    if method == "central":
        u_plus = _clamp_u(u_base + step, bounds)
        u_minus = _clamp_u(u_base - step, bounds)
        denom = u_plus - u_minus
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_plus) - value_fn(u_minus)) / denom)
    if method == "forward":
        u_plus = _clamp_u(u_base + step, bounds)
        denom = u_plus - u_base
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_plus) - value_fn(u_base)) / denom)
    if method == "backward":
        u_minus = _clamp_u(u_base - step, bounds)
        denom = u_base - u_minus
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_base) - value_fn(u_minus)) / denom)
    raise ValueError(f"Unknown numdiff method '{method}'.")


def resolve_true_grad_u_fn(
    objective_model: ObjectiveModel,
    correctness: CorrectnessSpec,
) -> TrueGradFn | None:
    """Return a per-sample u-gradient proxy based on correctness settings."""
    if correctness.gradient_source == "none":
        return None
    if correctness.gradient_source == "exact":
        return lambda x, u: objective_model.grad_u(x, u)
    if correctness.gradient_source == "numdiff":
        if correctness.numdiff_aggregate != "per-sample":
            raise ValueError(
                "numdiff_aggregate='batch' is not supported for theta-gradient correctness."
            )

        def numdiff_grad(x: StateVector, u: float) -> float:
            def value_fn(u_val: float) -> float:
                return objective_model.value(x, u_val)

            return _numdiff_grad(
                value_fn,
                u,
                correctness.numdiff_method,
                correctness.numdiff_step,
                correctness.numdiff_bounds,
            )

        return numdiff_grad
    raise ValueError(f"Unknown gradient_source '{correctness.gradient_source}'.")


def run_first_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with exact u-gradients."""
    return run_first_order_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )


def run_zeroth_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with Stein u-gradient estimates."""
    return run_zeroth_order_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )


def run_spsa(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with SPSA theta-gradient estimates."""
    return run_spsa_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )
