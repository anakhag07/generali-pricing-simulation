"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np

from objective.base import Objective
from experiments.config import CorrectnessSpec
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from optimization.solvers import (
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_spsa_minimize,
)


TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _clamp_theta(theta: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    """Clamp theta to bounds if provided."""
    theta_arr = np.asarray(theta, dtype=float)
    if bounds is None:
        return theta_arr
    lower, upper = bounds
    return np.clip(theta_arr, lower, upper)


def _numdiff_theta_grad(
    objective: Objective,
    theta: np.ndarray,
    x_batch: np.ndarray,
    method: str,
    step: float,
    bounds: tuple[float, float] | None,
) -> np.ndarray:
    """Compute theta gradient via numerical differentiation."""
    theta_arr = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta_arr)
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown numdiff method '{method}'.")

    for idx in range(theta_arr.size):
        basis = np.zeros_like(theta_arr)
        basis[idx] = 1.0
        if method == "central":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
            denom = theta_plus[idx] - theta_minus[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (objective.value(theta_plus, x_batch) - objective.value(theta_minus, x_batch)) / denom
            continue
        if method == "forward":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            denom = theta_plus[idx] - theta_arr[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (objective.value(theta_plus, x_batch) - objective.value(theta_arr, x_batch)) / denom
            continue
        theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
        denom = theta_arr[idx] - theta_minus[idx]
        if denom == 0.0:
            grad[idx] = 0.0
        else:
            grad[idx] = (objective.value(theta_arr, x_batch) - objective.value(theta_minus, x_batch)) / denom
    return grad


def resolve_true_grad_theta_fn(
    objective: Objective,
    correctness: CorrectnessSpec,
) -> TrueThetaGradFn | None:
    """Return a theta-gradient proxy based on correctness settings."""
    if correctness.gradient_source == "none":
        return None
    if correctness.gradient_source == "exact":
        return lambda theta, x_batch: objective.grad(theta, x_batch)
    if correctness.gradient_source == "numdiff":
        return lambda theta, x_batch: _numdiff_theta_grad(
            objective,
            theta,
            x_batch,
            correctness.numdiff_method,
            correctness.numdiff_step,
            correctness.numdiff_bounds,
        )
    raise ValueError(f"Unknown gradient_source '{correctness.gradient_source}'.")


def run_first_order(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run first-order optimization."""
    del rng, step_rule, step_size  # Not used by first-order minimize
    return run_first_order_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )


def run_gauss_stein(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run Gauss-Stein zeroth-order optimization."""
    del step_rule, step_size  # Not used by Gauss-Stein minimize
    return run_gauss_stein_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )


def run_spsa(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run SPSA zeroth-order optimization."""
    del step_rule, step_size  # Not used by SPSA minimize
    return run_spsa_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )
