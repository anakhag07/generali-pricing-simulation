"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np

from objective.base import Objective
from experiments.config import CorrectnessSpec
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from optimization.helpers import finite_difference_theta_grad
from optimization.solvers import (
    run_constant_minimize,
    run_finite_difference_minimize,
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_spsa_minimize,
    run_stein_difference_minimize,
)


TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

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
        return lambda theta, x_batch: finite_difference_theta_grad(
            lambda theta_eval: objective.value(theta_eval, x_batch),
            theta,
            method=correctness.numdiff_method,
            step=correctness.numdiff_step,
            bounds=correctness.numdiff_bounds,
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run first-order optimization."""
    return run_first_order_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
    )


def run_constant(
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run optimized constant-policy baseline."""
    return run_constant_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
    )


def run_finite_difference(
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run finite-difference value-query optimization."""
    return run_finite_difference_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run Gauss-Stein zeroth-order optimization."""
    return run_gauss_stein_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run SPSA zeroth-order optimization."""
    return run_spsa_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
    )


def run_stein_difference(
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
    perturbation_space: str = "theta",
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    initial_constr_penalty: float | None = None,
    step_reporter: StepReporter | None = None,
    gradient_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Run Stein-difference zeroth-order optimization."""
    return run_stein_difference_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        perturbation_space=perturbation_space,
        algorithm=step_rule,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        initial_constr_penalty=initial_constr_penalty,
        step_reporter=step_reporter,
        batch_rng=rng,
        gradient_rng=gradient_rng,
    )
