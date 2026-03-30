"""Compatibility solver wrappers over the Optimization class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from objective.base import Objective
from optimization.base import Optimization, TrueThetaGradFn
from optimization.gradients import (
    FiniteDifferenceGradient,
    FirstOrderGradient,
    GaussSteinGradient,
    SPSAGradient,
    SteinDifferenceGradient,
)
from optimization.steps import STEP_RULE_LBFGSB

if TYPE_CHECKING:
    from experiments.reporters import StepReporter
    from experiments.results import OptimizationTrace


def run_first_order_minimize(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    algorithm: str = STEP_RULE_LBFGSB,
    step_size: float = 0.01,
    batch_size: int | None = None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective,
        x_samples,
        FirstOrderGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="first-order",
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


def run_finite_difference_minimize(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    algorithm: str = STEP_RULE_LBFGSB,
    step_size: float = 0.01,
    batch_size: int | None = None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective,
        x_samples,
        FiniteDifferenceGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="finite-difference",
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


def run_gauss_stein_minimize(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    algorithm: str = STEP_RULE_LBFGSB,
    step_size: float = 0.01,
    batch_size: int | None = None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective,
        x_samples,
        GaussSteinGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="gauss-stein",
        rng=rng,
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


def run_spsa_minimize(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    algorithm: str = STEP_RULE_LBFGSB,
    step_size: float = 0.01,
    batch_size: int | None = None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective,
        x_samples,
        SPSAGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="spsa",
        rng=rng,
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


def run_stein_difference_minimize(
    theta_start: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    algorithm: str = STEP_RULE_LBFGSB,
    step_size: float = 0.01,
    batch_size: int | None = None,
    true_grad_theta_fn: TrueThetaGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective,
        x_samples,
        SteinDifferenceGradient(),
        algorithm=algorithm,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        step_size=step_size,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="stein-difference",
        rng=rng,
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


__all__ = [
    "run_first_order_minimize",
    "run_finite_difference_minimize",
    "run_gauss_stein_minimize",
    "run_spsa_minimize",
    "run_stein_difference_minimize",
]
