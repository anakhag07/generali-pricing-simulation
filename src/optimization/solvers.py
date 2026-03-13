"""Compatibility solver wrappers over the Optimization class."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from objective.base import ObjectiveModel, StateVector
from optimization.base import Optimization, TrueGradFn
from optimization.gradients import FirstOrderGradient, GaussSteinGradient, SPSAGradient
from optimization.steps import STEP_RULE_LBFGSB


def run_first_order_minimize(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective_model,
        policy_kind,
        x_samples,
        FirstOrderGradient(),
        algorithm=STEP_RULE_LBFGSB,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="first-order",
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


def run_gauss_stein_minimize(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective_model,
        policy_kind,
        x_samples,
        GaussSteinGradient(),
        algorithm=STEP_RULE_LBFGSB,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
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
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    optimizer = Optimization(
        objective_model,
        policy_kind,
        x_samples,
        SPSAGradient(),
        algorithm=STEP_RULE_LBFGSB,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="spsa",
        rng=rng,
        minimize_fn=minimize,
    )
    return optimizer.solve(theta_start)


__all__ = ["run_first_order_minimize", "run_gauss_stein_minimize", "run_spsa_minimize"]
