"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from data.models import ObjectiveModel, ObjectiveResult, StateVector
from experiments.logging import log_grad, log_step
from experiments.visualization import OptimizationTrace
from optimization.gradients.first_order import stein_first_order_grad
from optimization.gradients.zeroth_order import stein_zeroth_order_grad


ObjectiveFn = Callable[[float], float]
OracleGradFn = Callable[[float], ObjectiveResult]
GradFn = Callable[[float], float]


def build_objective_fns(
    objective_model: ObjectiveModel,
    x: StateVector,
) -> Tuple[ObjectiveFn, OracleGradFn, GradFn]:
    def objective_fn(u: float) -> float:
        return objective_model.value(x, u)

    def oracle_grad_fn(u: float) -> ObjectiveResult:
        return objective_model.evaluate(x, u)

    def grad_fn(u: float) -> float:
        return objective_model.grad_u(x, u)

    return objective_fn, oracle_grad_fn, grad_fn


def run_first_order(
    u_start: float,
    objective_fn: ObjectiveFn,
    oracle_grad_fn: OracleGradFn,
    true_grad_fn: Optional[GradFn],
    rng: np.random.Generator,
    t_steps: int,
    step_size: float,
    n_samples: int,
    sigma: float,
) -> tuple[float, OptimizationTrace]:
    u = u_start
    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    grad_estimates: list[float] = []
    true_grads: list[float] = []
    for step in range(1, t_steps + 1):
        grad = stein_first_order_grad(
            u,
            oracle_grad_fn,
            rng,
            n_samples=n_samples,
            sigma=sigma,
        )
        log_grad("first-order", step, grad)
        u = u - step_size * grad
        value = objective_fn(u)
        log_step("first-order", step, u, value)
        steps.append(step)
        u_values.append(u)
        values.append(value)
        grad_estimates.append(grad)
        if true_grad_fn is not None:
            true_grads.append(true_grad_fn(u))
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
    )
    return u, trace


def run_zeroth_order(
    u_start: float,
    objective_fn: ObjectiveFn,
    true_grad_fn: Optional[GradFn],
    rng: np.random.Generator,
    t_steps: int,
    step_size: float,
    n_samples: int,
    sigma: float,
) -> tuple[float, OptimizationTrace]:
    u = u_start
    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    grad_estimates: list[float] = []
    true_grads: list[float] = []
    for step in range(1, t_steps + 1):
        grad = stein_zeroth_order_grad(
            u,
            objective_fn,
            rng,
            n_samples=n_samples,
            sigma=sigma,
        )
        log_grad("zeroth-order", step, grad)
        u = u - step_size * grad
        value = objective_fn(u)
        log_step("zeroth-order", step, u, value)
        steps.append(step)
        u_values.append(u)
        values.append(value)
        grad_estimates.append(grad)
        if true_grad_fn is not None:
            true_grads.append(true_grad_fn(u))
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
    )
    return u, trace


def run_lbfgs(
    u_start: float,
    objective_fn: ObjectiveFn,
    grad_fn: GradFn,
    maxiter: int,
) -> tuple[float, float]:
    x0 = np.asarray([u_start], dtype=float)

    def value_fn(x: np.ndarray) -> float:
        return objective_fn(float(x[0]))

    def grad_fn_vec(x: np.ndarray) -> np.ndarray:
        return np.asarray([grad_fn(float(x[0]))], dtype=float)

    result = minimize(
        value_fn,
        x0=x0,
        jac=grad_fn_vec,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )
    u_lbfgs = float(result.x[0])
    value_lbfgs = objective_fn(u_lbfgs)
    return u_lbfgs, value_lbfgs
