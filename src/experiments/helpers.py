"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from data.models import ObjectiveModel, ObjectiveResult, StateVector
from experiments.logging import log_grad, log_step
from experiments.visualization import OptimizationTrace
from optimization.gradients.first_order import stein_first_order_grad
from optimization.gradients.zeroth_order import stein_zeroth_order_grad
from optimization.policy import policy_grad_theta, policy_u


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


def build_batch_objective_fns(
    objective_model: ObjectiveModel,
    x_samples: Sequence[StateVector],
) -> Tuple[ObjectiveFn, OracleGradFn, GradFn]:
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")

    def objective_fn(u: float) -> float:
        values = [objective_model.value(x, u) for x in x_list]
        return float(np.mean(values))

    def oracle_grad_fn(u: float) -> ObjectiveResult:
        evaluations = [objective_model.evaluate(x, u) for x in x_list]
        value = float(np.mean([result.value for result in evaluations]))
        grad = float(np.mean([result.grad_u for result in evaluations]))
        return ObjectiveResult(value=value, grad_u=grad)

    def grad_fn(u: float) -> float:
        grads = [objective_model.grad_u(x, u) for x in x_list]
        return float(np.mean(grads))

    return objective_fn, oracle_grad_fn, grad_fn


def run_first_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    log_steps: bool = True,
) -> tuple[np.ndarray, OptimizationTrace]:
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    theta = np.asarray(theta_start, dtype=float)
    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    grad_estimates: list[float] = []
    true_grads: list[float] = []
    for step in range(1, t_steps + 1):
        grad_values: list[float] = []
        true_grad_values: list[float] = []
        grad_theta = np.zeros_like(theta)
        for x in x_list:
            u = policy_u(theta, x, kind=policy_kind)
            grad = stein_first_order_grad(
                u,
                lambda u_val, x_val=x: objective_model.evaluate(x_val, u_val),
                rng,
                n_samples=n_grad_samples,
                sigma=sigma,
            )
            grad_values.append(grad)
            true_grad_values.append(objective_model.grad_u(x, u))
            grad_theta = grad_theta + grad * policy_grad_theta(theta, x, kind=policy_kind)
        grad_theta = grad_theta / float(len(x_list))
        theta = theta - step_size * grad_theta
        u_next_values = [policy_u(theta, x, kind=policy_kind) for x in x_list]
        value = float(
            np.mean([objective_model.value(x, u_next) for x, u_next in zip(x_list, u_next_values)])
        )
        mean_u = float(np.mean(u_next_values))
        mean_grad = float(np.mean(grad_values))
        mean_true_grad = float(np.mean(true_grad_values))
        if log_steps:
            log_grad("first-order", step, mean_grad)
            log_step("first-order", step, mean_u, value)
        steps.append(step)
        u_values.append(mean_u)
        values.append(value)
        grad_estimates.append(mean_grad)
        true_grads.append(mean_true_grad)
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
    )
    return theta, trace


def run_zeroth_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    log_steps: bool = True,
) -> tuple[np.ndarray, OptimizationTrace]:
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    theta = np.asarray(theta_start, dtype=float)
    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    grad_estimates: list[float] = []
    true_grads: list[float] = []
    for step in range(1, t_steps + 1):
        grad_values: list[float] = []
        true_grad_values: list[float] = []
        grad_theta = np.zeros_like(theta)
        for x in x_list:
            u = policy_u(theta, x, kind=policy_kind)
            grad = stein_zeroth_order_grad(
                u,
                lambda u_val, x_val=x: objective_model.value(x_val, u_val),
                rng,
                n_samples=n_grad_samples,
                sigma=sigma,
            )
            grad_values.append(grad)
            true_grad_values.append(objective_model.grad_u(x, u))
            grad_theta = grad_theta + grad * policy_grad_theta(theta, x, kind=policy_kind)
        grad_theta = grad_theta / float(len(x_list))
        theta = theta - step_size * grad_theta
        u_next_values = [policy_u(theta, x, kind=policy_kind) for x in x_list]
        value = float(
            np.mean([objective_model.value(x, u_next) for x, u_next in zip(x_list, u_next_values)])
        )
        mean_u = float(np.mean(u_next_values))
        mean_grad = float(np.mean(grad_values))
        mean_true_grad = float(np.mean(true_grad_values))
        if log_steps:
            log_grad("zeroth-order", step, mean_grad)
            log_step("zeroth-order", step, mean_u, value)
        steps.append(step)
        u_values.append(mean_u)
        values.append(value)
        grad_estimates.append(mean_grad)
        true_grads.append(mean_true_grad)
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
    )
    return theta, trace


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
