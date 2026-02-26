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
from optimization.steps import (
    STEP_RULE_ARMIJO,
    STEP_RULE_CONSTANT,
    armijo_backtracking_step_size,
    constant_step_size,
)


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
    step_rule: str,
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
    theta_grad_norms: list[float] = []
    true_theta_grad_norms: list[float] = []
    step_sizes: list[float] = []
    theta_values: list[np.ndarray] = [theta.copy()]

    def theta_objective(theta_vec: np.ndarray) -> float:
        theta_arr = np.asarray(theta_vec, dtype=float)
        u_vals = [policy_u(theta_arr, x, kind=policy_kind) for x in x_list]
        return float(
            np.mean([objective_model.value(x, u) for x, u in zip(x_list, u_vals)])
        )

    for step in range(1, t_steps + 1):
        grad_values: list[float] = []
        true_grad_values: list[float] = []
        grad_theta = np.zeros_like(theta)
        grad_theta_true = np.zeros_like(theta)
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
            true_grad = objective_model.grad_u(x, u)
            true_grad_values.append(true_grad)
            grad_theta = grad_theta + grad * policy_grad_theta(theta, x, kind=policy_kind)
            grad_theta_true = grad_theta_true + true_grad * policy_grad_theta(
                theta, x, kind=policy_kind
            )
        grad_theta = grad_theta / float(len(x_list))
        grad_theta_true = grad_theta_true / float(len(x_list))
        if step_rule == STEP_RULE_CONSTANT:
            step_now = constant_step_size(step_size)
        elif step_rule == STEP_RULE_ARMIJO:
            step_now = armijo_backtracking_step_size(
                theta,
                grad_theta,
                theta_objective,
                initial_step=step_size,
            )
        else:
            raise ValueError(f"Unknown step_rule: {step_rule}.")
        theta = theta - step_now * grad_theta
        theta_values.append(theta.copy())
        u_next_values = [policy_u(theta, x, kind=policy_kind) for x in x_list]
        value = float(
            np.mean([objective_model.value(x, u_next) for x, u_next in zip(x_list, u_next_values)])
        )
        mean_u = float(np.mean(u_next_values))
        mean_grad = float(np.mean(grad_values))
        mean_true_grad = float(np.mean(true_grad_values))
        theta_grad_norm = float(np.linalg.norm(grad_theta))
        true_theta_grad_norm = float(np.linalg.norm(grad_theta_true))
        if log_steps:
            log_grad("first-order", step, theta_grad_norm)
            log_step("first-order", step, mean_u, value)
        steps.append(step)
        u_values.append(mean_u)
        values.append(value)
        grad_estimates.append(mean_grad)
        true_grads.append(mean_true_grad)
        theta_grad_norms.append(theta_grad_norm)
        true_theta_grad_norms.append(true_theta_grad_norm)
        step_sizes.append(step_now)
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
        theta_grad_norms=theta_grad_norms,
        true_theta_grad_norms=true_theta_grad_norms,
        step_sizes=step_sizes,
        theta_values=theta_values,
    )
    return theta, trace


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
    theta_grad_norms: list[float] = []
    true_theta_grad_norms: list[float] = []
    step_sizes: list[float] = []
    theta_values: list[np.ndarray] = [theta.copy()]

    def theta_objective(theta_vec: np.ndarray) -> float:
        theta_arr = np.asarray(theta_vec, dtype=float)
        u_vals = [policy_u(theta_arr, x, kind=policy_kind) for x in x_list]
        return float(
            np.mean([objective_model.value(x, u) for x, u in zip(x_list, u_vals)])
        )

    for step in range(1, t_steps + 1):
        grad_values: list[float] = []
        true_grad_values: list[float] = []
        grad_theta = np.zeros_like(theta)
        grad_theta_true = np.zeros_like(theta)
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
            true_grad = objective_model.grad_u(x, u)
            true_grad_values.append(true_grad)
            grad_theta = grad_theta + grad * policy_grad_theta(theta, x, kind=policy_kind)
            grad_theta_true = grad_theta_true + true_grad * policy_grad_theta(
                theta, x, kind=policy_kind
            )
        grad_theta = grad_theta / float(len(x_list))
        grad_theta_true = grad_theta_true / float(len(x_list))
        if step_rule == STEP_RULE_CONSTANT:
            step_now = constant_step_size(step_size)
        elif step_rule == STEP_RULE_ARMIJO:
            step_now = armijo_backtracking_step_size(
                theta,
                grad_theta,
                theta_objective,
                initial_step=step_size,
            )
        else:
            raise ValueError(f"Unknown step_rule: {step_rule}.")
        theta = theta - step_now * grad_theta
        theta_values.append(theta.copy())
        u_next_values = [policy_u(theta, x, kind=policy_kind) for x in x_list]
        value = float(
            np.mean([objective_model.value(x, u_next) for x, u_next in zip(x_list, u_next_values)])
        )
        mean_u = float(np.mean(u_next_values))
        mean_grad = float(np.mean(grad_values))
        mean_true_grad = float(np.mean(true_grad_values))
        theta_grad_norm = float(np.linalg.norm(grad_theta))
        true_theta_grad_norm = float(np.linalg.norm(grad_theta_true))
        if log_steps:
            log_grad("zeroth-order", step, theta_grad_norm)
            log_step("zeroth-order", step, mean_u, value)
        steps.append(step)
        u_values.append(mean_u)
        values.append(value)
        grad_estimates.append(mean_grad)
        true_grads.append(mean_true_grad)
        theta_grad_norms.append(theta_grad_norm)
        true_theta_grad_norms.append(true_theta_grad_norm)
        step_sizes.append(step_now)
    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads if true_grads else None,
        theta_grad_norms=theta_grad_norms,
        true_theta_grad_norms=true_theta_grad_norms,
        step_sizes=step_sizes,
        theta_values=theta_values,
    )
    return theta, trace


def run_lbfgs_theta(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    maxiter: int,
) -> tuple[np.ndarray, float, OptimizationTrace]:
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    theta0 = np.asarray(theta_start, dtype=float)

    def value_fn(theta_vec: np.ndarray) -> float:
        theta_arr = np.asarray(theta_vec, dtype=float)
        values = [
            objective_model.value(x, policy_u(theta_arr, x, kind=policy_kind)) for x in x_list
        ]
        return float(np.mean(values))

    def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta_vec, dtype=float)
        grad = np.zeros_like(theta_arr)
        for x in x_list:
            u = policy_u(theta_arr, x, kind=policy_kind)
            grad_u = objective_model.grad_u(x, u)
            grad = grad + grad_u * policy_grad_theta(theta_arr, x, kind=policy_kind)
        grad = grad / float(len(x_list))
        return grad

    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    grad_estimates: list[float] = []
    true_grads: list[float] = []
    theta_grad_norms: list[float] = []
    true_theta_grad_norms: list[float] = []
    theta_values: list[np.ndarray] = []

    def record(theta_vec: np.ndarray) -> None:
        theta_arr = np.asarray(theta_vec, dtype=float)
        u_vals = [policy_u(theta_arr, x, kind=policy_kind) for x in x_list]
        mean_u = float(np.mean(u_vals))
        mean_value = float(
            np.mean([objective_model.value(x, u) for x, u in zip(x_list, u_vals)])
        )
        grad_u_vals = [objective_model.grad_u(x, u) for x, u in zip(x_list, u_vals)]
        mean_grad_u = float(np.mean(grad_u_vals))
        grad_theta = grad_fn(theta_arr)
        theta_grad_norm = float(np.linalg.norm(grad_theta))
        steps.append(len(steps))
        u_values.append(mean_u)
        values.append(mean_value)
        grad_estimates.append(mean_grad_u)
        true_grads.append(mean_grad_u)
        theta_grad_norms.append(theta_grad_norm)
        true_theta_grad_norms.append(theta_grad_norm)
        theta_values.append(theta_arr.copy())

    record(theta0)

    def callback(theta_vec: np.ndarray) -> None:
        record(theta_vec)

    result = minimize(
        value_fn,
        x0=theta0,
        jac=grad_fn,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
        callback=callback,
    )
    theta_lbfgs = np.asarray(result.x, dtype=float)
    value_lbfgs = value_fn(theta_lbfgs)
    if not np.allclose(theta_lbfgs, theta_values[-1]):
        record(theta_lbfgs)

    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        grad_estimates=grad_estimates,
        true_gradients=true_grads,
        theta_grad_norms=theta_grad_norms,
        true_theta_grad_norms=true_theta_grad_norms,
        theta_values=theta_values,
    )
    return theta_lbfgs, value_lbfgs, trace
