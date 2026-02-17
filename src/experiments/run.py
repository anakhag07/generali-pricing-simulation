"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from data.models import Customer, default_rng
from experiments.config import ExperimentConfig
from experiments.logging import log_grad, log_step, log_summary
from experiments.visualization import (
    OptimizationTrace,
    plot_fixed_regression_truth,
    plot_gradient_norms,
    plot_loss_curves,
)
from optimization.gradients.first_order import stein_first_order_grad
from optimization.gradients.zeroth_order import stein_zeroth_order_grad
from optimization.policy import apply_policy


def run_experiment(config: Optional[ExperimentConfig] = None) -> Tuple[float, float, float, float]:
    if config is None:
        config = ExperimentConfig()

    objective_model = config.objective_model
    policy_spec = config.policy_spec
    if objective_model is None or policy_spec is None:
        raise ValueError("ExperimentConfig must define objective_model and policy_spec.")

    rng = default_rng(config.seed)
    customer = Customer.sample(rng, state_dim=config.state_dim)

    def objective_fn(u: float) -> float:
        return objective_model.value(customer.x, u)

    def oracle_grad_fn(u: float):
        return objective_model.evaluate(customer.x, u)

    def run_first_order(u_start: float) -> tuple[float, OptimizationTrace]:
        u = u_start
        steps: list[int] = []
        u_values: list[float] = []
        values: list[float] = []
        grad_estimates: list[float] = []
        true_grads: list[float] = []
        for step in range(1, config.t_steps + 1):
            grad = stein_first_order_grad(
                u,
                oracle_grad_fn,
                rng,
                n_samples=config.n_samples,
                sigma=config.sigma,
            )
            log_grad("first-order", step, grad)
            u = u - config.step_size * grad
            value = objective_fn(u)
            log_step("first-order", step, u, value)
            steps.append(step)
            u_values.append(u)
            values.append(value)
            grad_estimates.append(grad)
            true_grad = objective_model.grad_u(customer.x, u)
            true_grads.append(true_grad)
        trace = OptimizationTrace(
            steps=steps,
            u_values=u_values,
            objective_values=values,
            grad_estimates=grad_estimates,
            true_gradients=true_grads if true_grads else None,
        )
        return u, trace

    def run_zeroth_order(u_start: float) -> tuple[float, OptimizationTrace]:
        u = u_start
        steps: list[int] = []
        u_values: list[float] = []
        values: list[float] = []
        grad_estimates: list[float] = []
        true_grads: list[float] = []
        for step in range(1, config.t_steps + 1):
            grad = stein_zeroth_order_grad(
                u,
                objective_fn,
                rng,
                n_samples=config.n_samples,
                sigma=config.sigma,
            )
            log_grad("zeroth-order", step, grad)
            u = u - config.step_size * grad
            value = objective_fn(u)
            log_step("zeroth-order", step, u, value)
            steps.append(step)
            u_values.append(u)
            values.append(value)
            grad_estimates.append(grad)
            true_grad = objective_model.grad_u(customer.x, u)
            true_grads.append(true_grad)
        trace = OptimizationTrace(
            steps=steps,
            u_values=u_values,
            objective_values=values,
            grad_estimates=grad_estimates,
            true_gradients=true_grads if true_grads else None,
        )
        return u, trace

    def lbfgs_objective(u: float) -> float:
        return objective_model.value(customer.x, u)

    def run_lbfgs(u_start: float) -> tuple[float, float]:
        x0 = np.asarray([u_start], dtype=float)
        def value_fn(x: np.ndarray) -> float:
            return objective_model.value(customer.x, float(x[0]))

        def grad_fn(x: np.ndarray) -> np.ndarray:
            return np.asarray([objective_model.grad_u(customer.x, float(x[0]))], dtype=float)

        result = minimize(
            value_fn,
            x0=x0,
            jac=grad_fn,
            method="L-BFGS-B",
            options={"maxiter": config.lbfgs_maxiter},
        )
        u_lbfgs = float(result.x[0])
        value_lbfgs = lbfgs_objective(u_lbfgs)
        return u_lbfgs, value_lbfgs

    u0 = apply_policy(policy_spec, customer.x)
    initial_value = objective_fn(u0)
    u_first, trace_first = run_first_order(u0)
    u_zero, trace_zero = run_zeroth_order(u0)
    u_lbfgs, value_lbfgs = run_lbfgs(u0)
    value_first = objective_fn(u_first)
    value_zero = objective_fn(u_zero)

    log_summary(
        initial_value,
        u_first,
        value_first,
        u_zero,
        value_zero,
        u_lbfgs,
        value_lbfgs,
        objective_model,
    )
    if config.plot:
        plot_loss_curves(trace_first, trace_zero, config.plot_dir, u_star=u_lbfgs)
        plot_gradient_norms(trace_first, trace_zero, config.plot_dir)
        plot_fixed_regression_truth(
            customer.x,
            objective_model,
            trace_first,
            trace_zero,
            config.plot_dir,
            u_lbfgs=u_lbfgs,
        )
    return initial_value, u_first, u_zero, u_lbfgs
