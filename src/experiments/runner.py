"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from data.models import Customer, default_rng
# from optimization.common import clip_u
from optimization.gradients.first_order import stein_first_order_grad
from optimization.gradients.zeroth_order import stein_zeroth_order_grad
from optimization.objective import (
    fixed_regression_objective,
    fixed_regression_objective_with_grad,
    objective,
    objective_with_oracle_grad,
)
from optimization.policy import POLICY_SOFTMAX, PolicySpec, apply_policy

from experiments.logging import log_grad, log_step, log_summary
from experiments.visualization import (
    OptimizationTrace,
    plot_fixed_regression_truth,
    plot_gradient_norms,
    plot_loss_curves,
)

OBJECTIVE_STOCHASTIC = "stochastic"
OBJECTIVE_FIXED_REGRESSION = "fixed_regression"
OBJECTIVE_KINDS = (OBJECTIVE_STOCHASTIC, OBJECTIVE_FIXED_REGRESSION)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    previous_policy_price: float = 1000.0
    t_steps: int = 100
    step_size: float = 0.01
    sigma: float = 0.1
    n_samples: int = 64
    objective_kind: str = OBJECTIVE_FIXED_REGRESSION
    beta_1: np.ndarray = field(
        default_factory=lambda: np.asarray([0.02, 0.2, 0.5], dtype=float)
    )
    beta_2: float = 0.8
    beta_3: np.ndarray = field(
        default_factory=lambda: np.asarray([0.005, 0.1, 0.2], dtype=float)
    )
    beta_4: float = 0.4
    policy_spec: PolicySpec = field(
        default_factory=lambda: PolicySpec(
            theta=np.asarray([0.1, 0.01, 0.01, 0.01], dtype=float),
            kind=POLICY_SOFTMAX,
        )
    )
    plot: bool = True
    plot_dir: str = "plots"

    def __post_init__(self) -> None:
        beta_1 = np.asarray(self.beta_1, dtype=float)
        beta_3 = np.asarray(self.beta_3, dtype=float)
        beta_2 = float(self.beta_2)
        beta_4 = float(self.beta_4)
        if beta_1.size < 3:
            raise ValueError("beta_1 must have at least 3 elements.")
        if beta_3.size < 3:
            raise ValueError("beta_3 must have at least 3 elements.")
        if np.any(beta_1 <= 0.0):
            raise ValueError("beta_1 entries must be positive.")
        if np.any(beta_3 <= 0.0):
            raise ValueError("beta_3 entries must be positive.")
        if beta_2 <= 0.0:
            raise ValueError("beta_2 must be positive.")
        if beta_4 <= 0.0:
            raise ValueError("beta_4 must be positive.")
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "beta_2", beta_2)
        object.__setattr__(self, "beta_3", beta_3)
        object.__setattr__(self, "beta_4", beta_4)


def run_demo(config: ExperimentConfig = ExperimentConfig()) -> Tuple[float, float, float]:
    rng = default_rng(config.seed)

    customer = Customer.sample(rng)

    if config.objective_kind not in OBJECTIVE_KINDS:
        raise ValueError(f"objective_kind must be one of {OBJECTIVE_KINDS}.")

    if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
        def objective_fn(u: float) -> float:
            return fixed_regression_objective(
                customer.x,
                u,
                config.beta_1,
                config.beta_2,
                config.beta_3,
                config.beta_4,
            )

        def oracle_grad_fn(u: float):
            return fixed_regression_objective_with_grad(
                customer.x,
                u,
                config.beta_1,
                config.beta_2,
                config.beta_3,
                config.beta_4,
            )
    else: # OBJECTIVE_STOCHASTIC
        def objective_fn(u: float) -> float:
            return objective(customer, u, config.previous_policy_price, rng)

        def oracle_grad_fn(u: float):
            return objective_with_oracle_grad(
                customer, u, config.previous_policy_price, rng
            )

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
            # u = clip_u(u - config.step_size * grad)
            u = u - config.step_size * grad
            value = objective_fn(u)
            log_step("first-order", step, u, value)
            steps.append(step)
            u_values.append(u)
            values.append(value)
            grad_estimates.append(grad)
            if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
                true_grad = fixed_regression_objective_with_grad(
                    customer.x,
                    u,
                    config.beta_1,
                    config.beta_2,
                    config.beta_3,
                    config.beta_4,
                ).grad_u
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
            # u = clip_u(u - config.step_size * grad)
            u = u - config.step_size * grad
            value = objective_fn(u)
            log_step("zeroth-order", step, u, value)
            steps.append(step)
            u_values.append(u)
            values.append(value)
            grad_estimates.append(grad)
            if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
                true_grad = fixed_regression_objective_with_grad(
                    customer.x,
                    u,
                    config.beta_1,
                    config.beta_2,
                    config.beta_3,
                    config.beta_4,
                ).grad_u
                true_grads.append(true_grad)
        trace = OptimizationTrace(
            steps=steps,
            u_values=u_values,
            objective_values=values,
            grad_estimates=grad_estimates,
            true_gradients=true_grads if true_grads else None,
        )
        return u, trace

    u0 = apply_policy(config.policy_spec, customer.x)
    value = objective_fn(u0)
    u_first, trace_first = run_first_order(u0)
    u_zero, trace_zero = run_zeroth_order(u0)

    u_star = None
    print(f"Objective type is {config.objective_kind}")
    log_summary(value, u_first, u_zero)
    if config.plot:
        plot_loss_curves(trace_first, trace_zero, config.plot_dir, u_star=u_star)
        plot_gradient_norms(trace_first, trace_zero, config.plot_dir)
        if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
            plot_fixed_regression_truth(
                customer.x,
                config.beta_1,
                config.beta_2,
                config.beta_3,
                config.beta_4,
                trace_first,
                trace_zero,
                config.plot_dir,
            )
    return value, u_first, u_zero
