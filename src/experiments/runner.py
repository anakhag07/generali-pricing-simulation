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
from optimization.policy import PolicySpec, apply_policy, phi

from experiments.logging import log_grad, log_step, log_summary

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
    fixed_w: np.ndarray = field(
        default_factory=lambda: np.asarray([0.2, -0.05, 0.1, 0.3], dtype=float)
    )
    fixed_c: float = 1.1
    policy_spec: PolicySpec = field(
        default_factory=lambda: PolicySpec(theta=np.asarray([1.0], dtype=float))
    )


def run_demo(config: ExperimentConfig = ExperimentConfig()) -> Tuple[float, float, float]:
    rng = default_rng(config.seed)

    customer = Customer.sample(rng)

    if config.objective_kind not in OBJECTIVE_KINDS:
        raise ValueError(f"objective_kind must be one of {OBJECTIVE_KINDS}.")

    if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
        def objective_fn(u: float) -> float:
            return fixed_regression_objective(customer.x, u, config.fixed_w, config.fixed_c)

        def oracle_grad_fn(u: float):
            return fixed_regression_objective_with_grad(
                customer.x, u, config.fixed_w, config.fixed_c
            )
    else: # OBJECTIVE_STOCHASTIC
        def objective_fn(u: float) -> float:
            return objective(customer, u, config.previous_policy_price, rng)

        def oracle_grad_fn(u: float):
            return objective_with_oracle_grad(
                customer, u, config.previous_policy_price, rng
            )

    def run_first_order(u_start: float) -> float:
        u = u_start
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
        return u

    def run_zeroth_order(u_start: float) -> float:
        u = u_start
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
        return u

    u0 = apply_policy(config.policy_spec, customer.x)
    value = objective_fn(u0)
    u_first = run_first_order(u0)
    u_zero = run_zeroth_order(u0)

    u_star = None
    value_star = None
    print(f"Objective type is {config.objective_kind}")
    if config.objective_kind == OBJECTIVE_FIXED_REGRESSION:
        features = phi(customer.x)
        w_dot_phi = float(np.dot(config.fixed_w[: features.size], features))
        # u_star = clip_u(w_dot_phi / config.fixed_c)
        u_star = w_dot_phi / config.fixed_c    
        value_star = objective_fn(u_star)

    log_summary(value, u_first, u_zero, u_star=u_star, value_star=value_star)
    return value, u_first, u_zero
