"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from data.models import Customer, default_rng
from optimization.common import clip_u
from optimization.gradients.first_order import stein_first_order_grad
from optimization.gradients.zeroth_order import stein_zeroth_order_grad
from optimization.objective import objective, objective_with_oracle_grad

from experiments.logging import log_step, log_summary


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    previous_policy_price: float = 1200.0
    u0: float = 1.0
    t_steps: int = 10
    step_size: float = 0.05
    sigma: float = 0.1
    n_samples: int = 64


def run_demo(config: ExperimentConfig = ExperimentConfig()) -> Tuple[float, float, float]:
    rng = default_rng(config.seed)

    customer = Customer.sample(rng)

    def objective_fn(u: float) -> float:
        return objective(customer, u, config.previous_policy_price, rng)

    def oracle_grad_fn(u: float):
        return objective_with_oracle_grad(customer, u, config.previous_policy_price, rng)

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
            u = clip_u(u - config.step_size * grad)
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
            u = clip_u(u - config.step_size * grad)
            value = objective_fn(u)
            log_step("zeroth-order", step, u, value)
        return u

    value = objective_fn(config.u0)
    u_first = run_first_order(config.u0)
    u_zero = run_zeroth_order(config.u0)

    log_summary(value, u_first, u_zero)
    return value, u_first, u_zero
