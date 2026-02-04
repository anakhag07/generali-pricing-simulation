"""Run a small optimization demo."""

from __future__ import annotations

import numpy as np

from data import Customer, default_rng
from optimization.common import clip_u
from optimization.first_order.stein_grad import stein_first_order_grad
from optimization.objective import objective, objective_with_oracle_grad
from optimization.zeroth_order.stein_zo import stein_zeroth_order_grad


def main() -> None:
    seed = 7
    rng = default_rng(seed)

    previous_policy_price = 1200.0
    customer = Customer.sample(rng)
    u0 = 1.0
    t_steps = 10
    step_size = 0.05
    sigma = 0.1
    n_samples = 64

    def objective_fn(u: float) -> float:
        return objective(customer, u, previous_policy_price, rng)

    def oracle_grad_fn(u: float):
        return objective_with_oracle_grad(customer, u, previous_policy_price, rng)

    def run_first_order(u_start: float) -> float:
        u = u_start
        for step in range(1, t_steps + 1):
            grad = stein_first_order_grad(
                u,
                oracle_grad_fn,
                rng,
                n_samples=n_samples,
                sigma=sigma,
            )
            u = clip_u(u - step_size * grad)
            value = objective_fn(u)
            print(f"[first-order] step={step} u={u:.4f} value={value:.4f}")
        return u

    def run_zeroth_order(u_start: float) -> float:
        u = u_start
        for step in range(1, t_steps + 1):
            grad = stein_zeroth_order_grad(
                u,
                objective_fn,
                rng,
                n_samples=n_samples,
                sigma=sigma,
            )
            u = clip_u(u - step_size * grad)
            value = objective_fn(u)
            print(f"[zeroth-order] step={step} u={u:.4f} value={value:.4f}")
        return u

    value = objective_fn(u0)
    u_first = run_first_order(u0)
    u_zero = run_zeroth_order(u0)

    print("Objective value:", value)
    print("Final u (first-order):", u_first)
    print("Final u (zeroth-order):", u_zero)


if __name__ == "__main__":
    main()
