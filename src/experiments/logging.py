"""Logging helpers for experiment outputs."""

from __future__ import annotations

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from optimization.policy import PolicySpec


def log_step(method: str, step: int, u: float, value: float) -> None:
    print(f"[{method}] step={step} u={u:.4f} value={value:.4f}")


def log_grad(method: str, step: int, grad: float) -> None:
    print(f"[{method}] step={step} grad={grad:.4f}")


def log_summary(
    initial_value: float,
    u_first: float,
    value_first: float,
    u_zero: float,
    value_zero: float,
    u_lbfgs: float,
    value_lbfgs: float,
    time_first: float,
    time_zero: float,
    time_lbfgs: float,
    objective_model: object,
    policy_spec: PolicySpec,
    theta_first: object,
    theta_zero: object,
    theta_lbfgs: object,
    t_steps: int,
    n_samples: int,
    step_size: float,
) -> None:
    def format_array(values: object, precision: int = 3) -> str:
        arr = np.asarray(values, dtype=float)
        formatted = ", ".join(f"{val:.{precision}f}" for val in arr)
        return f"[{formatted}]"

    if isinstance(objective_model, FixedRegressionObjective):
        beta_1 = format_array(objective_model.acceptance.beta_1)
        beta_2 = objective_model.acceptance.beta_2
        beta_3 = format_array(objective_model.loss.beta_3)
        beta_4 = objective_model.revenue.beta_4
        print(
            "Objective: f(u; x) = sigmoid(beta_1·x + beta_2*u) * (beta_3·x - beta_4*u)"
        )
        print(
            "Betas: "
            f"beta_1={beta_1}, beta_2={beta_2:.3f}, beta_3={beta_3}, beta_4={beta_4:.3f}"
        )
    else:
        print(f"Objective: {type(objective_model).__name__}")

    print(f"Run: steps={t_steps}, n_samples={n_samples}, step_size={step_size:.4f}")
    print(f"Initial objective value: {initial_value:.4f}")
    print("=== Results ===")
    print(
        "Final u: "
        f"first-order={u_first:.4f}, zeroth-order={u_zero:.4f}, L-BFGS={u_lbfgs:.4f}"
    )
    print(
        "Final objective: "
        f"first-order={value_first:.4f}, zeroth-order={value_zero:.4f}, L-BFGS={value_lbfgs:.4f}"
    )
    print(f"Initial theta: {format_array(policy_spec.theta)}")
    print(f"Final theta (first-order): {format_array(theta_first)}")
    print(f"Final theta (zeroth-order): {format_array(theta_zero)}")
    print(f"Final theta (L-BFGS): {format_array(theta_lbfgs)}")
    print("=== Runtime (s) ===")
    print(f"First-order: {time_first:.4f}")
    print(f"Zeroth-order: {time_zero:.4f}")
    print(f"L-BFGS: {time_lbfgs:.4f}")
