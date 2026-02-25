"""Logging helpers for experiment outputs."""

from __future__ import annotations

from data.fixed_objective import FixedRegressionObjective
from optimization.policy import POLICY_SOFTMAX, PolicySpec


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
) -> None:
    print("Initial objective value:", initial_value)
    print("Final u (first-order):", u_first)
    print("Final objective (first-order):", value_first)
    print("Final u (zeroth-order):", u_zero)
    print("Final objective (zeroth-order):", value_zero)
    print("Final u (L-BFGS):", u_lbfgs)
    print("Final objective (L-BFGS):", value_lbfgs)
    print(f"Runtime (first-order): {time_first:.4f}s")
    print(f"Runtime (zeroth-order): {time_zero:.4f}s")
    print(f"Runtime (L-BFGS): {time_lbfgs:.4f}s")
    if isinstance(objective_model, FixedRegressionObjective):
        print("beta_1:", objective_model.acceptance.beta_1)
        print("beta_2:", objective_model.acceptance.beta_2)
        print("beta_3:", objective_model.loss.beta_3)
        print("beta_4:", objective_model.revenue.beta_4)
    else:
        print("objective_model:", type(objective_model).__name__)
    if policy_spec.kind == POLICY_SOFTMAX:
        print("Initial policy theta:", policy_spec.theta)
        print("Final policy theta (first-order):", theta_first)
        print("Final policy theta (zeroth-order):", theta_zero)
