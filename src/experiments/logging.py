"""Logging helpers for experiment outputs."""

from __future__ import annotations

from typing import Mapping, Optional, Protocol

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from optimization.policy import PolicySpec


class _EstimatorResult(Protocol):
    theta: object
    u: float
    value: float
    time: float


def log_step(method: str, step: int, u: float, value: float) -> None:
    print(f"[{method}] step={step} u={u:.4f} value={value:.4f}")


def log_grad(method: str, step: int, grad: float) -> None:
    print(f"[{method}] step={step} grad={grad:.4f}")


def log_summary(
    initial_value: float,
    results: Mapping[str, _EstimatorResult],
    objective_model: object,
    policy_spec: PolicySpec,
    u_star: Optional[float],
    value_at_u_star: Optional[float],
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
    u_star_value = float(u_star) if u_star is not None else None
    value_at_u_star_value = (
        float(value_at_u_star) if value_at_u_star is not None else None
    )
    if u_star_value is not None:
        print(f"Known optimum u*: {u_star_value:.4f}")
        if value_at_u_star_value is not None:
            print(f"Objective at u*: {value_at_u_star_value:.4f}")
    print("=== Results ===")

    order = ("first_order", "zeroth_order", "lbfgs")
    labels = {
        "first_order": "first-order",
        "zeroth_order": "zeroth-order",
        "lbfgs": "L-BFGS",
    }
    ordered = [name for name in order if name in results]
    if ordered:
        final_u = ", ".join(
            f"{labels[name]}={float(results[name].u):.4f}" for name in ordered
        )
        final_value = ", ".join(
            f"{labels[name]}={float(results[name].value):.4f}" for name in ordered
        )
        print(f"Final u: {final_u}")
        print(f"Final objective: {final_value}")
        if u_star_value is not None:
            u_gap = ", ".join(
                f"{labels[name]}={abs(results[name].u - u_star_value):.4f}" for name in ordered
            )
            print(f"|u - u*|: {u_gap}")
            if value_at_u_star_value is not None:
                value_gap = ", ".join(
                    f"{labels[name]}={results[name].value - value_at_u_star_value:.4f}"
                    for name in ordered
                )
                print(f"Objective gap: {value_gap}")
        print(f"Initial theta: {format_array(policy_spec.theta)}")
        for name in ordered:
            theta = results[name].theta
            print(f"Final theta ({labels[name]}): {format_array(theta)}")
        print("=== Runtime (s) ===")
        for name in ordered:
            runtime = float(results[name].time)
            print(f"{labels[name]}: {runtime:.4f}")
