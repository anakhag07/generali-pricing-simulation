"""Logging helpers for experiment outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np

from objective.objectives import FixedRegressionObjective, PlantedLogisticObjective
from experiments.results import ExperimentResult


def log_step(
    method: str,
    step: int,
    u: float,
    value: float,
    grad_norm: float | None = None,
    step_size: float | None = None,
) -> None:
    """Print a single optimization step to console."""
    parts = [f"[{method}] step={step}", f"u={u:.4f}", f"value={value:.4f}"]
    if grad_norm is not None:
        parts.append(f"grad_norm={grad_norm:.4f}")
    if step_size is not None:
        parts.append(f"step_size={step_size:.6f}")
    print(" ".join(parts))


def log_summary(result: ExperimentResult) -> None:
    def format_array(values: object, precision: int = 3) -> str:
        arr = np.asarray(values, dtype=float)
        formatted = ", ".join(f"{val:.{precision}f}" for val in arr)
        return f"[{formatted}]"

    config = result.config
    objective = config.objective
    u_star: Optional[float] = result.u_star
    value_at_u_star: Optional[float] = result.value_at_u_star

    if isinstance(objective, FixedRegressionObjective):
        beta_1 = format_array(objective.beta_1)
        beta_2 = objective.beta_2
        beta_3 = format_array(objective.beta_3)
        beta_4 = objective.beta_4
        print(
            "Objective: f(u; x) = sigmoid(beta_1·x + beta_2*u) * (beta_3·x - beta_4*u)"
        )
        print(
            "Betas: "
            f"beta_1={beta_1}, beta_2={beta_2:.3f}, beta_3={beta_3}, beta_4={beta_4:.3f}"
        )
    elif isinstance(objective, PlantedLogisticObjective):
        beta = format_array(objective.beta)
        print("Objective: L(u; x) = log(1 + exp(z)) - p*(x) * z")
        print("z = alpha * u + beta·x + bias")
        print("p*(x) = sigmoid(alpha * u* + beta·x + bias)")
        print(
            "Params: "
            f"alpha={objective.alpha:.3f}, bias={objective.bias:.3f}, "
            f"u*={objective.u_star:.3f}, beta={beta}"
        )
    else:
        print(f"Objective: {type(objective).__name__}")

    print(
        "Run: "
        f"steps={config.t_steps}, n_samples={config.n_samples}, step_size={config.step_size:.4f}, "
        f"step_rule={config.step_rule}"
    )
    print(f"Initial objective value: {result.initial_value:.4f}")
    u_star_value = float(u_star) if u_star is not None else None
    value_at_u_star_value = (
        float(value_at_u_star) if value_at_u_star is not None else None
    )
    if u_star_value is not None:
        print(f"Known optimum u*: {u_star_value:.4f}")
        if value_at_u_star_value is not None:
            print(f"Objective at u*: {value_at_u_star_value:.4f}")
    print("=== Results ===")

    order = ("first_order", "gauss_stein", "spsa")
    labels = {
        "first_order": "first-order",
        "gauss_stein": "gauss-stein",
        "spsa": "SPSA",
    }
    ordered = [name for name in order if name in result.results]
    if ordered:
        final_u = ", ".join(
            f"{labels[name]}={float(result.results[name].u):.4f}" for name in ordered
        )
        final_value = ", ".join(
            f"{labels[name]}={float(result.results[name].value):.4f}" for name in ordered
        )
        print(f"Final u: {final_u}")
        print(f"Final objective: {final_value}")
        if u_star_value is not None:
            u_gap = ", ".join(
                f"{labels[name]}={abs(result.results[name].u - u_star_value):.4f}"
                for name in ordered
            )
            print(f"|u - u*|: {u_gap}")
            if value_at_u_star_value is not None:
                value_gap = ", ".join(
                    f"{labels[name]}={result.results[name].value - value_at_u_star_value:.4f}"
                    for name in ordered
                )
                print(f"Objective gap: {value_gap}")
        print(f"Initial theta: {format_array(config.theta0)}")
        for name in ordered:
            theta = result.results[name].theta
            theta_l2 = float(np.linalg.norm(theta))
            theta_delta_l2 = float(np.linalg.norm(theta - config.theta0))
            print(f"Final theta ({labels[name]}): {format_array(theta)}")
            print(
                f"Final theta norms ({labels[name]}): "
                f"||theta||_2={theta_l2:.4f}, ||theta-theta0||_2={theta_delta_l2:.4f}"
            )
        print("=== Runtime (s) ===")
        for name in ordered:
            runtime = float(result.results[name].time)
            print(f"{labels[name]}: {runtime:.4f}")
