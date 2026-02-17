"""Logging helpers for experiment outputs."""

from __future__ import annotations

import numpy as np


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
    beta_1: np.ndarray,
    beta_2: float,
    beta_3: np.ndarray,
    beta_4: float,
) -> None:
    print("Initial objective value:", initial_value)
    print("Final u (first-order):", u_first)
    print("Final objective (first-order):", value_first)
    print("Final u (zeroth-order):", u_zero)
    print("Final objective (zeroth-order):", value_zero)
    print("Final u (L-BFGS):", u_lbfgs)
    print("Final objective (L-BFGS):", value_lbfgs)
    print("beta_1:", beta_1)
    print("beta_2:", beta_2)
    print("beta_3:", beta_3)
    print("beta_4:", beta_4)
