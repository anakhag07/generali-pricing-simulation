"""Logging helpers for experiment outputs."""

from __future__ import annotations

from typing import Optional

def log_step(method: str, step: int, u: float, value: float) -> None:
    print(f"[{method}] step={step} u={u:.4f} value={value:.4f}")


def log_grad(method: str, step: int, grad: float) -> None:
    print(f"[{method}] step={step} grad={grad:.4f}")


def log_summary(
    value: float,
    u_first: float,
    u_zero: float,
    u_star: Optional[float] = None,
    value_star: Optional[float] = None,
) -> None:
    print("Objective value:", value)
    if u_star is not None:
        print("Analytic minimizer u*:", u_star)
    if value_star is not None:
        print("Objective at u*:", value_star)
    print("Final u (first-order):", u_first)
    print("Final u (zeroth-order):", u_zero)
