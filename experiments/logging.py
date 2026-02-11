"""Logging helpers for experiment outputs."""

from __future__ import annotations


def log_step(method: str, step: int, u: float, value: float) -> None:
    print(f"[{method}] step={step} u={u:.4f} value={value:.4f}")


def log_summary(value: float, u_first: float, u_zero: float) -> None:
    print("Objective value:", value)
    print("Final u (first-order):", u_first)
    print("Final u (zeroth-order):", u_zero)
