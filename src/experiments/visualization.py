"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from optimization.common import U_BOUNDS
from optimization.objective import fixed_regression_objective, fixed_regression_objective_with_grad

matplotlib.use("Agg")


@dataclass(frozen=True)
class OptimizationTrace:
    steps: Sequence[int]
    u_values: Sequence[float]
    objective_values: Sequence[float]
    grad_estimates: Sequence[float]
    true_gradients: Optional[Sequence[float]] = None


def _ensure_plot_dir(plot_dir: str) -> Path:
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curves(
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    plot_dir: str,
    u_star: Optional[float] = None,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    if u_star is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_loss, ax_dist = axes
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_dist = None

    ax_loss.plot(trace_first.steps, trace_first.objective_values, label="first-order")
    ax_loss.plot(trace_zero.steps, trace_zero.objective_values, label="zeroth-order")
    ax_loss.set_ylabel("Objective value")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if ax_dist is not None and u_star is not None:
        dist_first = [abs(u - u_star) for u in trace_first.u_values]
        dist_zero = [abs(u - u_star) for u in trace_zero.u_values]
        ax_dist.plot(trace_first.steps, dist_first, label="first-order")
        ax_dist.plot(trace_zero.steps, dist_zero, label="zeroth-order")
        ax_dist.set_ylabel("|u - u*|")
        ax_dist.set_xlabel("Step")
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
    else:
        ax_loss.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(path / "loss_curves.png", dpi=200)
    plt.close(fig)


def plot_gradient_norms(
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    plot_dir: str,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    has_true = trace_first.true_gradients is not None and trace_zero.true_gradients is not None

    if has_true:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_norm, ax_err = axes
    else:
        fig, ax_norm = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_err = None

    norm_first = [abs(g) for g in trace_first.grad_estimates]
    norm_zero = [abs(g) for g in trace_zero.grad_estimates]
    ax_norm.plot(trace_first.steps, norm_first, label="first-order")
    ax_norm.plot(trace_zero.steps, norm_zero, label="zeroth-order")
    ax_norm.set_ylabel("|estimated grad|")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)

    if ax_err is not None and trace_first.true_gradients is not None and trace_zero.true_gradients is not None:
        err_first = [abs(g - t) for g, t in zip(trace_first.grad_estimates, trace_first.true_gradients)]
        err_zero = [abs(g - t) for g, t in zip(trace_zero.grad_estimates, trace_zero.true_gradients)]
        ax_err.plot(trace_first.steps, err_first, label="first-order")
        ax_err.plot(trace_zero.steps, err_zero, label="zeroth-order")
        ax_err.set_ylabel("|grad error|")
        ax_err.set_xlabel("Step")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)
    else:
        ax_norm.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(path / "gradient_norms.png", dpi=200)
    plt.close(fig)


def plot_fixed_regression_truth(
    x,
    w: np.ndarray,
    c: float,
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    plot_dir: str,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    u_grid = np.linspace(U_BOUNDS[0], U_BOUNDS[1], 200)
    obj_grid = [fixed_regression_objective(x, u, w, c) for u in u_grid]
    grad_grid = [fixed_regression_objective_with_grad(x, u, w, c).grad_u for u in u_grid]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_obj, ax_grad = axes

    ax_obj.plot(u_grid, obj_grid, color="black", label="objective")
    ax_obj.scatter(trace_first.u_values, trace_first.objective_values, color="#1f77b4", label="first-order")
    ax_obj.scatter(trace_zero.u_values, trace_zero.objective_values, color="#ff7f0e", label="zeroth-order")
    ax_obj.set_ylabel("Objective value")
    ax_obj.legend()
    ax_obj.grid(True, alpha=0.3)

    ax_grad.plot(u_grid, grad_grid, color="black", label="true grad")
    ax_grad.scatter(trace_first.u_values, trace_first.grad_estimates, color="#1f77b4", label="first-order est")
    ax_grad.scatter(trace_zero.u_values, trace_zero.grad_estimates, color="#ff7f0e", label="zeroth-order est")
    ax_grad.set_ylabel("Gradient")
    ax_grad.set_xlabel("u")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / "fixed_regression_truth.png", dpi=200)
    plt.close(fig)
