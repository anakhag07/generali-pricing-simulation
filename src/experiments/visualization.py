"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.models import ObjectiveModel

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

    ax_loss.plot(
        trace_first.steps,
        trace_first.objective_values,
        label="first-order",
        alpha=0.5,
    )
    ax_loss.plot(
        trace_zero.steps,
        trace_zero.objective_values,
        label="zeroth-order",
        alpha=0.5,
    )
    ax_loss.set_ylabel("Objective value")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if ax_dist is not None and u_star is not None:
        dist_first = [abs(u - u_star) for u in trace_first.u_values]
        dist_zero = [abs(u - u_star) for u in trace_zero.u_values]
        ax_dist.plot(trace_first.steps, dist_first, label="first-order", alpha=0.5)
        ax_dist.plot(trace_zero.steps, dist_zero, label="zeroth-order", alpha=0.5)
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
    ax_norm.plot(trace_first.steps, norm_first, label="first-order", alpha=0.5)
    ax_norm.plot(trace_zero.steps, norm_zero, label="zeroth-order", alpha=0.5)
    ax_norm.set_ylabel("|estimated grad|")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)

    if ax_err is not None and trace_first.true_gradients is not None and trace_zero.true_gradients is not None:
        err_first = [abs(g - t) for g, t in zip(trace_first.grad_estimates, trace_first.true_gradients)]
        err_zero = [abs(g - t) for g, t in zip(trace_zero.grad_estimates, trace_zero.true_gradients)]
        ax_err.plot(trace_first.steps, err_first, label="first-order", alpha=0.5)
        ax_err.plot(trace_zero.steps, err_zero, label="zeroth-order", alpha=0.5)
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
    objective_model: ObjectiveModel,
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    plot_dir: str,
    u_lbfgs: Optional[float] = None,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    u_values = list(trace_first.u_values) + list(trace_zero.u_values)
    if u_lbfgs is not None:
        u_values.append(u_lbfgs)
    if u_values:
        u_min = float(min(u_values))
        u_max = float(max(u_values))
        if np.isclose(u_min, u_max):
            pad = 0.1 if u_min == 0.0 else abs(u_min) * 0.1
        else:
            pad = 0.1 * (u_max - u_min)
        u_grid = np.linspace(u_min - pad, u_max + pad, 200)
    else:
        u_grid = np.linspace(0.5, 1.5, 200)
    obj_grid = [objective_model.value(x, u) for u in u_grid]
    grad_grid = [objective_model.grad_u(x, u) for u in u_grid]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_obj, ax_grad = axes

    ax_obj.plot(u_grid, obj_grid, color="black", label="objective", alpha=0.5)
    ax_obj.scatter(
        trace_first.u_values,
        trace_first.objective_values,
        color="#1f77b4",
        label="first-order",
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax_obj.scatter(
        trace_zero.u_values,
        trace_zero.objective_values,
        color="#ff7f0e",
        label="zeroth-order",
        marker="s",
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
    )
    if u_lbfgs is not None:
        value_lbfgs = objective_model.value(x, u_lbfgs)
        ax_obj.scatter([u_lbfgs], [value_lbfgs], color="#2ca02c", marker="x", label="L-BFGS")
    ax_obj.set_ylabel("Objective value")
    ax_obj.legend()
    ax_obj.grid(True, alpha=0.3)

    ax_grad.plot(u_grid, grad_grid, color="black", label="true grad", alpha=0.5)
    ax_grad.scatter(
        trace_first.u_values,
        trace_first.grad_estimates,
        color="#1f77b4",
        label="first-order est",
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    ax_grad.scatter(
        trace_zero.u_values,
        trace_zero.grad_estimates,
        color="#ff7f0e",
        label="zeroth-order est",
        marker="s",
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
    )
    if u_lbfgs is not None:
        grad_lbfgs = objective_model.grad_u(x, u_lbfgs)
        ax_grad.scatter([u_lbfgs], [grad_lbfgs], color="#2ca02c", marker="x", label="L-BFGS")
    ax_grad.set_ylabel("Gradient")
    ax_grad.set_xlabel("u")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / "fixed_regression_truth.png", dpi=200)
    plt.close(fig)
