"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.models import ObjectiveModel, StateVector
from optimization.policy import PolicySpec, policy_u

matplotlib.use("Agg")


@dataclass(frozen=True)
class OptimizationTrace:
    steps: Sequence[int]
    u_values: Sequence[float]
    objective_values: Sequence[float]
    grad_estimates: Sequence[float]
    true_gradients: Optional[Sequence[float]] = None
    theta_grad_norms: Optional[Sequence[float]] = None
    true_theta_grad_norms: Optional[Sequence[float]] = None
    theta_values: Optional[Sequence[np.ndarray]] = None


def _ensure_plot_dir(plot_dir: str) -> Path:
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_state_list(x_samples: Sequence[StateVector]) -> list[StateVector]:
    if isinstance(x_samples, StateVector):
        return [x_samples]
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    return x_list


def plot_loss_curves(
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    trace_lbfgs: Optional[OptimizationTrace],
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
        alpha=0.6,
    )
    ax_loss.plot(
        trace_zero.steps,
        trace_zero.objective_values,
        label="zeroth-order",
        alpha=0.6,
    )
    if trace_lbfgs is not None:
        ax_loss.plot(
            trace_lbfgs.steps,
            trace_lbfgs.objective_values,
            label="L-BFGS",
            color="#2ca02c",
            alpha=0.6,
        )
    ax_loss.set_ylabel("Objective value")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if ax_dist is not None and u_star is not None:
        dist_first = [abs(u - u_star) for u in trace_first.u_values]
        dist_zero = [abs(u - u_star) for u in trace_zero.u_values]
        ax_dist.plot(trace_first.steps, dist_first, label="first-order", alpha=0.6)
        ax_dist.plot(trace_zero.steps, dist_zero, label="zeroth-order", alpha=0.6)
        if trace_lbfgs is not None:
            dist_lbfgs = [abs(u - u_star) for u in trace_lbfgs.u_values]
            ax_dist.plot(
                trace_lbfgs.steps,
                dist_lbfgs,
                label="L-BFGS",
                color="#2ca02c",
                alpha=0.6,
            )
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
    trace_lbfgs: Optional[OptimizationTrace],
    plot_dir: str,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    has_true = (
        trace_first.true_theta_grad_norms is not None
        and trace_zero.true_theta_grad_norms is not None
    )

    if has_true:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_norm, ax_err = axes
    else:
        fig, ax_norm = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_err = None

    if trace_first.theta_grad_norms is None or trace_zero.theta_grad_norms is None:
        raise ValueError("theta_grad_norms must be provided for gradient norm plots.")
    norm_first = list(trace_first.theta_grad_norms)
    norm_zero = list(trace_zero.theta_grad_norms)
    ax_norm.plot(trace_first.steps, norm_first, label="first-order", alpha=0.6)
    ax_norm.plot(trace_zero.steps, norm_zero, label="zeroth-order", alpha=0.6)
    if trace_lbfgs is not None:
        if trace_lbfgs.theta_grad_norms is None:
            raise ValueError("theta_grad_norms must be provided for L-BFGS traces.")
        ax_norm.plot(
            trace_lbfgs.steps,
            trace_lbfgs.theta_grad_norms,
            label="L-BFGS",
            color="#2ca02c",
            alpha=0.6,
        )
    ax_norm.set_ylabel("|theta grad norm|")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)

    if ax_err is not None and trace_first.true_theta_grad_norms is not None and trace_zero.true_theta_grad_norms is not None:
        err_first = [
            abs(g - t)
            for g, t in zip(trace_first.theta_grad_norms, trace_first.true_theta_grad_norms)
        ]
        err_zero = [
            abs(g - t)
            for g, t in zip(trace_zero.theta_grad_norms, trace_zero.true_theta_grad_norms)
        ]
        ax_err.plot(trace_first.steps, err_first, label="first-order", alpha=0.6)
        ax_err.plot(trace_zero.steps, err_zero, label="zeroth-order", alpha=0.6)
        ax_err.set_ylabel("|grad error|")
        ax_err.set_xlabel("Step")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)
    else:
        ax_norm.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(path / "gradient_norms.png", dpi=200)
    plt.close(fig)


def plot_objective_u_slice(
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    trace_first: OptimizationTrace,
    trace_zero: OptimizationTrace,
    trace_lbfgs: Optional[OptimizationTrace],
    plot_dir: str,
    u_lbfgs: Optional[float] = None,
) -> None:
    path = _ensure_plot_dir(plot_dir)
    x_list = _as_state_list(x_samples)
    u_values = list(trace_first.u_values) + list(trace_zero.u_values)
    if trace_lbfgs is not None:
        u_values += list(trace_lbfgs.u_values)
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
    obj_grid = [float(np.mean([objective_model.value(x, u) for x in x_list])) for u in u_grid]
    grad_grid = [float(np.mean([objective_model.grad_u(x, u) for x in x_list])) for u in u_grid]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_obj, ax_grad = axes

    ax_obj.plot(u_grid, obj_grid, color="black", label="objective", alpha=0.6)
    ax_obj.scatter(
        trace_first.u_values,
        trace_first.objective_values,
        color="#1f77b4",
        label="first-order",
        marker="o",
        edgecolors="#1f77b4",
        linewidths=0.4,
        alpha=0.6,
        zorder=3,
    )
    ax_obj.scatter(
        trace_zero.u_values,
        trace_zero.objective_values,
        color="#ff7f0e",
        label="zeroth-order",
        marker="o",
        edgecolors="#ff7f0e",
        linewidths=0.4,
        alpha=0.6,
        zorder=4,
    )
    if trace_lbfgs is not None:
        ax_obj.plot(
            trace_lbfgs.u_values,
            trace_lbfgs.objective_values,
            color="#2ca02c",
            label="L-BFGS path",
            alpha=0.6,
        )
    if trace_lbfgs is not None and trace_lbfgs.u_values and trace_lbfgs.objective_values:
        ax_obj.scatter(
            [trace_lbfgs.u_values[-1]],
            [trace_lbfgs.objective_values[-1]],
            color="#2ca02c",
            marker="x",
            label="L-BFGS final",
        )
    elif u_lbfgs is not None:
        value_lbfgs = float(np.mean([objective_model.value(x, u_lbfgs) for x in x_list]))
        ax_obj.scatter(
            [u_lbfgs],
            [value_lbfgs],
            color="#2ca02c",
            marker="x",
            label="L-BFGS final",
        )
    ax_obj.set_ylabel("Objective value")
    ax_obj.legend()
    ax_obj.grid(True, alpha=0.3)

    ax_grad.plot(u_grid, grad_grid, color="black", label="true grad", alpha=0.6)
    ax_grad.scatter(
        trace_first.u_values,
        trace_first.grad_estimates,
        color="#1f77b4",
        label="first-order est",
        marker="o",
        edgecolors="#1f77b4",
        linewidths=0.4,
        alpha=0.6,
        zorder=3,
    )
    ax_grad.scatter(
        trace_zero.u_values,
        trace_zero.grad_estimates,
        color="#ff7f0e",
        label="zeroth-order est",
        marker="o",
        edgecolors="#ff7f0e",
        linewidths=0.4,
        alpha=0.6,
        zorder=4,
    )
    if trace_lbfgs is not None:
        ax_grad.plot(
            trace_lbfgs.u_values,
            trace_lbfgs.grad_estimates,
            color="#2ca02c",
            label="L-BFGS path",
            alpha=0.6,
        )
    if trace_lbfgs is not None and trace_lbfgs.u_values and trace_lbfgs.grad_estimates:
        ax_grad.scatter(
            [trace_lbfgs.u_values[-1]],
            [trace_lbfgs.grad_estimates[-1]],
            color="#2ca02c",
            marker="x",
            label="L-BFGS final",
        )
    elif u_lbfgs is not None:
        grad_lbfgs = float(np.mean([objective_model.grad_u(x, u_lbfgs) for x in x_list]))
        ax_grad.scatter(
            [u_lbfgs],
            [grad_lbfgs],
            color="#2ca02c",
            marker="x",
            label="L-BFGS final",
        )
    ax_grad.set_ylabel("Gradient")
    ax_grad.set_xlabel("u")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / "objective_u_slice.png", dpi=200)
    plt.close(fig)


def _theta_axis_grid(
    theta_base: np.ndarray,
    axis_index: int,
    theta_refs: Optional[Sequence[np.ndarray]],
    grid_size: int,
    pad_ratio: float,
    min_pad: float,
) -> np.ndarray:
    values = [float(np.asarray(theta_base, dtype=float)[axis_index])]
    if theta_refs is not None:
        for theta in theta_refs:
            theta_arr = np.asarray(theta, dtype=float)
            if axis_index >= theta_arr.size:
                raise ValueError("theta_refs must include values for each axis index.")
            values.append(float(theta_arr[axis_index]))
    min_val = min(values)
    max_val = max(values)
    if np.isclose(min_val, max_val):
        center = float(values[0])
        pad = max(min_pad, abs(center) * pad_ratio)
    else:
        pad = max(min_pad, (max_val - min_val) * pad_ratio)
    return np.linspace(min_val - pad, max_val + pad, grid_size)


def select_theta_axes_max_variance(theta_points: Sequence[np.ndarray]) -> tuple[int, int]:
    if not theta_points:
        raise ValueError("theta_points must contain at least one theta array.")
    theta_stack = np.asarray([np.asarray(theta, dtype=float) for theta in theta_points])
    if theta_stack.ndim != 2:
        raise ValueError("theta_points must be a sequence of 1D arrays with matching sizes.")
    if theta_stack.shape[1] < 2:
        raise ValueError("theta_points must have at least two dimensions.")
    variances = np.var(theta_stack, axis=0)
    top_two = np.argsort(variances)[-2:]
    ordered = top_two[np.argsort(variances[top_two])[::-1]]
    return int(ordered[0]), int(ordered[1])


def theta_objective_contour_grid(
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    policy_spec: PolicySpec,
    theta_base: np.ndarray,
    axis_indices: tuple[int, int] = (0, 1),
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    grid_size: int = 60,
    pad_ratio: float = 0.2,
    min_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list = _as_state_list(x_samples)
    theta_arr = np.asarray(theta_base, dtype=float)
    if len(axis_indices) != 2:
        raise ValueError("axis_indices must contain exactly two indices.")
    if axis_indices[0] == axis_indices[1]:
        raise ValueError("axis_indices must refer to two distinct components.")
    if any(index < 0 or index >= theta_arr.size for index in axis_indices):
        raise ValueError("axis_indices must be valid indices for theta.")
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")

    theta_x = _theta_axis_grid(theta_arr, axis_indices[0], theta_refs, grid_size, pad_ratio, min_pad)
    theta_y = _theta_axis_grid(theta_arr, axis_indices[1], theta_refs, grid_size, pad_ratio, min_pad)
    grid_x, grid_y = np.meshgrid(theta_x, theta_y)
    objective_grid = np.zeros_like(grid_x, dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            theta = theta_arr.copy()
            theta[axis_indices[0]] = grid_x[i, j]
            theta[axis_indices[1]] = grid_y[i, j]
            objective_values = [
                objective_model.value(x, policy_u(theta, x, kind=policy_spec.kind)) for x in x_list
            ]
            objective_grid[i, j] = float(np.mean(objective_values))

    return grid_x, grid_y, objective_grid


def plot_theta_objective_contours(
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    policy_spec: PolicySpec,
    theta_base: np.ndarray,
    plot_dir: str,
    axis_indices: tuple[int, int] = (0, 1),
    axis_labels: Optional[tuple[str, str]] = None,
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    theta_points: Optional[Sequence[tuple[np.ndarray, str, str, str]]] = None,
    trace_first: Optional[OptimizationTrace] = None,
    trace_zero: Optional[OptimizationTrace] = None,
    trace_lbfgs: Optional[OptimizationTrace] = None,
    grid_size: int = 60,
    levels: int = 15,
    filename: str = "theta_objective_contours.png",
) -> None:
    path = _ensure_plot_dir(plot_dir)
    grid_x, grid_y, objective_grid = theta_objective_contour_grid(
        x_samples,
        objective_model,
        policy_spec,
        theta_base,
        axis_indices=axis_indices,
        theta_refs=theta_refs,
        grid_size=grid_size,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    contour = ax.contourf(grid_x, grid_y, objective_grid, levels=levels, cmap="viridis")
    ax.contour(grid_x, grid_y, objective_grid, levels=levels, colors="black", linewidths=0.4, alpha=0.35)
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label("Objective value")

    if axis_labels is None:
        ax.set_xlabel(f"theta[{axis_indices[0]}]")
        ax.set_ylabel(f"theta[{axis_indices[1]}]")
    else:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    ax.set_title("Objective contour over theta slice")

    show_legend = False
    if trace_first is not None and trace_first.theta_values is not None:
        theta_path = np.asarray(trace_first.theta_values, dtype=float)
        ax.plot(
            theta_path[:, axis_indices[0]],
            theta_path[:, axis_indices[1]],
            color="#1f77b4",
            alpha=0.6,
            linewidth=1.4,
            label="first-order path",
        )
        show_legend = True

    if trace_zero is not None and trace_zero.theta_values is not None:
        theta_path = np.asarray(trace_zero.theta_values, dtype=float)
        ax.plot(
            theta_path[:, axis_indices[0]],
            theta_path[:, axis_indices[1]],
            color="#ff7f0e",
            alpha=0.6,
            linewidth=1.4,
            label="zeroth-order path",
        )
        show_legend = True

    if trace_lbfgs is not None and trace_lbfgs.theta_values is not None:
        theta_path = np.asarray(trace_lbfgs.theta_values, dtype=float)
        ax.plot(
            theta_path[:, axis_indices[0]],
            theta_path[:, axis_indices[1]],
            color="#2ca02c",
            alpha=0.6,
            linewidth=1.4,
            label="L-BFGS path",
        )
        show_legend = True

    if theta_points is not None:
        for theta, label, color, marker in theta_points:
            theta_arr = np.asarray(theta, dtype=float)
            ax.scatter(
                [theta_arr[axis_indices[0]]],
                [theta_arr[axis_indices[1]]],
                label=label,
                color=color,
                marker=marker,
                edgecolors=color,
                linewidths=0.5,
                alpha=0.6,
                zorder=5,
            )
        show_legend = True

    if show_legend:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)
