"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.models import ObjectiveModel, StateVector
from experiments.results import OptimizationTrace
from optimization.policy import PolicySpec, policy_u

matplotlib.use("Agg")


ESTIMATOR_STYLES = {
    "first_order": {"label": "first-order", "color": "#1f77b4", "marker": "o"},
    "zeroth_order": {"label": "zeroth-order", "color": "#ff7f0e", "marker": "o"},
    "lbfgs": {"label": "L-BFGS", "color": "#2ca02c", "marker": "x"},
}
_TRACE_ORDER = ("first_order", "zeroth_order", "lbfgs")


def _ordered_traces(
    traces: Mapping[str, OptimizationTrace],
) -> list[tuple[str, OptimizationTrace]]:
    return [(name, traces[name]) for name in _TRACE_ORDER if name in traces]


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
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
    u_star: Optional[float] = None,
) -> None:
    trace_items = _ordered_traces(traces)
    if not trace_items:
        return
    path = _ensure_plot_dir(plot_dir)
    if u_star is not None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_loss, ax_dist = axes
    else:
        fig, ax_loss = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_dist = None

    for name, trace in trace_items:
        style = ESTIMATOR_STYLES[name]
        ax_loss.plot(
            trace.steps,
            trace.objective_values,
            label=style["label"],
            color=style["color"],
            alpha=0.6,
        )
    ax_loss.set_ylabel("Objective value")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if ax_dist is not None and u_star is not None:
        for name, trace in trace_items:
            style = ESTIMATOR_STYLES[name]
            dist_values = [abs(u - u_star) for u in trace.u_values]
            ax_dist.plot(
                trace.steps,
                dist_values,
                label=style["label"],
                color=style["color"],
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
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
) -> None:
    trace_items = _ordered_traces(traces)
    if not trace_items:
        return
    path = _ensure_plot_dir(plot_dir)
    has_true = any(trace.true_theta_grad_norms is not None for _, trace in trace_items)

    if has_true:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_norm, ax_err = axes
    else:
        fig, ax_norm = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_err = None

    for name, trace in trace_items:
        if trace.theta_grad_norms is None:
            raise ValueError("theta_grad_norms must be provided for gradient norm plots.")
        style = ESTIMATOR_STYLES[name]
        ax_norm.plot(
            trace.steps,
            trace.theta_grad_norms,
            label=style["label"],
            color=style["color"],
            alpha=0.6,
        )
    ax_norm.set_ylabel("|theta grad norm|")
    ax_norm.legend()
    ax_norm.grid(True, alpha=0.3)

    if ax_err is not None:
        for name, trace in trace_items:
            if trace.true_theta_grad_norms is None or trace.theta_grad_norms is None:
                continue
            style = ESTIMATOR_STYLES[name]
            err_values = [
                abs(g - t)
                for g, t in zip(trace.theta_grad_norms, trace.true_theta_grad_norms)
            ]
            ax_err.plot(
                trace.steps,
                err_values,
                label=f"{style['label']} error",
                color=style["color"],
                alpha=0.6,
            )
        ax_err.set_ylabel("|grad error|")
        ax_err.set_xlabel("Step")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)
    else:
        ax_norm.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(path / "gradient_norms.png", dpi=200)
    plt.close(fig)


def plot_step_sizes(
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
) -> None:
    trace_items = _ordered_traces(traces)
    if not trace_items:
        return
    path = _ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    has_series = False

    for name, trace in trace_items:
        if trace.step_sizes is None:
            continue
        if len(trace.step_sizes) != len(trace.steps):
            raise ValueError("step_sizes must match steps length for plotting.")
        style = ESTIMATOR_STYLES[name]
        ax.plot(
            trace.steps,
            trace.step_sizes,
            label=style["label"],
            color=style["color"],
            alpha=0.6,
        )
        has_series = True

    if not has_series:
        plt.close(fig)
        return

    ax.set_ylabel("Step size")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / "step_sizes.png", dpi=200)
    plt.close(fig)


def plot_objective_u_slice(
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
    u_star: Optional[float] = None,
) -> None:
    trace_items = _ordered_traces(traces)
    if not trace_items:
        return
    path = _ensure_plot_dir(plot_dir)
    x_list = _as_state_list(x_samples)
    u_values: list[float] = []
    for _, trace in trace_items:
        u_values.extend(list(trace.u_values))
    if u_star is not None:
        u_values.append(float(u_star))
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
    for name, trace in trace_items:
        style = ESTIMATOR_STYLES[name]
        if name == "lbfgs":
            ax_obj.plot(
                trace.u_values,
                trace.objective_values,
                color=style["color"],
                label=f"{style['label']} path",
                alpha=0.6,
            )
            if trace.u_values and trace.objective_values:
                ax_obj.scatter(
                    [trace.u_values[-1]],
                    [trace.objective_values[-1]],
                    color=style["color"],
                    marker=style["marker"],
                    label=f"{style['label']} final",
                )
        else:
            zorder = 4 if name == "zeroth_order" else 3
            ax_obj.scatter(
                trace.u_values,
                trace.objective_values,
                color=style["color"],
                label=style["label"],
                marker=style["marker"],
                edgecolors=style["color"],
                linewidths=0.4,
                alpha=0.6,
                zorder=zorder,
            )
    ax_obj.set_ylabel("Objective value")
    if u_star is not None:
        ax_obj.axvline(
            u_star,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="u*",
        )
    ax_obj.legend()
    ax_obj.grid(True, alpha=0.3)

    ax_grad.plot(u_grid, grad_grid, color="black", label="true grad", alpha=0.6)
    for name, trace in trace_items:
        style = ESTIMATOR_STYLES[name]
        if name == "lbfgs":
            ax_grad.plot(
                trace.u_values,
                trace.grad_estimates,
                color=style["color"],
                label=f"{style['label']} path",
                alpha=0.6,
            )
            if trace.u_values and trace.grad_estimates:
                ax_grad.scatter(
                    [trace.u_values[-1]],
                    [trace.grad_estimates[-1]],
                    color=style["color"],
                    marker=style["marker"],
                    label=f"{style['label']} final",
                )
        else:
            zorder = 4 if name == "zeroth_order" else 3
            ax_grad.scatter(
                trace.u_values,
                trace.grad_estimates,
                color=style["color"],
                label=f"{style['label']} est",
                marker=style["marker"],
                edgecolors=style["color"],
                linewidths=0.4,
                alpha=0.6,
                zorder=zorder,
            )
    ax_grad.set_ylabel("Gradient")
    ax_grad.set_xlabel("u")
    if u_star is not None:
        ax_grad.axvline(
            u_star,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="u*",
        )
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
    traces: Optional[Mapping[str, OptimizationTrace]] = None,
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
    if traces is not None:
        for name, trace in _ordered_traces(traces):
            if trace.theta_values is None:
                continue
            style = ESTIMATOR_STYLES[name]
            theta_path = np.asarray(trace.theta_values, dtype=float)
            ax.plot(
                theta_path[:, axis_indices[0]],
                theta_path[:, axis_indices[1]],
                color=style["color"],
                alpha=0.6,
                linewidth=1.4,
                label=f"{style['label']} path",
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
