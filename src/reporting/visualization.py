"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from objective.base import Objective
from objective.utils import _policy_value
from experiments.results import ConstantBaselineResult, OptimizationTrace

matplotlib.use("Agg")


ESTIMATOR_STYLES = {
    "first_order": {
        "label": "first-order",
        "color": "#1f77b4",
        "marker": "X",
        "marker_size": 6.2,
        "scatter_size": 28.0,
        "theta_point_size": 60.0,
    },
    "finite_difference": {"label": "finite-difference", "color": "#8c564b", "marker": "P"},
    "gauss_stein": {"label": "gauss-stein", "color": "#ff7f0e", "marker": "s"},
    "stein_difference": {"label": "stein-difference", "color": "#2ca02c", "marker": "^"},
    "spsa": {"label": "SPSA", "color": "#d62728", "marker": "D"},
}
_TRACE_ORDER = ("first_order", "finite_difference", "gauss_stein", "stein_difference", "spsa")
_LINE_ALPHA = 0.5
_LINE_WIDTH = 1.8
_MARKER_SIZE = 4.2
_SCATTER_ALPHA = 0.25
_SCATTER_SIZE = 16.0


def _marker_every(num_points: int) -> int:
    return max(1, num_points // 15)


def _style_marker_size(style: Mapping[str, object]) -> float:
    return float(style.get("marker_size", _MARKER_SIZE))


def _style_scatter_size(style: Mapping[str, object]) -> float:
    return float(style.get("scatter_size", _SCATTER_SIZE))


def _style_theta_point_size(style: Mapping[str, object]) -> float:
    return float(style.get("theta_point_size", _SCATTER_SIZE))


def _constant_baseline_styles(
    baselines: Sequence[ConstantBaselineResult],
) -> list[tuple[ConstantBaselineResult, dict[str, object]]]:
    if not baselines:
        return []
    ordered = sorted(baselines, key=lambda baseline: (float(baseline.value), float(baseline.u)))
    cmap = matplotlib.colormaps["cividis"]
    if len(ordered) == 1:
        levels = np.asarray([0.85], dtype=float)
    else:
        levels = np.linspace(0.85, 0.25, num=len(ordered), dtype=float)
    styled: list[tuple[ConstantBaselineResult, dict[str, object]]] = []
    for rank, (baseline, level) in enumerate(zip(ordered, levels, strict=True), start=1):
        color = cmap(float(level))
        rank_label = "best" if rank == 1 else f"rank {rank}"
        styled.append(
            (
                baseline,
                {
                    "color": color,
                    "linewidth": 1.35 if rank == 1 else 1.1,
                    "alpha": 0.8 if rank == 1 else 0.6,
                    "label": f"const u={float(baseline.u):.2f} ({rank_label})",
                },
            )
        )
    return styled


def _ordered_traces(
    traces: Mapping[str, OptimizationTrace],
) -> list[tuple[str, OptimizationTrace]]:
    return [(name, traces[name]) for name in _TRACE_ORDER if name in traces]


def _ensure_plot_dir(plot_dir: str) -> Path:
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curves(
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
    u_star: Optional[float] = None,
    constant_u_baselines: Sequence[ConstantBaselineResult] = (),
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
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
            marker=style["marker"],
            markersize=_style_marker_size(style),
            markevery=_marker_every(len(trace.steps)),
        )
    for baseline, baseline_style in _constant_baseline_styles(constant_u_baselines):
        ax_loss.axhline(
            float(baseline.value),
            color=baseline_style["color"],
            linestyle="--",
            linewidth=float(baseline_style["linewidth"]),
            alpha=float(baseline_style["alpha"]),
            label=str(baseline_style["label"]),
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
                alpha=_LINE_ALPHA,
                linewidth=_LINE_WIDTH,
                marker=style["marker"],
                markersize=_style_marker_size(style),
                markevery=_marker_every(len(trace.steps)),
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
    has_est = any(trace.theta_grad_norms is not None for _, trace in trace_items)

    if has_true and has_est:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax_norm, ax_err = axes
    else:
        fig, ax_norm = plt.subplots(1, 1, figsize=(8, 4.5))
        ax_err = None

    for name, trace in trace_items:
        series = trace.true_theta_grad_norms
        if series is None:
            series = trace.theta_grad_norms
        if series is None:
            raise ValueError("theta_grad_norms or true_theta_grad_norms must be provided.")
        style = ESTIMATOR_STYLES[name]
        ax_norm.plot(
            trace.steps,
            series,
            label=style["label"],
            color=style["color"],
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
            marker=style["marker"],
            markersize=_style_marker_size(style),
            markevery=_marker_every(len(trace.steps)),
        )
    if has_true:
        ax_norm.set_ylabel("|theta grad norm| (true)")
    else:
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
                alpha=_LINE_ALPHA,
                linewidth=_LINE_WIDTH,
                marker=style["marker"],
                markersize=_style_marker_size(style),
                markevery=_marker_every(len(trace.steps)),
            )
        ax_err.set_ylabel("|norm error|")
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
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
            marker=style["marker"],
            markersize=_style_marker_size(style),
            markevery=_marker_every(len(trace.steps)),
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


def _policy_output_histogram_bins(series_list: Sequence[np.ndarray]) -> np.ndarray:
    combined = np.concatenate([np.asarray(series, dtype=float).reshape(-1) for series in series_list])
    min_val = float(np.min(combined))
    max_val = float(np.max(combined))
    if np.isclose(min_val, max_val):
        pad = 0.05 if min_val == 0.0 else abs(min_val) * 0.05
        return np.linspace(min_val - pad, max_val + pad, 20)
    return np.histogram_bin_edges(combined, bins="auto")


def _policy_outputs_for_theta(
    objective: Objective,
    theta: np.ndarray,
    x_samples: np.ndarray,
) -> np.ndarray:
    u_values = np.asarray(_policy_value(objective, np.asarray(theta, dtype=float), x_samples), dtype=float)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float)
    return u_values.reshape(-1)


def _row_objective_values(
    objective: Objective,
    x_samples: np.ndarray,
    u_values: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    if u_arr.shape != (x_arr.shape[0],):
        raise ValueError("u_values must match the number of x_samples rows.")

    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        values = np.asarray(value_batch_fn(x_arr, u_arr), dtype=float)
        if values.shape != (x_arr.shape[0],):
            raise ValueError("objective._value_batch(x_array, u_array) must return shape (n_samples,).")
        return values

    value_at_u_fn = getattr(objective, "value_at_u", None)
    if callable(value_at_u_fn):
        value_at_u_typed = cast(Callable[[np.ndarray, float], float], value_at_u_fn)
        values = np.empty(x_arr.shape[0], dtype=float)
        for idx, u_val in enumerate(u_arr):
            values[idx] = float(value_at_u_typed(x_arr[idx : idx + 1], float(u_val)))
        return values

    raise ValueError(
        "Objective diagnostics require objective._value_batch(x_array, u_array) or "
        "objective.value_at_u(x_batch, u)."
    )


def _binned_mean_line(
    x_values: np.ndarray,
    y_values: np.ndarray,
    bins: np.ndarray,
    min_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_values, dtype=float).reshape(-1)
    y_arr = np.asarray(y_values, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_values and y_values must have matching shapes.")
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("bins must be a 1D array with at least two edges.")

    bin_ids = np.digitize(x_arr, bins, right=False) - 1
    last_bin_index = bins.size - 2
    bin_ids = np.clip(bin_ids, 0, last_bin_index)

    def collect(required_count: int) -> tuple[np.ndarray, np.ndarray]:
        centers: list[float] = []
        means: list[float] = []
        for idx in range(bins.size - 1):
            mask = bin_ids == idx
            count = int(np.count_nonzero(mask))
            if count < required_count:
                continue
            centers.append(float(0.5 * (bins[idx] + bins[idx + 1])))
            means.append(float(np.mean(y_arr[mask])))
        return np.asarray(centers, dtype=float), np.asarray(means, dtype=float)

    centers, means = collect(min_count)
    if centers.size > 0:
        return centers, means

    return collect(1)


def _plot_policy_u_histograms(
    observed_u: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    filename: str = "policy_u_histograms.png",
) -> None:
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    if observed_u_arr.shape != (x_arr.shape[0],):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return

    path = _ensure_plot_dir(plot_dir)
    ordered_names = [name for name in _TRACE_ORDER if name in theta_by_estimator]
    extra_names = sorted(name for name in theta_by_estimator if name not in _TRACE_ORDER)
    ordered_names.extend(extra_names)
    policy_outputs = {
        name: _policy_outputs_for_theta(objective, theta_by_estimator[name], x_arr)
        for name in ordered_names
    }
    bins = _policy_output_histogram_bins([observed_u_arr, *policy_outputs.values()])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.75))
    ax.hist(
        observed_u_arr,
        bins=bins,
        density=True,
        label="observed U",
        color="#bdbdbd",
        edgecolor="#969696",
        alpha=_SCATTER_ALPHA,
        linewidth=0.8,
    )
    for name in ordered_names:
        style = ESTIMATOR_STYLES[name]
        ax.hist(
            policy_outputs[name],
            bins=bins,
            density=True,
            label=style["label"],
            color=style["color"],
            histtype="step",
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
        )

    ax.set_xlabel("u")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _plot_policy_u_vs_objective(
    observed_u: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    filename: str = "policy_u_vs_objective.png",
) -> None:
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    if observed_u_arr.shape != (x_arr.shape[0],):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return

    path = _ensure_plot_dir(plot_dir)
    ordered_names = [name for name in _TRACE_ORDER if name in theta_by_estimator]
    extra_names = sorted(name for name in theta_by_estimator if name not in _TRACE_ORDER)
    ordered_names.extend(extra_names)
    policy_outputs = {
        name: _policy_outputs_for_theta(objective, theta_by_estimator[name], x_arr)
        for name in ordered_names
    }
    bins = _policy_output_histogram_bins([observed_u_arr, *policy_outputs.values()])
    observed_values = _row_objective_values(objective, x_arr, observed_u_arr)

    fig, ax = plt.subplots(1, 1, figsize=(8.25, 5.0))
    ax.scatter(
        observed_u_arr,
        observed_values,
        color="#969696",
        alpha=_SCATTER_ALPHA,
        s=_SCATTER_SIZE,
        linewidths=0.0,
    )
    observed_centers, observed_means = _binned_mean_line(observed_u_arr, observed_values, bins)
    if observed_centers.size > 0:
        ax.plot(
            observed_centers,
            observed_means,
            color="#636363",
            linewidth=_LINE_WIDTH,
            alpha=_LINE_ALPHA,
            label="observed U",
        )

    for name in ordered_names:
        style = ESTIMATOR_STYLES[name]
        policy_u = policy_outputs[name]
        objective_values = _row_objective_values(objective, x_arr, policy_u)
        ax.scatter(
            policy_u,
            objective_values,
            color=style["color"],
            marker=style["marker"],
            alpha=_SCATTER_ALPHA,
            s=_style_scatter_size(style),
            linewidths=0.0,
        )
        centers, means = _binned_mean_line(policy_u, objective_values, bins)
        if centers.size == 0:
            continue
        ax.plot(
            centers,
            means,
            color=style["color"],
            linewidth=_LINE_WIDTH,
            alpha=_LINE_ALPHA,
            label=style["label"],
        )

    ax.set_xlabel("u")
    ax.set_ylabel("M(x, u)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def plot_objective_u_slice(
    x_samples: np.ndarray,
    objective: Objective,
    traces: Mapping[str, OptimizationTrace],
    plot_dir: str,
    u_star: Optional[float] = None,
    constant_u_baselines: Sequence[ConstantBaselineResult] = (),
) -> None:
    """Plot objective value as a function of u.

    Uses the objective's value_at_u method for computing values at fixed u.
    """
    trace_items = _ordered_traces(traces)
    if not trace_items:
        return
    path = _ensure_plot_dir(plot_dir)
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")

    u_values: list[float] = []
    for _, trace in trace_items:
        u_values.extend(list(trace.u_values))
    if u_star is not None:
        u_values.append(float(u_star))
    for baseline in constant_u_baselines:
        u_values.append(float(baseline.u))
    if u_values:
        u_min = float(min(u_values))
        u_max = float(max(u_values))
        if np.isclose(u_min, u_max):
            pad = 0.1 if u_min == 0.0 else abs(u_min) * 0.1
        else:
            pad = 0.1 * (u_max - u_min)
        u_grid = np.linspace(u_min - pad, u_max + pad, 200)
    else:
        u_grid = np.linspace(-0.5, 0.5, 200)

    # Use value_at_u method if available
    value_at_u_fn = getattr(objective, "value_at_u", None)
    if not callable(value_at_u_fn):
        return
    value_at_u_typed = cast(Callable[[np.ndarray, float], float], value_at_u_fn)

    def value_at_u_scalar(u: float) -> float:
        return float(value_at_u_typed(x_arr, float(u)))

    obj_grid = [value_at_u_scalar(float(u)) for u in u_grid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(u_grid, obj_grid, color="black", label="objective", alpha=_LINE_ALPHA, linewidth=_LINE_WIDTH)
    for name, trace in trace_items:
        style = ESTIMATOR_STYLES[name]
        zorder = 4 if name == "gauss_stein" else 3
        ax.scatter(
            trace.u_values,
            trace.objective_values,
            color=style["color"],
            label=style["label"],
            marker=style["marker"],
            edgecolors=style["color"],
            linewidths=0.4,
            alpha=0.65,
            zorder=zorder,
            s=_style_scatter_size(style),
        )
    ax.set_ylabel("Objective value")
    ax.set_xlabel("u")
    if u_star is not None:
        ax.axvline(
            u_star,
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="u*",
        )
    for baseline, baseline_style in _constant_baseline_styles(constant_u_baselines):
        ax.axvline(
            float(baseline.u),
            color=baseline_style["color"],
            linestyle=":",
            linewidth=1.0,
            alpha=float(baseline_style["alpha"]),
        )
        ax.scatter(
            [float(baseline.u)],
            [float(baseline.value)],
            color=baseline_style["color"],
            marker="x",
            s=48.0,
            alpha=float(baseline_style["alpha"]),
            label=str(baseline_style["label"]),
            zorder=5,
        )
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    x_samples: np.ndarray,
    objective: Objective,
    theta_base: np.ndarray,
    axis_indices: tuple[int, int] = (0, 1),
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    grid_size: int = 60,
    pad_ratio: float = 0.2,
    min_pad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute contour grid for theta-level objective."""
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")
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
            objective_grid[i, j] = float(objective.value(theta, x_arr))

    return grid_x, grid_y, objective_grid


def plot_theta_objective_contours(
    x_samples: np.ndarray,
    objective: Objective,
    theta_base: np.ndarray,
    plot_dir: str,
    axis_indices: tuple[int, int] = (0, 1),
    axis_labels: Optional[tuple[str, str]] = None,
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    theta_points: Optional[Sequence[tuple[np.ndarray, str, str, str, float | None]]] = None,
    traces: Optional[Mapping[str, OptimizationTrace]] = None,
    grid_size: int = 60,
    levels: int = 15,
    filename: str = "theta_objective_contours.png",
) -> None:
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array.")
    path = _ensure_plot_dir(plot_dir)
    grid_x, grid_y, objective_grid = theta_objective_contour_grid(
        x_arr,
        objective,
        theta_base,
        axis_indices=axis_indices,
        theta_refs=theta_refs,
        grid_size=grid_size,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    contour = ax.contourf(grid_x, grid_y, objective_grid, levels=levels, cmap="viridis")
    ax.contour(grid_x, grid_y, objective_grid, levels=levels, colors="black", linewidths=0.4, alpha=0.3)
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
                alpha=_LINE_ALPHA,
                linewidth=_LINE_WIDTH,
                marker=style["marker"],
                markersize=_style_marker_size(style),
                markevery=_marker_every(theta_path.shape[0]),
                label=f"{style['label']} path",
            )
            show_legend = True

    if theta_points is not None:
        for theta, label, color, marker, size in theta_points:
            theta_arr = np.asarray(theta, dtype=float)
            ax.scatter(
                [theta_arr[axis_indices[0]]],
                [theta_arr[axis_indices[1]]],
                label=label,
                color=color,
                marker=marker,
                s=_SCATTER_SIZE if size is None else float(size),
                edgecolors=color,
                linewidths=0.5,
                alpha=0.5,
                zorder=5,
            )
        show_legend = True

    if show_legend:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)
