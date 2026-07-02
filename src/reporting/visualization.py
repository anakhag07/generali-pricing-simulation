"""Visualization utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from objective.base import Objective
from objective.utils import _policy_value

matplotlib.use("Agg")

if TYPE_CHECKING:
    from experiments.results import ConstantBaselineResult, OptimizationTrace


ESTIMATOR_STYLES = {
    "constant": {"label": "constant", "color": "#9467bd", "marker": "o"},
    "first_order": {
        "label": "first-order",
        "color": "#1f77b4",
        "marker": "X",
        "marker_size": 6.2,
        "scatter_size": 28.0,
    },
    "finite_difference": {"label": "finite-difference", "color": "#8c564b", "marker": "P"},
    "gauss_stein": {"label": "gauss-stein", "color": "#ff7f0e", "marker": "s"},
    "stein_difference": {"label": "stein-difference", "color": "#2ca02c", "marker": "^"},
    "spsa": {"label": "SPSA", "color": "#d62728", "marker": "D"},
}
_TRACE_ORDER = ("constant", "first_order", "finite_difference", "gauss_stein", "stein_difference", "spsa")
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


def _theta_contour_point_style(label: str) -> dict[str, object]:
    if label == "initial":
        return {
            "marker": "o",
            "s": 64.0,
            "facecolors": "white",
            "edgecolors": "#111111",
            "linewidths": 1.0,
            "alpha": 0.95,
            "zorder": 6,
        }
    if label == "first-order final point":
        return {
            "marker": "X",
            "s": 120.0,
            "color": "black",
            "edgecolors": "black",
            "linewidths": 1.2,
            "alpha": 0.95,
            "zorder": 7,
        }
    return {
        "marker": "o",
        "s": _SCATTER_SIZE,
        "color": "#636363",
        "edgecolors": "#636363",
        "linewidths": 0.5,
        "alpha": 0.5,
        "zorder": 5,
    }


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


def _estimator_style(estimator: str) -> Mapping[str, object]:
    return ESTIMATOR_STYLES.get(
        estimator,
        {
            "label": estimator,
            "color": "#636363",
            "marker": "o",
            "marker_size": _MARKER_SIZE,
            "scatter_size": _SCATTER_SIZE,
        },
    )


def _sweep_points_by_estimator(
    points: Sequence[Mapping[str, object]],
    *,
    sweep_key: str,
) -> list[tuple[str, list[Mapping[str, object]]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for point in points:
        estimator = str(point["estimator"])
        grouped.setdefault(estimator, []).append(point)
    ordered: list[tuple[str, list[Mapping[str, object]]]] = []
    for estimator in _TRACE_ORDER:
        if estimator in grouped:
            grouped[estimator].sort(key=lambda point: float(point[sweep_key]))
            ordered.append((estimator, grouped[estimator]))
    for estimator in sorted(name for name in grouped if name not in _TRACE_ORDER):
        grouped[estimator].sort(key=lambda point: float(point[sweep_key]))
        ordered.append((estimator, grouped[estimator]))
    return ordered


def _ordered_labels(rows: Sequence[Mapping[str, object]], key: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = str(row[key])
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _ordered_estimators_from_rows(rows: Sequence[Mapping[str, object]]) -> list[str]:
    estimators = {str(row["estimator"]) for row in rows}
    ordered = [name for name in _TRACE_ORDER if name in estimators]
    ordered.extend(sorted(name for name in estimators if name not in _TRACE_ORDER))
    return ordered


def _optional_float_value(value: object) -> float | None:
    if value is None or value == "":
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def _comparison_hatch(index: int) -> str:
    hatches = ("", "//", "\\\\", "xx", "..", "++", "oo", "--")
    return hatches[index % len(hatches)]


def _comparison_alpha(index: int) -> float:
    alphas = (0.95, 0.82, 0.70, 0.58)
    return alphas[index % len(alphas)]


def _comparison_trace_groups(
    trace_rows: Sequence[Mapping[str, object]],
    value_key: str,
) -> list[tuple[str, str, list[tuple[int, float]]]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in trace_rows:
        value = _optional_float_value(row.get(value_key))
        if value is None:
            continue
        comparison = str(row["comparison"])
        estimator = str(row["estimator"])
        step = int(row["step"])
        grouped.setdefault((comparison, estimator), []).append((step, value))

    comparison_order = {label: idx for idx, label in enumerate(_ordered_labels(trace_rows, "comparison"))}
    estimator_order = {label: idx for idx, label in enumerate(_ordered_estimators_from_rows(trace_rows))}
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            comparison_order.get(key[0], len(comparison_order)),
            estimator_order.get(key[1], len(estimator_order)),
            key[0],
            key[1],
        ),
    )
    return [
        (comparison, estimator, sorted(grouped[(comparison, estimator)]))
        for comparison, estimator in ordered_keys
    ]


def _plot_comparison_curve(
    trace_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    value_key: str,
    y_label: str,
    filename: str,
) -> None:
    if not trace_rows:
        return
    groups = _comparison_trace_groups(trace_rows, value_key)
    if not groups:
        return
    path = _ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    comparison_order = {label: idx for idx, label in enumerate(_ordered_labels(trace_rows, "comparison"))}
    for comparison, estimator, points in groups:
        style = _estimator_style(estimator)
        comparison_idx = comparison_order.get(comparison, 0)
        steps = [step for step, _ in points]
        values = [value for _, value in points]
        ax.plot(
            steps,
            values,
            label=f"{comparison} / {style['label']}",
            color=style["color"],
            alpha=_comparison_alpha(comparison_idx),
            linewidth=_LINE_WIDTH,
            marker=style["marker"],
            markersize=_style_marker_size(style),
            markevery=_marker_every(len(points)),
            linestyle=("-" if comparison_idx % 2 == 0 else "--"),
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", ncols=2)
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def plot_comparison_objective_curves(
    trace_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    filename: str = "objective_curves.png",
) -> None:
    """Plot comparison objective traces grouped by policy and estimator."""
    _plot_comparison_curve(
        trace_rows,
        plot_dir,
        value_key="objective",
        y_label="Objective value",
        filename=filename,
    )


def plot_comparison_u_curves(
    trace_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    filename: str = "u_curves.png",
) -> None:
    """Plot comparison action traces grouped by policy and estimator."""
    _plot_comparison_curve(
        trace_rows,
        plot_dir,
        value_key="u",
        y_label="Mean u",
        filename=filename,
    )


def plot_comparison_final_metric(
    final_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    metric_key: str,
    metric_label: str,
    filename: str,
) -> None:
    """Plot a grouped bar chart for a final comparison metric."""
    if not final_rows:
        return
    comparisons = _ordered_labels(final_rows, "comparison")
    estimators = _ordered_estimators_from_rows(final_rows)
    if not comparisons or not estimators:
        return

    value_by_pair: dict[tuple[str, str], float] = {}
    for row in final_rows:
        value = _optional_float_value(row.get(metric_key))
        if value is None:
            continue
        value_by_pair[(str(row["comparison"]), str(row["estimator"]))] = value
    if not value_by_pair:
        return

    path = _ensure_plot_dir(plot_dir)
    x_positions = np.arange(len(comparisons), dtype=float)
    group_width = 0.78
    bar_width = group_width / max(len(estimators), 1)
    first_bar_positions = x_positions - group_width / 2.0 + bar_width / 2.0

    fig, ax = plt.subplots(1, 1, figsize=(max(7.5, 1.35 * len(comparisons)), 5.5))
    for estimator_idx, estimator in enumerate(estimators):
        style = _estimator_style(estimator)
        plotted_label = False
        for comparison_idx, comparison in enumerate(comparisons):
            value = value_by_pair.get((comparison, estimator))
            if value is None:
                continue
            ax.bar(
                first_bar_positions[comparison_idx] + estimator_idx * bar_width,
                value,
                width=bar_width * 0.92,
                color=style["color"],
                alpha=_comparison_alpha(comparison_idx),
                hatch=_comparison_hatch(comparison_idx),
                edgecolor="#252525",
                linewidth=0.7,
                label=str(style["label"]) if not plotted_label else None,
            )
            plotted_label = True

    ax.set_xticks(x_positions)
    ax.set_xticklabels(comparisons, rotation=25, ha="right")
    ax.set_xlabel("Policy comparison")
    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Estimator")
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def plot_policy_pca_final_objective(
    final_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    filename: str = "policy_pca_final_objective.png",
) -> None:
    """Plot mean final objective vs policy PCA dimension with seed bands."""
    _plot_policy_pca_metric(
        final_rows,
        plot_dir,
        value_key="final_value",
        y_label="Final objective value",
        filename=filename,
    )


def plot_policy_pca_richness_gap(
    final_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    filename: str = "policy_pca_richness_gap.png",
) -> None:
    """Plot J(constant) - J(policy class) vs policy PCA dimension."""
    gap_rows = _policy_pca_gap_rows(final_rows)
    _plot_policy_pca_metric(
        gap_rows,
        plot_dir,
        value_key="richness_gap",
        y_label="J(constant) - J(policy)",
        filename=filename,
        skip_policy="constant",
    )


def plot_policy_pca_u_spread(
    final_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    filename: str = "policy_pca_u_spread.png",
) -> None:
    """Plot final 90% action spread vs policy PCA dimension."""
    _plot_policy_pca_metric(
        final_rows,
        plot_dir,
        value_key="final_u_iqr90",
        y_label="Final u 5-95% spread",
        filename=filename,
    )


def plot_policy_pca_acceptance_spread(
    final_rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    filename: str = "policy_pca_acceptance_spread.png",
) -> None:
    """Plot final 90% acceptance spread vs policy PCA dimension."""
    _plot_policy_pca_metric(
        final_rows,
        plot_dir,
        value_key="final_acceptance_iqr90",
        y_label="Final acceptance 5-95% spread",
        filename=filename,
    )


def _plot_policy_pca_metric(
    rows: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    value_key: str,
    y_label: str,
    filename: str,
    skip_policy: str | None = None,
) -> None:
    valid_rows = [row for row in rows if _row_float(row.get(value_key)) is not None]
    if skip_policy is not None:
        valid_rows = [row for row in valid_rows if str(row.get("policy_class")) != skip_policy]
    if not valid_rows:
        return

    labels = _policy_pca_labels(valid_rows)
    x_positions = np.arange(len(labels), dtype=float)
    label_to_x = {label: idx for idx, label in enumerate(labels)}
    policies = _ordered_labels(valid_rows, "policy_class")

    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(max(7.5, 1.2 * len(labels)), 5.5))

    for policy_idx, policy_class in enumerate(policies):
        if policy_class == skip_policy:
            continue
        means: list[float] = []
        stds: list[float] = []
        xs: list[float] = []
        for label in labels:
            values = [
                float(cast(float, _row_float(row.get(value_key))))
                for row in valid_rows
                if str(row.get("policy_class")) == policy_class
                and _pca_label(row.get("pca_dim")) == label
            ]
            if not values:
                continue
            xs.append(float(label_to_x[label]))
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=0)))
        if not xs:
            continue
        color = plt.cm.tab10(policy_idx % 10)
        mean_arr = np.asarray(means, dtype=float)
        std_arr = np.asarray(stds, dtype=float)
        x_arr = np.asarray(xs, dtype=float)
        ax.plot(x_arr, mean_arr, marker="o", linewidth=2.0, label=policy_class, color=color)
        ax.fill_between(x_arr, mean_arr - std_arr, mean_arr + std_arr, color=color, alpha=0.16)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Policy PCA dimension")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _policy_pca_gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    constant_by_key: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if str(row.get("policy_class")) != "constant":
            continue
        value = _row_float(row.get("final_value"))
        if value is None:
            continue
        key = (str(row.get("estimator")), str(row.get("seed")), _pca_label(row.get("pca_dim")))
        constant_by_key[key] = float(value)

    gap_rows: list[dict[str, object]] = []
    for row in rows:
        value = _row_float(row.get("final_value"))
        if value is None:
            continue
        key = (str(row.get("estimator")), str(row.get("seed")), _pca_label(row.get("pca_dim")))
        constant_value = constant_by_key.get(key)
        if constant_value is None:
            continue
        gap_row = dict(row)
        gap_row["richness_gap"] = constant_value - float(value)
        gap_rows.append(gap_row)
    return gap_rows


def _policy_pca_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    labels = {_pca_label(row.get("pca_dim")) for row in rows}
    return sorted(labels, key=_pca_label_sort_key)


def _pca_label(value: object) -> str:
    if value is None:
        return "none"
    text = str(value)
    if text == "" or text.lower() == "none":
        return "none"
    return text


def _pca_label_sort_key(label: str) -> tuple[int, float | str]:
    if label == "none":
        return (1, float("inf"))
    try:
        return (0, float(label))
    except ValueError:
        return (0, label)


def _row_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


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
            style = _estimator_style(name)
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


def _as_2d_x_samples(x_samples: object) -> object:
    if hasattr(x_samples, "iloc") and hasattr(x_samples, "columns"):
        x_frame = x_samples.reset_index(drop=True)
        if x_frame.ndim != 2:
            raise ValueError("x_samples must be a 2D array/DataFrame.")
        return x_frame
    x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array/DataFrame.")
    return x_arr


def _x_sample_count(x_samples: object) -> int:
    return int(x_samples.shape[0])  # type: ignore[attr-defined]


def _x_sample_slice(x_samples: object, start: int, stop: int) -> object:
    if hasattr(x_samples, "iloc"):
        return x_samples.iloc[start:stop].reset_index(drop=True)
    return x_samples[start:stop]  # type: ignore[index]


def _policy_outputs_for_theta(
    objective: Objective,
    theta: np.ndarray,
    x_samples: object,
) -> np.ndarray:
    u_values = np.asarray(_policy_value(objective, np.asarray(theta, dtype=float), x_samples), dtype=float)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float)
    return u_values.reshape(-1)


def _policy_outputs_for_estimator(
    objective: Objective,
    estimator_name: str,
    theta: np.ndarray,
    x_samples: object,
) -> np.ndarray:
    if estimator_name != "constant":
        return _policy_outputs_for_theta(objective, theta, x_samples)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if theta_arr.size != 1:
        raise ValueError("constant estimator theta must contain one scalar.")
    u_values = np.full(_x_sample_count(x_samples), float(theta_arr[0]), dtype=float)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float)
    return u_values.reshape(-1)


def _ordered_policy_estimator_names(theta_by_estimator: Mapping[str, np.ndarray]) -> list[str]:
    ordered_names = [name for name in _TRACE_ORDER if name in theta_by_estimator]
    ordered_names.extend(sorted(name for name in theta_by_estimator if name not in _TRACE_ORDER))
    return ordered_names


def _row_objective_values(
    objective: Objective,
    x_samples: object,
    u_values: np.ndarray,
) -> np.ndarray:
    x_data = _as_2d_x_samples(x_samples)
    n_samples = _x_sample_count(x_data)
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    if u_arr.shape != (n_samples,):
        raise ValueError("u_values must match the number of x_samples rows.")

    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        values = np.asarray(value_batch_fn(x_data, u_arr), dtype=float)
        if values.shape != (n_samples,):
            raise ValueError("objective._value_batch(x_array, u_array) must return shape (n_samples,).")
        return values

    value_at_u_fn = getattr(objective, "value_at_u", None)
    if callable(value_at_u_fn):
        value_at_u_typed = cast(Callable[[object, float], float], value_at_u_fn)
        values = np.empty(n_samples, dtype=float)
        for idx, u_val in enumerate(u_arr):
            values[idx] = float(value_at_u_typed(_x_sample_slice(x_data, idx, idx + 1), float(u_val)))
        return values

    raise ValueError(
        "Objective diagnostics require objective._value_batch(x_array, u_array) or "
        "objective.value_at_u(x_batch, u)."
    )


def _row_acceptance_values(
    objective: Objective,
    x_samples: object,
    u_values: np.ndarray,
) -> np.ndarray:
    x_data = _as_2d_x_samples(x_samples)
    n_samples = _x_sample_count(x_data)
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    if u_arr.shape != (n_samples,):
        raise ValueError("u_values must match the number of x_samples rows.")

    acceptance_fn = getattr(objective, "_acceptance_proba", None)
    if not callable(acceptance_fn):
        raise ValueError("Acceptance diagnostics require objective._acceptance_proba(x_array, u_array).")
    values = np.asarray(acceptance_fn(x_data, u_arr), dtype=float).reshape(-1)
    if values.shape != (n_samples,):
        raise ValueError("objective._acceptance_proba(x_array, u_array) must return shape (n_samples,).")
    return values


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
    filename: str = "u_histogram.png",
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    if observed_u_arr.shape != (_x_sample_count(x_data),):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return

    path = _ensure_plot_dir(plot_dir)
    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    policy_outputs = {
        name: _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
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
        style = _estimator_style(name)
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


def _plot_policy_acceptance_histograms(
    observed_u: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    filename: str = "acceptance_histograms.png",
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    if observed_u_arr.shape != (_x_sample_count(x_data),):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return

    path = _ensure_plot_dir(plot_dir)
    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    observed_acceptance = _row_acceptance_values(objective, x_data, observed_u_arr)
    policy_acceptance = {
        name: _row_acceptance_values(
            objective,
            x_data,
            _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data),
        )
        for name in ordered_names
    }
    bins = _policy_output_histogram_bins([observed_acceptance, *policy_acceptance.values()])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.75))
    ax.hist(
        observed_acceptance,
        bins=bins,
        density=True,
        label="observed U",
        color="#bdbdbd",
        edgecolor="#969696",
        alpha=_SCATTER_ALPHA,
        linewidth=0.8,
    )
    for name in ordered_names:
        style = _estimator_style(name)
        ax.hist(
            policy_acceptance[name],
            bins=bins,
            density=True,
            label=style["label"],
            color=style["color"],
            histtype="step",
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
        )

    ax.set_xlabel("Acceptance probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _plot_policy_delta_u_histograms(
    observed_u: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    filename: str = "delta_u_histogram.png",
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    n_samples = _x_sample_count(x_data)
    if observed_u_arr.shape != (n_samples,):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return

    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    delta_by_estimator: dict[str, np.ndarray] = {}
    for name in ordered_names:
        policy_u = _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
        delta_u = policy_u - observed_u_arr
        finite_delta = delta_u[np.isfinite(delta_u)]
        if finite_delta.size > 0:
            delta_by_estimator[name] = finite_delta
    if not delta_by_estimator:
        return

    bins = _policy_output_histogram_bins(list(delta_by_estimator.values()))
    path = _ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.75))
    for name in ordered_names:
        if name not in delta_by_estimator:
            continue
        style = _estimator_style(name)
        ax.hist(
            delta_by_estimator[name],
            bins=bins,
            density=True,
            label=style["label"],
            color=style["color"],
            histtype="step",
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
        )

    ax.axvline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xlabel("Δu = optimized customer u - historical u")
    ax.set_ylabel("Density")
    ax.set_title("Optimized minus historical customer u")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _plot_policy_delta_u_by_elasticity(
    observed_u: np.ndarray,
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    *,
    u_ref: float = 0.08,
    filename: str = "delta_u_by_sensitivity.png",
    max_scatter_points: int = 30000,
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    n_samples = _x_sample_count(x_data)
    if observed_u_arr.shape != (n_samples,):
        raise ValueError("observed_u must match the number of x_samples rows.")
    if not theta_by_estimator:
        return
    derivative_fn = getattr(objective, "_d_acceptance_du_batch", None)
    if not callable(derivative_fn):
        return

    u_ref_value = float(u_ref)
    u_ref_arr = np.full(n_samples, u_ref_value, dtype=float)
    sensitivity = np.abs(np.asarray(derivative_fn(x_data, u_ref_arr), dtype=float).reshape(-1))
    if sensitivity.shape != (n_samples,):
        raise ValueError(
            "objective._d_acceptance_du_batch(x_array, u_array) must return shape (n_samples,)."
        )
    finite_base = np.isfinite(observed_u_arr) & np.isfinite(sensitivity)
    if not np.any(finite_base):
        return

    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    bins = _policy_output_histogram_bins([sensitivity[finite_base]])
    path = _ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.25))
    thinned = False
    has_series = False

    for name in ordered_names:
        style = _estimator_style(name)
        policy_u = _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
        delta_u = policy_u - observed_u_arr
        finite = finite_base & np.isfinite(delta_u)
        if not np.any(finite):
            continue
        has_series = True

        plot_indices = np.flatnonzero(finite)
        if plot_indices.size > max_scatter_points:
            thinned = True
            plot_indices = plot_indices[
                np.linspace(0, plot_indices.size - 1, max_scatter_points, dtype=int)
            ]
        ax.scatter(
            sensitivity[plot_indices],
            delta_u[plot_indices],
            color=style["color"],
            marker=style["marker"],
            alpha=0.18,
            s=_style_scatter_size(style),
            linewidths=0.0,
            label=style["label"],
        )
        centers, mean_delta = _binned_mean_line(sensitivity[finite], delta_u[finite], bins)
        if centers.size > 0:
            ax.plot(
                centers,
                mean_delta,
                color=style["color"],
                linewidth=_LINE_WIDTH,
                alpha=0.95,
            )

    if not has_series:
        plt.close(fig)
        return

    ax.axhline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_xlabel(
        f"Absolute sensitivity |d p_accept / du| evaluated at u = {u_ref_value:.2f}"
    )
    ax.set_ylabel("Δu = optimized customer u - historical u")
    ax.set_title("Optimized price changes by reference acceptance sensitivity")
    if thinned:
        ax.text(
            0.01,
            0.01,
            f"Scatter thinned to {max_scatter_points:,} customers per estimator; "
            "lines use all rows",
            transform=ax.transAxes,
            fontsize=8,
            color="#525252",
            va="bottom",
        )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _plot_policy_objective_contribution_summary(
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    filename: str = "objective_contribution_summary.png",
    max_scatter_points: int = 30000,
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    if not theta_by_estimator:
        return
    if not callable(getattr(objective, "_acceptance_proba", None)):
        return

    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    profit_by_estimator: dict[str, np.ndarray] = {}
    acceptance_by_estimator: dict[str, np.ndarray] = {}
    for name in ordered_names:
        policy_u = _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
        objective_values = _row_objective_values(objective, x_data, policy_u)
        acceptance_values = _row_acceptance_values(objective, x_data, policy_u)
        expected_profit = -objective_values
        finite = np.isfinite(expected_profit) & np.isfinite(acceptance_values)
        if np.any(finite):
            profit_by_estimator[name] = expected_profit[finite]
            acceptance_by_estimator[name] = acceptance_values[finite]
    if not profit_by_estimator:
        return

    bins = _policy_output_histogram_bins(list(profit_by_estimator.values()))
    path = _ensure_plot_dir(plot_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.75))
    hist_ax = axes[0]
    scatter_ax = axes[1]
    thinned = False

    for name in ordered_names:
        if name not in profit_by_estimator:
            continue
        style = _estimator_style(name)
        profits = profit_by_estimator[name]
        acceptance = acceptance_by_estimator[name]
        hist_ax.hist(
            profits,
            bins=bins,
            density=True,
            label=style["label"],
            color=style["color"],
            histtype="step",
            alpha=_LINE_ALPHA,
            linewidth=_LINE_WIDTH,
        )

        plot_indices = np.arange(profits.size)
        if plot_indices.size > max_scatter_points:
            thinned = True
            plot_indices = plot_indices[
                np.linspace(0, plot_indices.size - 1, max_scatter_points, dtype=int)
            ]
        scatter_ax.scatter(
            acceptance[plot_indices],
            profits[plot_indices],
            color=style["color"],
            marker=style["marker"],
            alpha=0.18,
            s=_style_scatter_size(style),
            linewidths=0.0,
            label=style["label"],
        )

    hist_ax.axvline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.75)
    hist_ax.set_xlabel("Expected profit contribution = -objective contribution")
    hist_ax.set_ylabel("Density")
    hist_ax.set_title("Customer-level expected profit spread")
    hist_ax.grid(True, alpha=0.3)
    hist_ax.legend()

    scatter_ax.axhline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.75)
    scatter_ax.set_xlabel("Predicted acceptance probability")
    scatter_ax.set_ylabel("Expected profit contribution = -objective contribution")
    scatter_ax.set_title("Expected profit vs predicted acceptance")
    scatter_ax.set_xlim(0.0, 1.0)
    scatter_ax.grid(True, alpha=0.3)
    scatter_ax.legend()
    if thinned:
        scatter_ax.text(
            0.01,
            0.01,
            f"Scatter thinned to {max_scatter_points:,} customers per estimator",
            transform=scatter_ax.transAxes,
            fontsize=8,
            color="#525252",
            va="bottom",
        )

    fig.suptitle("Positive expected profit means predicted money made")
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _metric_summary(values: np.ndarray) -> tuple[float, float, float]:
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    finite_values = values_arr[np.isfinite(values_arr)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan
    q25, q75 = np.quantile(finite_values, [0.25, 0.75])
    return float(np.mean(finite_values)), float(q25), float(q75)


def _plot_summary_metric_axis(
    ax: matplotlib.axes.Axes,
    names: Sequence[str],
    summaries: Mapping[str, tuple[float, float, float]],
    *,
    title: str,
    ylabel: str,
    show_interval: bool,
) -> None:
    x_positions = np.arange(len(names), dtype=float)
    means = np.asarray([summaries[name][0] for name in names], dtype=float)
    colors = [str(_estimator_style(name)["color"]) for name in names]
    labels = [str(_estimator_style(name)["label"]) for name in names]
    ax.bar(x_positions, means, color=colors, alpha=0.85, edgecolor="#252525", linewidth=0.7)
    if show_interval:
        q25 = np.asarray([summaries[name][1] for name in names], dtype=float)
        q75 = np.asarray([summaries[name][2] for name in names], dtype=float)
        yerr = np.vstack([np.maximum(means - q25, 0.0), np.maximum(q75 - means, 0.0)])
        ax.errorbar(
            x_positions,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="#111111",
            elinewidth=1.2,
            capsize=4.0,
            alpha=0.45,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)


def _plot_policy_final_summary_metrics(
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    runtime_by_estimator: Mapping[str, float],
    plot_dir: str,
    filename: str = "final_summary_metrics.png",
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    if not theta_by_estimator:
        return
    if not callable(getattr(objective, "_acceptance_proba", None)):
        return

    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)
    u_summaries: dict[str, tuple[float, float, float]] = {}
    objective_summaries: dict[str, tuple[float, float, float]] = {}
    acceptance_summaries: dict[str, tuple[float, float, float]] = {}
    runtime_summaries: dict[str, tuple[float, float, float]] = {}

    for name in ordered_names:
        policy_u = _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
        objective_values = _row_objective_values(objective, x_data, policy_u)
        acceptance_values = _row_acceptance_values(objective, x_data, policy_u)
        runtime = float(runtime_by_estimator.get(name, np.nan))
        u_summaries[name] = _metric_summary(policy_u)
        objective_summaries[name] = _metric_summary(objective_values)
        acceptance_summaries[name] = _metric_summary(acceptance_values)
        runtime_summaries[name] = (runtime, runtime, runtime)

    path = _ensure_plot_dir(plot_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    flat_axes = axes.reshape(-1)
    _plot_summary_metric_axis(
        flat_axes[0],
        ordered_names,
        objective_summaries,
        title="Objective contribution",
        ylabel="Mean M(x, u)",
        show_interval=True,
    )
    _plot_summary_metric_axis(
        flat_axes[1],
        ordered_names,
        u_summaries,
        title="Final policy u",
        ylabel="Mean u",
        show_interval=True,
    )
    _plot_summary_metric_axis(
        flat_axes[2],
        ordered_names,
        acceptance_summaries,
        title="Acceptance probability",
        ylabel="Mean acceptance",
        show_interval=True,
    )
    _plot_summary_metric_axis(
        flat_axes[3],
        ordered_names,
        runtime_summaries,
        title="Runtime",
        ylabel="Seconds",
        show_interval=False,
    )

    fig.suptitle("Final policy summary: bars are means, whiskers are 25-75% customer ranges")
    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def _plot_policy_u_acceptance_histograms(
    x_samples: np.ndarray,
    objective: Objective,
    theta_by_estimator: Mapping[str, np.ndarray],
    plot_dir: str,
    acceptance_floor: float | None = None,
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    if not theta_by_estimator:
        return
    if not callable(getattr(objective, "_acceptance_proba", None)):
        return

    path = _ensure_plot_dir(plot_dir)
    ordered_names = _ordered_policy_estimator_names(theta_by_estimator)

    for name in ordered_names:
        style = _estimator_style(name)
        policy_u = _policy_outputs_for_estimator(objective, name, theta_by_estimator[name], x_data)
        acceptance_values = _row_acceptance_values(objective, x_data, policy_u)
        objective_values = _row_objective_values(objective, x_data, policy_u)
        finite_objective_values = objective_values[np.isfinite(objective_values)]
        bins = _policy_output_histogram_bins([policy_u])
        centers, mean_acceptance = _binned_mean_line(policy_u, acceptance_values, bins)
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.0))
        hist_ax = axes[0]
        scatter_ax = axes[1]
        objective_ax = axes[2]

        hist_ax.hist(
            policy_u,
            bins=bins,
            density=False,
            label="customers",
            color=style["color"],
            edgecolor="#252525",
            alpha=0.35,
            linewidth=0.6,
        )
        if centers.size > 0:
            mean_ax = hist_ax.twinx()
            mean_ax.plot(
                centers,
                mean_acceptance,
                color="#111111",
                linewidth=_LINE_WIDTH,
                marker="o",
                markersize=3.2,
                alpha=0.85,
                label="bin mean acceptance",
            )
            mean_ax.set_ylabel("Mean acceptance")
            mean_ax.set_ylim(0.0, 1.0)
            if acceptance_floor is not None:
                mean_ax.axhline(
                    float(acceptance_floor),
                    color="#7f7f7f",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.75,
                    label="acceptance floor",
                )
            mean_ax.legend(loc="upper right", fontsize="small")

        hist_ax.set_title(f"{style['label']}: final u distribution")
        hist_ax.set_xlabel("Final policy u")
        hist_ax.set_ylabel("Customer count")
        hist_ax.grid(True, alpha=0.3)
        hist_ax.legend(loc="upper left", fontsize="small")

        scatter_ax.scatter(
            policy_u,
            acceptance_values,
            color=style["color"],
            marker=style["marker"],
            alpha=_SCATTER_ALPHA,
            s=_style_scatter_size(style),
            linewidths=0.0,
        )
        if centers.size > 0:
            scatter_ax.plot(
                centers,
                mean_acceptance,
                color="#111111",
                linewidth=_LINE_WIDTH,
                marker="o",
                markersize=3.2,
                alpha=0.85,
                label="bin mean acceptance",
            )
        if acceptance_floor is not None:
            scatter_ax.axhline(
                float(acceptance_floor),
                color="#7f7f7f",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label="acceptance floor",
            )
        scatter_ax.set_title(f"{style['label']}: customer acceptance vs final u")
        scatter_ax.set_xlabel("Final policy u")
        scatter_ax.set_ylabel("Acceptance probability")
        scatter_ax.set_ylim(0.0, 1.0)
        scatter_ax.grid(True, alpha=0.3)
        scatter_ax.legend(fontsize="small")

        if finite_objective_values.size > 0:
            objective_bins = _policy_output_histogram_bins([finite_objective_values])
            objective_ax.hist(
                finite_objective_values,
                bins=objective_bins,
                density=False,
                label="customers",
                color=style["color"],
                edgecolor="#252525",
                alpha=0.35,
                linewidth=0.6,
            )
        objective_ax.axvline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.75)
        objective_ax.set_title(f"{style['label']}: objective contribution")
        objective_ax.set_xlabel("Objective contribution M(x, u)")
        objective_ax.set_ylabel("Customer count")
        objective_ax.grid(True, alpha=0.3)
        if finite_objective_values.size > 0:
            objective_ax.legend(loc="upper right", fontsize="small")

        fig.tight_layout()
        fig.savefig(path / f"{name}.png", dpi=200)
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
    x_data = _as_2d_x_samples(x_samples)

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
    value_at_u_typed = cast(Callable[[object, float], float], value_at_u_fn)

    def value_at_u_scalar(u: float) -> float:
        return float(value_at_u_typed(x_data, float(u)))

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
    x_samples: object,
    objective: Objective,
    theta_base: np.ndarray,
    axis_indices: tuple[int, int] = (0, 1),
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    grid_size: int = 60,
    pad_ratio: float = 0.08,
    min_pad: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute contour grid for theta-level objective."""
    x_data = _as_2d_x_samples(x_samples)
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
            objective_grid[i, j] = float(objective.value(theta, x_data))

    return grid_x, grid_y, objective_grid


def _adaptive_contour_norm(values: np.ndarray) -> matplotlib.colors.Normalize | None:
    finite_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return None
    min_val = float(np.min(finite_values))
    max_val = float(np.max(finite_values))
    if np.isclose(min_val, max_val):
        return None

    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size == finite_values.size:
        positive_min = float(np.min(positive_values))
        ratio = max_val / positive_min if positive_min > 0.0 else np.inf
        if ratio >= 100.0:
            return matplotlib.colors.LogNorm(vmin=positive_min, vmax=max_val)
        return None

    abs_values = np.abs(finite_values)
    nonzero_abs = abs_values[abs_values > 0.0]
    if nonzero_abs.size == 0:
        return None
    max_abs = float(np.max(nonzero_abs))
    min_abs = float(np.min(nonzero_abs))
    if max_abs / min_abs < 100.0:
        return None
    linthresh = max(min_abs, max_abs / 1000.0)
    return matplotlib.colors.SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=min_val,
        vmax=max_val,
        base=10.0,
    )


def plot_theta_objective_contours(
    x_samples: object,
    objective: Objective,
    theta_base: np.ndarray,
    plot_dir: str,
    axis_indices: tuple[int, int] = (0, 1),
    axis_labels: Optional[tuple[str, str]] = None,
    theta_refs: Optional[Sequence[np.ndarray]] = None,
    theta_points: Optional[Sequence[tuple[np.ndarray, str]]] = None,
    traces: Optional[Mapping[str, OptimizationTrace]] = None,
    grid_size: int = 60,
    levels: int = 15,
    filename: str = "theta_objective_contours.png",
) -> None:
    x_data = _as_2d_x_samples(x_samples)
    path = _ensure_plot_dir(plot_dir)
    grid_x, grid_y, objective_grid = theta_objective_contour_grid(
        x_data,
        objective,
        theta_base,
        axis_indices=axis_indices,
        theta_refs=theta_refs,
        grid_size=grid_size,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    norm = _adaptive_contour_norm(objective_grid)
    contour = ax.contourf(grid_x, grid_y, objective_grid, levels=levels, cmap="viridis", norm=norm)
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
            if theta_path.ndim != 2 or max(axis_indices) >= theta_path.shape[1]:
                continue
            ax.plot(
                theta_path[:, axis_indices[0]],
                theta_path[:, axis_indices[1]],
                color=style["color"],
                alpha=_LINE_ALPHA,
                linewidth=_LINE_WIDTH,
                marker=style["marker"],
                markersize=_style_marker_size(style),
                markevery=_marker_every(theta_path.shape[0]),
                label=str(style["label"]),
            )
            show_legend = True

    if theta_points is not None:
        for theta, label in theta_points:
            theta_arr = np.asarray(theta, dtype=float)
            point_style = _theta_contour_point_style(label)
            ax.scatter(
                [theta_arr[axis_indices[0]]],
                [theta_arr[axis_indices[1]]],
                label=label,
                **point_style,
            )
        show_legend = True

    if show_legend:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def plot_sweep_tradeoffs(
    points: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    sweep_key: str,
    sweep_label: str,
    filename: str,
) -> None:
    """Plot final action and acceptance versus a generic sweep parameter."""
    if not points:
        return
    path = _ensure_plot_dir(plot_dir)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_u, ax_acceptance = axes

    for estimator, estimator_points in _sweep_points_by_estimator(points, sweep_key=sweep_key):
        style = _estimator_style(estimator)
        sweep_values = [float(point[sweep_key]) for point in estimator_points]
        u_values = [float(point["u"]) for point in estimator_points]
        acceptance_values = [float(point["mean_acceptance"]) for point in estimator_points]
        plot_kwargs = {
            "label": str(style["label"]),
            "color": style["color"],
            "alpha": _LINE_ALPHA,
            "linewidth": _LINE_WIDTH,
            "marker": style["marker"],
            "markersize": _style_marker_size(style),
        }
        ax_u.plot(sweep_values, u_values, **plot_kwargs)
        ax_acceptance.plot(sweep_values, acceptance_values, **plot_kwargs)

    ax_u.set_ylabel("Final u")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)
    ax_acceptance.set_xlabel(sweep_label)
    ax_acceptance.set_ylabel("Mean acceptance")
    ax_acceptance.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)


def plot_sweep_pareto_frontier(
    points: Sequence[Mapping[str, object]],
    plot_dir: str,
    *,
    sweep_key: str,
    sweep_label: str,
    y_key: str,
    y_label: str,
    filename: str,
) -> None:
    """Plot a generic sweep Pareto frontier colored by the sweep parameter."""
    if not points:
        return
    path = _ensure_plot_dir(plot_dir)
    sweep_values = np.asarray([float(point[sweep_key]) for point in points], dtype=float)
    sweep_min = float(np.min(sweep_values))
    sweep_max = float(np.max(sweep_values))
    if np.isclose(sweep_min, sweep_max):
        sweep_max = sweep_min + 1.0
    norm = matplotlib.colors.Normalize(vmin=sweep_min, vmax=sweep_max)
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for estimator, estimator_points in _sweep_points_by_estimator(points, sweep_key=sweep_key):
        style = _estimator_style(estimator)
        x_values = [float(point["mean_acceptance"]) for point in estimator_points]
        y_values = [float(point[y_key]) for point in estimator_points]
        point_sweep_values = np.asarray([float(point[sweep_key]) for point in estimator_points], dtype=float)
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            alpha=0.25,
            linewidth=1.0,
        )
        ax.scatter(
            x_values,
            y_values,
            c=point_sweep_values,
            cmap=cmap,
            norm=norm,
            label=str(style["label"]),
            marker=style["marker"],
            s=_style_scatter_size(style),
            edgecolors=style["color"],
            linewidths=0.6,
            alpha=0.9,
        )

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(sweep_values)
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label(sweep_label)
    ax.set_xlabel("Mean acceptance")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path / filename, dpi=200)
    plt.close(fig)
