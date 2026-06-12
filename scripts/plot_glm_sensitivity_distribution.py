"""Plot GLM customer elasticity distributions across action values."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from data.loader import (
    eligible_csv_row_indices,
    extract_glm_acceptance_coefficients,
    load_model_artifacts,
    load_x_frame,
    sample_csv_row_indices,
)
from experiments.sensitivity_buckets import (
    glm_price_derivative_matrix,
)

DEFAULT_OUTPUT_ROOT = Path("outputs") / "glm-sensitivity-distribution"
DEFAULT_HIST_U_VALUES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _resolve_u_grid(u_min: float, u_max: float, u_count: int) -> np.ndarray:
    if not np.isfinite([u_min, u_max]).all():
        raise ValueError("u_min and u_max must be finite.")
    if u_min > u_max:
        raise ValueError("u_min must be <= u_max.")
    if u_count <= 0:
        raise ValueError("u_count must be positive.")
    return np.linspace(float(u_min), float(u_max), int(u_count), dtype=float)


def _resolve_row_indices(n_rows: int | None, seed: int | None) -> np.ndarray:
    if n_rows is None:
        return eligible_csv_row_indices("glm")
    if n_rows <= 0:
        raise ValueError("n_rows must be positive when provided.")
    return sample_csv_row_indices("glm", n_rows=int(n_rows), seed=seed)


def _summary_rows(
    u_values: Sequence[float],
    value_matrix: np.ndarray,
) -> list[dict[str, float | int]]:
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    values = np.asarray(value_matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("value_matrix must be 2D.")
    if values.shape[1] != u_arr.size:
        raise ValueError("u_values length must match value_matrix columns.")
    if values.shape[0] == 0:
        raise ValueError("value_matrix must contain at least one customer row.")

    quantiles = np.quantile(values, _QUANTILES, axis=0)
    return [
        {
            "u": float(u),
            "n_rows": int(values.shape[0]),
            "mean": float(np.mean(column)),
            "median": float(quantiles[2, idx]),
            "q05": float(quantiles[0, idx]),
            "q25": float(quantiles[1, idx]),
            "q75": float(quantiles[3, idx]),
            "q95": float(quantiles[4, idx]),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
        for idx, (u, column) in enumerate(zip(u_arr, values.T))
    ]


def _write_summary_csv(rows: Sequence[Mapping[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "u",
        "n_rows",
        "mean",
        "median",
        "q05",
        "q25",
        "q75",
        "q95",
        "min",
        "max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _theoretical_derivative_bound(acceptance_model: object, u_coef: float | None) -> float:
    coeffs = extract_glm_acceptance_coefficients(acceptance_model)
    beta_u = float(u_coef) if u_coef is not None else float(coeffs["u_coef"])
    probability_target = coeffs.get(
        "probability_target",
        getattr(acceptance_model, "probability_target", "acceptance"),
    )
    sign = 1.0 if probability_target == "acceptance" else -1.0
    return sign * beta_u * 0.25


def _plot_mean_elasticity_by_u(
    rows: Sequence[Mapping[str, float | int]],
    output_dir: Path,
    *,
    derivative_bound: float | None,
) -> Path:
    if not rows:
        raise ValueError("At least one summary row is required to plot elasticity by u.")
    output_dir.mkdir(parents=True, exist_ok=True)
    u_values = np.asarray([float(row["u"]) for row in rows], dtype=float)
    mean = np.asarray([float(row["mean"]) for row in rows], dtype=float)
    median = np.asarray([float(row["median"]) for row in rows], dtype=float)
    q05 = np.asarray([float(row["q05"]) for row in rows], dtype=float)
    q25 = np.asarray([float(row["q25"]) for row in rows], dtype=float)
    q75 = np.asarray([float(row["q75"]) for row in rows], dtype=float)
    q95 = np.asarray([float(row["q95"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.fill_between(
        u_values,
        q05,
        q95,
        color="#9ecae1",
        alpha=0.22,
        label="5-95% customers",
    )
    ax.fill_between(
        u_values,
        q25,
        q75,
        color="#4292c6",
        alpha=0.28,
        label="25-75% customers",
    )
    ax.axhline(0.0, color="#636363", linewidth=1.0, alpha=0.7, label="zero")
    if derivative_bound is not None:
        ax.axhline(
            derivative_bound,
            color="#54278f",
            linewidth=1.2,
            linestyle=":",
            label=f"theoretical bound ({derivative_bound:.3f})",
        )
    ax.plot(u_values, mean, color="#08519c", linewidth=2.0, label="mean")
    ax.plot(
        u_values,
        median,
        color="#f16913",
        linewidth=1.8,
        linestyle="--",
        label="median",
    )
    ax.set_xlabel("u")
    ax.set_ylabel("Elasticity (d p_accept / du)")
    ax.set_title("GLM customer elasticity by action value")
    ax.text(
        0.01,
        0.01,
        "Bands show customer quantiles; no y clipping",
        transform=ax.transAxes,
        fontsize=8,
        color="#525252",
        va="bottom",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "mean_elasticity_by_u.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_selected_u_histograms(
    hist_u_values: Sequence[float],
    selected_derivatives: np.ndarray,
    output_dir: Path,
    *,
    bins: int,
    clip_low: float | None,
    clip_high: float | None,
    derivative_bound: float | None,
) -> Path:
    u_arr = np.asarray(hist_u_values, dtype=float).reshape(-1)
    values = np.asarray(selected_derivatives, dtype=float)
    if values.ndim != 2:
        raise ValueError("selected_derivatives must be 2D.")
    if values.shape[1] != u_arr.size:
        raise ValueError("hist_u_values length must match selected_derivatives columns.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("selected_derivatives must be non-empty.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if (clip_low is None) != (clip_high is None):
        raise ValueError("clip_low and clip_high must be provided together.")
    if clip_low is not None and clip_high is not None:
        if not 0.0 <= clip_low < clip_high <= 100.0:
            raise ValueError("clip percentiles must satisfy 0 <= low < high <= 100.")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cols = min(4, u_arr.size)
    n_rows = int(np.ceil(u_arr.size / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_arr = np.asarray(axes, dtype=object).reshape(-1)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    clipped = clip_low is not None and clip_high is not None
    if clipped:
        plot_min, plot_max = np.percentile(values, [float(clip_low), float(clip_high)])
        min_value = float(plot_min)
        max_value = float(plot_max)
    hist_bins: int | np.ndarray = bins
    if min_value < max_value:
        hist_bins = np.linspace(min_value, max_value, bins + 1)

    for ax, u_val, column in zip(axes_arr, u_arr, values.T):
        mean = float(np.mean(column))
        median = float(np.median(column))
        ax.hist(
            column,
            bins=hist_bins,
            color="#9ecae1",
            edgecolor="#6baed6",
            alpha=0.82,
        )
        ax.axvline(0.0, color="#636363", linewidth=1.0, alpha=0.65, label="zero")
        if derivative_bound is not None:
            ax.axvline(
                derivative_bound,
                color="#54278f",
                linewidth=1.2,
                linestyle=":",
                label="theoretical bound",
            )
        if clipped:
            ax.axvline(
                min_value,
                color="#cb181d",
                linewidth=1.0,
                linestyle=":",
                label=f"{clip_low:g}% clip",
            )
            ax.axvline(
                max_value,
                color="#cb181d",
                linewidth=1.0,
                linestyle="--",
                label=f"{clip_high:g}% clip",
            )
            ax.set_xlim(min_value, max_value)
        ax.axvline(mean, color="#08519c", linewidth=1.5, label="mean")
        ax.axvline(
            median,
            color="#f16913",
            linewidth=1.4,
            linestyle="--",
            label="median",
        )
        ax.set_title(f"u = {u_val:.1f}")
        ax.grid(True, alpha=0.25)

    for ax in axes_arr[u_arr.size :]:
        ax.set_visible(False)
    for ax in axes_arr[: u_arr.size]:
        ax.set_xlabel("Elasticity (d p_accept / du)")
        ax.set_ylabel("Customers")
    axes_arr[0].legend()
    title = "Customer elasticity distributions at selected u values"
    if clipped:
        title += f" (x-axis clipped to {clip_low:g}-{clip_high:g}%)"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    path = output_dir / "elasticity_histograms_by_u.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GLM customer elasticity by u and selected-u elasticity histograms."
    )
    parser.add_argument(
        "--u-min",
        type=float,
        default=-0.3,
        help="Minimum u for the dense curve.",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=0.3,
        help="Maximum u for the dense curve.",
    )
    parser.add_argument(
        "--u-count",
        type=int,
        default=121,
        help="Number of evenly spaced u values.",
    )
    parser.add_argument(
        "--hist-u",
        type=float,
        nargs="+",
        default=list(DEFAULT_HIST_U_VALUES),
        help="Selected u values for customer elasticity histograms.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Optional sampled row count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when --n-rows samples rows.",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count.")
    parser.add_argument(
        "--u-coef",
        type=float,
        default=None,
        help="Optional GLM u coefficient override.",
    )
    parser.add_argument(
        "--hist-clip-low",
        type=float,
        default=0.5,
        help="Lower percentile for histogram x-axis clipping. Defaults to 0.5.",
    )
    parser.add_argument(
        "--hist-clip-high",
        type=float,
        default=99.5,
        help="Upper percentile for histogram x-axis clipping. Defaults to 99.5.",
    )
    parser.add_argument(
        "--no-hist-clip",
        action="store_true",
        help="Disable histogram x-axis percentile clipping.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under --output-root. Defaults to elasticity_distribution_<timestamp>.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    u_values = _resolve_u_grid(args.u_min, args.u_max, args.u_count)
    hist_u_values = np.asarray(args.hist_u, dtype=float).reshape(-1)
    if hist_u_values.size == 0 or not np.isfinite(hist_u_values).all():
        raise ValueError("--hist-u values must be non-empty and finite.")

    row_indices = _resolve_row_indices(args.n_rows, args.seed)
    acceptance_model, _ = load_model_artifacts("glm")
    x_frame = load_x_frame("glm", row_indices=row_indices)
    derivative_bound = _theoretical_derivative_bound(acceptance_model, args.u_coef)
    elasticity_matrix = glm_price_derivative_matrix(
        acceptance_model,
        x_frame,
        u_values=u_values,
        u_coef=args.u_coef,
    )
    selected_derivatives = glm_price_derivative_matrix(
        acceptance_model,
        x_frame,
        u_values=hist_u_values,
        u_coef=args.u_coef,
    )

    summary_rows = _summary_rows(u_values, elasticity_matrix)
    selected_rows = _summary_rows(hist_u_values, selected_derivatives)
    output_subdir = args.output_subdir or (
        f"elasticity_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_csv = output_dir / "glm_elasticity_by_u.csv"
    selected_csv = output_dir / "glm_selected_u_elasticity_summary.csv"
    _write_summary_csv(summary_rows, dense_csv)
    _write_summary_csv(selected_rows, selected_csv)
    curve_path = _plot_mean_elasticity_by_u(
        summary_rows,
        output_dir,
        derivative_bound=derivative_bound,
    )
    clip_low = None if args.no_hist_clip else float(args.hist_clip_low)
    clip_high = None if args.no_hist_clip else float(args.hist_clip_high)
    hist_path = _plot_selected_u_histograms(
        hist_u_values,
        selected_derivatives,
        output_dir,
        bins=int(args.bins),
        clip_low=clip_low,
        clip_high=clip_high,
        derivative_bound=derivative_bound,
    )

    mean_values = np.asarray([float(row["mean"]) for row in summary_rows], dtype=float)
    peak_idx = int(
        np.argmin(mean_values) if derivative_bound < 0.0 else np.argmax(mean_values)
    )
    print(
        f"Computed GLM elasticities for {elasticity_matrix.shape[0]} rows "
        f"and {elasticity_matrix.shape[1]} u values."
    )
    print(
        f"Most negative average elasticity at u={float(u_values[peak_idx]):.6f}: "
        f"{mean_values[peak_idx]:.6f}."
    )
    print(f"Theoretical signed derivative bound is {derivative_bound:.6f}.")
    print(f"Wrote elasticity curve summary to {dense_csv}.")
    print(f"Wrote selected-u elasticity summary to {selected_csv}.")
    print(f"Wrote elasticity curve to {curve_path}.")
    print(f"Wrote selected-u elasticity histograms to {hist_path}.")


if __name__ == "__main__":
    main()
