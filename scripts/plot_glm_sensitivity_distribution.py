"""Plot GLM customer price-sensitivity distributions across action values."""

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
    load_model_artifacts,
    load_x_frame,
    sample_csv_row_indices,
)
from experiments.sensitivity_buckets import glm_price_sensitivity_matrix

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
    sensitivity_matrix: np.ndarray,
) -> list[dict[str, float | int]]:
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    values = np.asarray(sensitivity_matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("sensitivity_matrix must be 2D.")
    if values.shape[1] != u_arr.size:
        raise ValueError("u_values length must match sensitivity_matrix columns.")
    if values.shape[0] == 0:
        raise ValueError("sensitivity_matrix must contain at least one customer row.")

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


def _plot_mean_sensitivity_by_u(
    rows: Sequence[Mapping[str, float | int]],
    output_dir: Path,
) -> Path:
    if not rows:
        raise ValueError("At least one summary row is required to plot sensitivity by u.")
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
    ax.plot(u_values, mean, color="#08519c", linewidth=2.0, label="mean")
    ax.plot(u_values, median, color="#f16913", linewidth=1.8, linestyle="--", label="median")
    ax.set_xlabel("u")
    ax.set_ylabel("Customer sensitivity |d p_accept / du|")
    ax.set_title("GLM customer price sensitivity by action value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "mean_sensitivity_by_u.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_selected_u_histograms(
    hist_u_values: Sequence[float],
    selected_sensitivities: np.ndarray,
    output_dir: Path,
    *,
    bins: int,
) -> Path:
    u_arr = np.asarray(hist_u_values, dtype=float).reshape(-1)
    values = np.asarray(selected_sensitivities, dtype=float)
    if values.ndim != 2:
        raise ValueError("selected_sensitivities must be 2D.")
    if values.shape[1] != u_arr.size:
        raise ValueError("hist_u_values length must match selected_sensitivities columns.")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("selected_sensitivities must be non-empty.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
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
    max_value = float(np.max(values))
    hist_bins: int | np.ndarray = (
        bins if max_value <= 0.0 else np.linspace(0.0, max_value, bins + 1)
    )

    for ax, u_val, column in zip(axes_arr, u_arr, values.T):
        mean = float(np.mean(column))
        median = float(np.median(column))
        ax.hist(column, bins=hist_bins, color="#9ecae1", edgecolor="#6baed6", alpha=0.82)
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
        ax.set_xlabel("|d p_accept / du|")
        ax.set_ylabel("Customers")
    axes_arr[0].legend()
    fig.suptitle("Customer sensitivity distributions at selected u values", y=1.02)
    fig.tight_layout()
    path = output_dir / "sensitivity_histograms_by_u.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot GLM customer sensitivity by u and selected-u sensitivity histograms."
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
        help="Selected u values for customer sensitivity histograms.",
    )
    parser.add_argument("--n-rows", type=int, default=None, help="Optional sampled row count.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when --n-rows samples rows.",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bin count.")
    parser.add_argument("--u-coef", type=float, default=None, help="Optional GLM u coefficient override.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under --output-root. Defaults to sensitivity_distribution_<timestamp>.",
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
    sensitivity_matrix = glm_price_sensitivity_matrix(
        acceptance_model,
        x_frame,
        u_values=u_values,
        u_coef=args.u_coef,
    )
    selected_sensitivities = glm_price_sensitivity_matrix(
        acceptance_model,
        x_frame,
        u_values=hist_u_values,
        u_coef=args.u_coef,
    )

    summary_rows = _summary_rows(u_values, sensitivity_matrix)
    selected_rows = _summary_rows(hist_u_values, selected_sensitivities)
    output_subdir = args.output_subdir or (
        f"sensitivity_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_csv = output_dir / "glm_sensitivity_by_u.csv"
    selected_csv = output_dir / "glm_selected_u_sensitivity_summary.csv"
    _write_summary_csv(summary_rows, dense_csv)
    _write_summary_csv(selected_rows, selected_csv)
    curve_path = _plot_mean_sensitivity_by_u(summary_rows, output_dir)
    hist_path = _plot_selected_u_histograms(
        hist_u_values,
        selected_sensitivities,
        output_dir,
        bins=int(args.bins),
    )

    mean_values = np.asarray([float(row["mean"]) for row in summary_rows], dtype=float)
    peak_idx = int(np.argmax(mean_values))
    print(
        f"Computed GLM sensitivities for {sensitivity_matrix.shape[0]} rows "
        f"and {sensitivity_matrix.shape[1]} u values."
    )
    print(
        f"Peak average sensitivity at u={float(u_values[peak_idx]):.6f}: "
        f"{mean_values[peak_idx]:.6f}."
    )
    print(f"Wrote sensitivity summaries to {dense_csv} and {selected_csv}.")
    print(f"Wrote sensitivity curve to {curve_path}.")
    print(f"Wrote selected-u histograms to {hist_path}.")


if __name__ == "__main__":
    main()
