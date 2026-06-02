"""Plot historical CSV action values against acceptance probability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.loader import dataset_csv_path
from reporting.visualization import (
    _LINE_WIDTH,
    _SCATTER_ALPHA,
    _SCATTER_SIZE,
    _binned_mean_line,
    _policy_output_histogram_bins,
)

DEFAULT_CSV_PATHS: dict[Literal["glm", "xgb"], Path] = {
    "glm": dataset_csv_path(),
    "xgb": dataset_csv_path(),
}
DEFAULT_OUTPUT_ROOT = Path("outputs") / "historical_acceptance"
DEFAULT_U_MAX = 0.14


def load_historical_acceptance_columns(
    csv_path: Path,
    *,
    u_col: str = "U",
    acceptance_col: str = "prob_acceptance",
) -> tuple[np.ndarray, np.ndarray]:
    """Load historical action and acceptance-probability columns from CSV."""
    try:
        df = pd.read_csv(csv_path, sep=";", usecols=[u_col, acceptance_col])
    except ValueError as exc:
        raise ValueError(
            f"CSV must contain columns '{u_col}' and '{acceptance_col}'."
        ) from exc

    values = df.loc[:, [u_col, acceptance_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if values.empty:
        raise ValueError("No finite historical U and acceptance rows are available to plot.")
    return (
        values[u_col].to_numpy(dtype=float),
        values[acceptance_col].to_numpy(dtype=float),
    )


def _sample_indices(n_rows: int, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or n_rows <= max_points:
        return np.arange(n_rows, dtype=int)
    if max_points <= 0:
        raise ValueError("max_points must be positive when provided.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_points, replace=False).astype(int))


def plot_historical_u_acceptance(
    u_values: np.ndarray,
    acceptance_values: np.ndarray,
    output_dir: Path,
    *,
    filename: str = "historical_u_acceptance_histogram.png",
    max_points: int | None = 5000,
    sample_seed: int = 0,
    u_max: float | None = DEFAULT_U_MAX,
) -> Path:
    """Write a historical-U histogram plus acceptance-vs-U scatter plot."""
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    acceptance_arr = np.asarray(acceptance_values, dtype=float).reshape(-1)
    if u_arr.shape != acceptance_arr.shape:
        raise ValueError("u_values and acceptance_values must have matching shapes.")
    if u_arr.size == 0:
        raise ValueError("At least one historical row is required to plot.")

    bins = _policy_output_histogram_bins([u_arr])
    centers, mean_acceptance = _binned_mean_line(u_arr, acceptance_arr, bins)
    scatter_idx = _sample_indices(u_arr.size, max_points, sample_seed)
    x_limits: tuple[float, float] | None = None
    if u_max is not None:
        u_max_val = float(u_max)
        if not np.isfinite(u_max_val):
            raise ValueError("u_max must be finite when provided.")
        u_min_val = float(np.min(u_arr))
        if u_min_val >= u_max_val:
            raise ValueError("u_max must be greater than the minimum historical U.")
        x_limits = (u_min_val, u_max_val)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), squeeze=False)
    hist_ax = axes[0, 0]
    scatter_ax = axes[0, 1]

    hist_ax.hist(
        u_arr,
        bins=bins,
        density=False,
        label="historical customers",
        color="#969696",
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
        mean_ax.legend(loc="upper right", fontsize="small")

    hist_ax.set_title("Historical U distribution")
    hist_ax.set_xlabel("Historical U")
    hist_ax.set_ylabel("Customer count")
    if x_limits is not None:
        hist_ax.set_xlim(*x_limits)
    hist_ax.grid(True, alpha=0.3)
    hist_ax.legend(loc="upper left", fontsize="small")

    scatter_ax.scatter(
        u_arr[scatter_idx],
        acceptance_arr[scatter_idx],
        color="#636363",
        alpha=_SCATTER_ALPHA,
        s=_SCATTER_SIZE,
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
    scatter_ax.set_title("Historical acceptance vs U")
    scatter_ax.set_xlabel("Historical U")
    scatter_ax.set_ylabel("Acceptance probability")
    scatter_ax.set_ylim(0.0, 1.0)
    if x_limits is not None:
        scatter_ax.set_xlim(*x_limits)
    scatter_ax.grid(True, alpha=0.3)
    scatter_ax.legend(fontsize="small")

    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_historical_acceptance_csv(
    csv_path: Path,
    output_dir: Path,
    *,
    u_col: str = "U",
    acceptance_col: str = "prob_acceptance",
    max_points: int | None = 5000,
    sample_seed: int = 0,
    u_max: float | None = DEFAULT_U_MAX,
) -> Path:
    """Load historical CSV columns and write the U/acceptance diagnostic plot."""
    u_values, acceptance_values = load_historical_acceptance_columns(
        csv_path,
        u_col=u_col,
        acceptance_col=acceptance_col,
    )
    return plot_historical_u_acceptance(
        u_values,
        acceptance_values,
        output_dir,
        max_points=max_points,
        sample_seed=sample_seed,
        u_max=u_max,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot historical CSV U distribution and acceptance probability."
    )
    parser.add_argument("--model-type", choices=("glm", "xgb"), default="glm")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Acceptance CSV path. Defaults to the selected model-type CSV.",
    )
    parser.add_argument("--u-col", default="U", help="Historical action column. Defaults to U.")
    parser.add_argument(
        "--acceptance-col",
        default="prob_acceptance",
        help="Acceptance probability column. Defaults to prob_acceptance.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum scatter points to draw. Histogram and binned line use all rows.",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for scatter downsampling.")
    parser.add_argument(
        "--u-max",
        type=float,
        default=DEFAULT_U_MAX,
        help="Shared upper x-axis limit for both panels. Defaults to 0.14.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for plots. Defaults to outputs/historical_acceptance.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under --output-root. Defaults to selected model type.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    csv_path = args.csv_path or DEFAULT_CSV_PATHS[args.model_type]
    output_dir = args.output_root / (args.output_subdir or args.model_type)
    output_path = plot_historical_acceptance_csv(
        csv_path,
        output_dir,
        u_col=args.u_col,
        acceptance_col=args.acceptance_col,
        max_points=args.max_points,
        sample_seed=args.sample_seed,
        u_max=args.u_max,
    )
    print(f"Wrote historical U acceptance plot to {output_path}.")


if __name__ == "__main__":
    main()
