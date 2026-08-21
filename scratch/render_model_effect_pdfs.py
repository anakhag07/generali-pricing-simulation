"""Regenerate saved model-effect figures as vector PDFs.

The task reads the collected acceptance-by-price and bonus-malus partial-
dependence CSVs.  It does not rerun model inference.  By default it targets the
canonical ``20260809_105106`` analysis sweep; pass ``--analysis-dir`` to reuse
the renderer with another collected sweep that has the same CSV schema.

Example
-------
python scratch/render_model_effect_pdfs.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "GENERALI_RESULTS_ROOT",
        Path.home() / "projects" / "generali-pricing" / "results",
    )
)
DEFAULT_ANALYSIS_DIR = (
    DEFAULT_RESULTS_ROOT
    / "model-acceptance-feature-analysis"
    / "sweeps"
    / "20260809_105106"
)

MODEL_ORDER = ("glm", "spline", "xgb")
MODEL_COLORS = {
    "glm": "#2166ac",
    "spline": "#1b7837",
    "xgb": "#b2182b",
}
ACCEPTANCE_YLIMS = {
    "glm": (0.70, 0.98),
    "spline": (0.80, 0.98),
    "xgb": (0.60, 0.98),
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help=(
            "Collected analysis directory containing acceptance_by_u.csv and "
            "the three bonus-malus partial-dependence CSVs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PDF destination; defaults to --analysis-dir.",
    )
    parser.add_argument(
        "--xgb-smoothing-bandwidth",
        type=float,
        default=0.005,
        help=(
            "Gaussian bandwidth in price-change units for the smoothed XGBoost "
            "mean/SD figure (default: 0.005)."
        ),
    )
    args = parser.parse_args(argv)
    if args.xgb_smoothing_bandwidth <= 0.0:
        parser.error("--xgb-smoothing-bandwidth must be positive")
    return args


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required plot data not found: {path}")
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _plot_curve(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    color: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
    add_y_margin: bool = False,
) -> None:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.shape != x_values.shape or x_values.size < 2:
        raise ValueError(f"Invalid one-dimensional curve for {output_path.name}")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError(f"Non-finite values in curve for {output_path.name}")

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(x_values, y_values, color=color, linewidth=3)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if add_y_margin:
        ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _plot_xgb_mean_std(
    frame: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Plot pointwise XGBoost mean acceptance and customer-level variation."""
    ordered = frame.sort_values("u")
    u = ordered["u"].to_numpy(dtype=float)
    mean = ordered["mean"].to_numpy(dtype=float)
    std = ordered["std"].to_numpy(dtype=float)
    lower = np.clip(mean - std, 0.0, 1.0)
    upper = np.clip(mean + std, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        color=MODEL_COLORS["xgb"],
        alpha=0.20,
        linewidth=0,
        label="Mean ± 1 SD across customers",
    )
    ax.plot(
        u,
        mean,
        color=MODEL_COLORS["xgb"],
        linewidth=3,
        label="Mean acceptance",
    )
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Acceptance Probability", fontsize=12)
    ax.set_title(
        "XGBoost Acceptance by Price Change Across Customers",
        fontsize=16,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=180)
    plt.close(fig)


def _plot_smoothed_xgb_mean_std(
    frame: pd.DataFrame,
    *,
    bandwidth: float,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Gaussian-smooth the aggregate XGBoost mean and SD functions."""
    ordered = frame.sort_values("u")
    u = ordered["u"].to_numpy(dtype=float)
    mean = ordered["mean"].to_numpy(dtype=float)
    std = ordered["std"].to_numpy(dtype=float)
    steps = np.diff(u)
    grid_step = float(np.median(steps))
    if not np.allclose(steps, grid_step, rtol=1e-6, atol=1e-12):
        raise ValueError("Gaussian smoothing requires an evenly spaced u grid")

    sigma_grid_points = bandwidth / grid_step
    smooth_mean = np.clip(
        gaussian_filter1d(mean, sigma=sigma_grid_points, mode="nearest"),
        0.0,
        1.0,
    )
    smooth_std = np.maximum(
        gaussian_filter1d(std, sigma=sigma_grid_points, mode="nearest"),
        0.0,
    )
    lower = np.clip(smooth_mean - smooth_std, 0.0, 1.0)
    upper = np.clip(smooth_mean + smooth_std, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        color=MODEL_COLORS["xgb"],
        alpha=0.20,
        linewidth=0,
        label="Smoothed mean ± 1 SD across customers",
    )
    ax.plot(
        u,
        smooth_mean,
        color=MODEL_COLORS["xgb"],
        linewidth=3,
        label="Smoothed mean acceptance",
    )
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Acceptance Probability", fontsize=12)
    ax.set_title(
        "Smoothed XGBoost Acceptance by Price Change Across Customers",
        fontsize=16,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=180)
    plt.close(fig)


def render_model_effect_pdfs(
    analysis_dir: Path,
    output_dir: Path,
    *,
    xgb_smoothing_bandwidth: float = 0.005,
) -> list[Path]:
    """Render price and bonus-malus effect PDFs from one collected sweep."""
    analysis_dir = analysis_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    acceptance_by_u = _read_csv(
        analysis_dir / "acceptance_by_u.csv",
        {"model", "u", "mean", "std"},
    )
    bonus_malus = {
        "glm": _read_csv(
            analysis_dir / "glm_bonus_malus_partial_dependence.csv",
            {
                "bonus_malus_rating",
                "mean_acceptance_probability_at_u_0_08",
                "mean_predicted_loss",
            },
        ),
        "xgb": _read_csv(
            analysis_dir / "xgb_bonus_malus_partial_dependence.csv",
            {
                "bonus_malus_rating",
                "mean_acceptance_probability_at_u_0_08",
                "mean_predicted_loss",
            },
        ),
        "spline": _read_csv(
            analysis_dir / "spline_bonus_malus_partial_dependence.csv",
            {
                "bonus_malus_rating",
                "mean_acceptance_probability_at_u_0_08",
            },
        ),
    }

    written: list[Path] = []
    for model in MODEL_ORDER:
        price_curve = acceptance_by_u.loc[acceptance_by_u["model"].eq(model)]
        if price_curve.empty:
            raise ValueError(f"acceptance_by_u.csv has no rows for model={model!r}")
        price_output = output_dir / f"{model}_acceptance_mean_only.pdf"
        _plot_curve(
            price_curve["u"],
            price_curve["mean"],
            color=MODEL_COLORS[model],
            xlabel="Proposed Price Change",
            ylabel="Acceptance Probability",
            title="Predicted Effect of Price Change on Acceptance Probability",
            output_path=price_output,
            ylim=ACCEPTANCE_YLIMS[model],
        )
        written.append(price_output)

        acceptance_output = output_dir / f"{model}_bonus_malus_vs_acceptance.pdf"
        _plot_curve(
            bonus_malus[model]["bonus_malus_rating"],
            bonus_malus[model]["mean_acceptance_probability_at_u_0_08"],
            color=MODEL_COLORS[model],
            xlabel="Bonus-Malus Rating",
            ylabel="Acceptance Probability",
            title="Predicted Effect of Bonus-Malus Rating on Acceptance Probability",
            output_path=acceptance_output,
            add_y_margin=True,
        )
        written.append(acceptance_output)

        # The spline family shares the XGBoost loss artifact, so its loss curve
        # is sourced from the XGBoost partial-dependence table and recolored.
        loss_source = bonus_malus["xgb"] if model == "spline" else bonus_malus[model]
        loss_output = output_dir / f"{model}_bonus_malus_vs_loss.pdf"
        _plot_curve(
            loss_source["bonus_malus_rating"],
            loss_source["mean_predicted_loss"],
            color=MODEL_COLORS[model],
            xlabel="Bonus-Malus Rating",
            ylabel="Predicted Loss",
            title="Predicted Effect of Bonus-Malus Rating on Loss",
            output_path=loss_output,
            add_y_margin=True,
        )
        written.append(loss_output)

    xgb_curve = acceptance_by_u.loc[acceptance_by_u["model"].eq("xgb")]
    xgb_band_output = output_dir / "xgb_acceptance_mean_std.pdf"
    _plot_xgb_mean_std(
        xgb_curve,
        pdf_path=xgb_band_output,
        png_path=output_dir / "xgb_acceptance_mean_std.png",
    )
    written.append(xgb_band_output)

    xgb_smoothed_output = output_dir / "xgb_acceptance_mean_std_smoothed.pdf"
    _plot_smoothed_xgb_mean_std(
        xgb_curve,
        bandwidth=xgb_smoothing_bandwidth,
        pdf_path=xgb_smoothed_output,
        png_path=output_dir / "xgb_acceptance_mean_std_smoothed.png",
    )
    written.append(xgb_smoothed_output)

    return written


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir
    )
    written = render_model_effect_pdfs(
        analysis_dir,
        output_dir,
        xgb_smoothing_bandwidth=args.xgb_smoothing_bandwidth,
    )
    print(f"Wrote {len(written)} PDFs to {output_dir}")
    for path in written:
        print(path.name)


if __name__ == "__main__":
    main()
