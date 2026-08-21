"""Regenerate saved model-effect figures as vector PDFs.

The task reads the collected acceptance-by-price and bonus-malus mean/standard-
deviation CSVs.  It does not rerun model inference.  By default it targets the
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
ACCEPTANCE_COLOR = "tab:blue"
CLAIMS_COLOR = "tab:red"
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
            "bonus_malus_mean_std.csv."
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


def _plot_mean_std(
    x: pd.Series | np.ndarray,
    mean: pd.Series | np.ndarray,
    std: pd.Series | np.ndarray,
    *,
    color: str,
    xlabel: str,
    ylabel: str,
    title: str,
    pdf_path: Path,
    png_path: Path,
    ylim: tuple[float, float] | None = None,
    add_y_margin: bool = False,
    probability: bool = False,
) -> None:
    x_values = np.asarray(x, dtype=float)
    mean_values = np.asarray(mean, dtype=float)
    std_values = np.asarray(std, dtype=float)
    if (
        x_values.ndim != 1
        or mean_values.shape != x_values.shape
        or std_values.shape != x_values.shape
        or x_values.size < 2
    ):
        raise ValueError(f"Invalid one-dimensional curve for {pdf_path.name}")
    if (
        not np.isfinite(x_values).all()
        or not np.isfinite(mean_values).all()
        or not np.isfinite(std_values).all()
        or np.any(std_values < 0.0)
    ):
        raise ValueError(f"Invalid mean/SD values for {pdf_path.name}")

    order = np.argsort(x_values)
    x_values = x_values[order]
    mean_values = mean_values[order]
    std_values = std_values[order]
    lower = mean_values - std_values
    upper = mean_values + std_values
    if probability:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.fill_between(
        x_values,
        lower,
        upper,
        color=color,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(x_values, mean_values, color=color, linewidth=3)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if add_y_margin:
        ax.margins(y=0.12)
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
        color=ACCEPTANCE_COLOR,
        alpha=0.20,
        linewidth=0,
    )
    ax.plot(
        u,
        smooth_mean,
        color=ACCEPTANCE_COLOR,
        linewidth=3,
    )
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Acceptance Probability", fontsize=12)
    ax.set_title(
        "Predicted Effect of Price Change on Acceptance Probability",
        fontsize=16,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
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
    bonus_malus = _read_csv(
        analysis_dir / "bonus_malus_mean_std.csv",
        {"model", "target", "bonus_malus_rating", "mean", "std"},
    )

    written: list[Path] = []
    for model in MODEL_ORDER:
        price_curve = acceptance_by_u.loc[acceptance_by_u["model"].eq(model)]
        if price_curve.empty:
            raise ValueError(f"acceptance_by_u.csv has no rows for model={model!r}")
        price_output = output_dir / f"{model}_acceptance_mean_std.pdf"
        _plot_mean_std(
            price_curve["u"],
            price_curve["mean"],
            price_curve["std"],
            color=ACCEPTANCE_COLOR,
            xlabel="Proposed Price Change",
            ylabel="Acceptance Probability",
            title="Predicted Effect of Price Change on Acceptance Probability",
            pdf_path=price_output,
            png_path=output_dir / f"{model}_acceptance_mean_std.png",
            ylim=ACCEPTANCE_YLIMS[model],
            probability=True,
        )
        written.append(price_output)

        acceptance_curve = bonus_malus.loc[
            bonus_malus["model"].eq(model)
            & bonus_malus["target"].eq("acceptance")
        ]
        if acceptance_curve.empty:
            raise ValueError(f"bonus_malus_mean_std.csv has no {model} acceptance rows")
        acceptance_output = output_dir / f"{model}_bonus_malus_vs_acceptance.pdf"
        _plot_mean_std(
            acceptance_curve["bonus_malus_rating"],
            acceptance_curve["mean"],
            acceptance_curve["std"],
            color=ACCEPTANCE_COLOR,
            xlabel="Bonus-Malus Rating",
            ylabel="Acceptance Probability",
            title="Predicted Effect of Bonus-Malus Rating on Acceptance Probability",
            pdf_path=acceptance_output,
            png_path=output_dir / f"{model}_bonus_malus_vs_acceptance.png",
            add_y_margin=True,
            probability=True,
        )
        written.append(acceptance_output)

        claims_curve = bonus_malus.loc[
            bonus_malus["model"].eq(model)
            & bonus_malus["target"].eq("claims")
        ]
        if claims_curve.empty:
            raise ValueError(f"bonus_malus_mean_std.csv has no {model} claims rows")
        claims_output = output_dir / f"{model}_bonus_malus_vs_claims.pdf"
        _plot_mean_std(
            claims_curve["bonus_malus_rating"],
            claims_curve["mean"],
            claims_curve["std"],
            color=CLAIMS_COLOR,
            xlabel="Bonus-Malus Rating",
            ylabel="Predicted Claims",
            title="Predicted Effect of Bonus-Malus Rating on Claims",
            pdf_path=claims_output,
            png_path=output_dir / f"{model}_bonus_malus_vs_claims.png",
            add_y_margin=True,
        )
        written.append(claims_output)

    xgb_curve = acceptance_by_u.loc[acceptance_by_u["model"].eq("xgb")]
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
