"""Recreate the slide-ready smooth nonconcave objective landscape.

The surface is illustrative: it combines smooth Gaussian features with a
low-amplitude wave so that the objective has a clear global optimum and
several local optima.  It is not calibrated to production data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np


TITLE_SIZE = 20
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 11
ANNOTATION_SIZE = 12


def objective(theta_1: np.ndarray, theta_2: np.ndarray) -> np.ndarray:
    """Return a smooth, bounded, nonconcave illustrative profit surface."""
    broad_global_peak = 18.0 * np.exp(
        -0.62 * (theta_1 - 0.05) ** 2 - 0.72 * (theta_2 - 1.25) ** 2
    )
    left_local_peak = 10.5 * np.exp(
        -1.05 * (theta_1 + 1.65) ** 2 - 0.95 * (theta_2 + 1.45) ** 2
    )
    right_front_peak = 7.2 * np.exp(
        -0.90 * (theta_1 - 1.75) ** 2 - 0.95 * (theta_2 + 1.75) ** 2
    )
    right_back_peak = 7.0 * np.exp(
        -0.90 * (theta_1 - 2.10) ** 2 - 0.95 * (theta_2 - 2.10) ** 2
    )
    central_valley = -8.5 * np.exp(
        -0.85 * (theta_1 - 0.05) ** 2 - 1.10 * (theta_2 + 1.15) ** 2
    )
    right_valley = -7.0 * np.exp(
        -1.15 * (theta_1 - 1.75) ** 2 - 1.05 * (theta_2 - 0.15) ** 2
    )
    smooth_wave = 1.8 * np.cos(1.45 * theta_1 + 0.15) * np.cos(1.30 * theta_2 - 0.20)
    edge_taper = -0.46 * (theta_1**2 + 0.78 * theta_2**2)
    return (
        broad_global_peak
        + left_local_peak
        + right_front_peak
        + right_back_peak
        + central_valley
        + right_valley
        + smooth_wave
        + edge_taper
        - 0.5
    )


def _grid_maximum(
    theta_1: np.ndarray,
    theta_2: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    candidates = np.where(mask, values, -np.inf) if mask is not None else values
    index = np.unravel_index(int(np.argmax(candidates)), candidates.shape)
    return float(theta_1[index]), float(theta_2[index]), float(values[index])


def _annotate_projected(
    ax: plt.Axes,
    point: tuple[float, float, float],
    label: str,
    offset: tuple[float, float],
) -> None:
    x_2d, y_2d, _ = proj3d.proj_transform(*point, ax.get_proj())
    ax.annotate(
        label,
        xy=(x_2d, y_2d),
        xycoords="data",
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE,
        ha="left" if offset[0] >= 0 else "right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.5},
    )


def plot(output_dir: Path, *, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = np.linspace(-3.0, 3.0, 241)
    theta_1, theta_2 = np.meshgrid(axis, axis)
    profit = objective(theta_1, theta_2)

    global_maximum = _grid_maximum(theta_1, theta_2, profit)
    local_mask = (theta_1 < -0.8) & (theta_2 < -0.55)
    local_maximum = _grid_maximum(theta_1, theta_2, profit, local_mask)

    fig = plt.figure(figsize=(13.333, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(
        (0.02, 0.05, 0.94, 0.84),
        projection="3d",
        computed_zorder=False,
    )

    stride = max(1, profit.shape[0] // 85)
    ax.plot_surface(
        theta_1,
        theta_2,
        profit,
        cmap="viridis",
        rstride=stride,
        cstride=stride,
        linewidth=0.12,
        antialiased=True,
        alpha=0.98,
        zorder=1,
    )

    for point in (local_maximum, global_maximum):
        ax.scatter(
            [point[0]],
            [point[1]],
            [point[2] + 0.40],
            marker="*",
            s=260,
            color="black",
            edgecolor="white",
            linewidth=1.6,
            depthshade=False,
            zorder=20,
        )

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_xlabel(r"Decision Parameter $\theta_1$", fontsize=AXIS_LABEL_SIZE, labelpad=16, rotation=0)
    ax.set_ylabel(r"Decision Parameter $\theta_2$", fontsize=AXIS_LABEL_SIZE, labelpad=30, rotation=0)
    ax.set_zlabel("Expected Profit", fontsize=AXIS_LABEL_SIZE, labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, pad=2)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_zlim(-15.0, 16.0)
    ax.set_xticks(np.arange(-3, 4, 1))
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_zticks(np.arange(-15, 16, 5))
    # Keep the decision plane visually close to square; the third value controls
    # the relative height of the profit axis.
    ax.set_box_aspect((1.15, 1.0, 0.78))
    ax.view_init(elev=25, azim=-58)

    fig.canvas.draw()
    _annotate_projected(ax, local_maximum, "Local Maximum", (-70, 48))
    _annotate_projected(ax, global_maximum, "Best Decision Rule", (44, 34))

    fig.suptitle(
        "Objective Landscape: Expected Profit",
        x=0.50,
        y=0.96,
        fontsize=TITLE_SIZE,
    )

    png_path = output_dir / "smooth_nonconcave_expected_profit.png"
    pdf_path = output_dir / "smooth_nonconcave_expected_profit.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    print(f"Global maximum: theta_1={global_maximum[0]:+.3f}, theta_2={global_maximum[1]:+.3f}")
    print(f"Local maximum: theta_1={local_maximum[0]:+.3f}, theta_2={local_maximum[1]:+.3f}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("smooth_nonconcave_surface"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    plot(args.output_dir.expanduser().resolve(), dpi=args.dpi)


if __name__ == "__main__":
    main()
