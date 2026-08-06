"""Create an executive-ready idealized pricing cost/profit surface.

The portfolio cost follows the production sign convention

    cost = sum_s weight_s * f_accept,s * (gross_loss_s - revenue_s)

for two illustrative customer segments.  Expected profit is ``-cost``.  The
two segment price changes provide a clean two-dimensional decision surface
with a unique, interior optimum; no production data or fitted artifacts are
used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


INK = "#15324B"
MUTED = "#607386"
GRID = "#DCE5EC"
ACCENT = "#F06449"
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "executive_blue",
    ["#E9F2F7", "#9BC6D7", "#3B91AE", "#155E7A", "#0B3D56"],
)


@dataclass(frozen=True)
class Segment:
    """Idealized inputs for one pricing segment."""

    name: str
    weight: float
    baseline_acceptance: float
    price_elasticity: float
    baseline_premium: float
    gross_loss: float


SEGMENTS = (
    Segment(
        name="Segment A",
        weight=0.58,
        baseline_acceptance=0.82,
        price_elasticity=8.0,
        baseline_premium=1_000.0,
        gross_loss=720.0,
    ),
    Segment(
        name="Segment B",
        weight=0.42,
        baseline_acceptance=0.72,
        price_elasticity=6.2,
        baseline_premium=880.0,
        gross_loss=650.0,
    ),
)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    price_change = np.linspace(args.price_min, args.price_max, args.grid_size)
    change_a, change_b = np.meshgrid(price_change, price_change)
    cost, acceptance, revenue, gross_loss = portfolio_cost(change_a, change_b)
    profit = -cost

    peak_index = np.unravel_index(int(np.argmax(profit)), profit.shape)
    peak_a = float(change_a[peak_index])
    peak_b = float(change_b[peak_index])
    peak_profit = float(profit[peak_index])
    peak_cost = float(cost[peak_index])

    plot_profit_surface(
        output_dir / "idealized_profit_surface_3d.png",
        change_a,
        change_b,
        profit,
        peak_a,
        peak_b,
        peak_profit,
        dpi=args.dpi,
    )
    plot_profit_contour(
        output_dir / "idealized_profit_contour.png",
        change_a,
        change_b,
        profit,
        peak_a,
        peak_b,
        peak_profit,
        dpi=args.dpi,
    )
    plot_cost_surface(
        output_dir / "idealized_cost_surface_3d.png",
        change_a,
        change_b,
        cost,
        peak_a,
        peak_b,
        peak_cost,
        dpi=args.dpi,
    )
    plot_executive_combined(
        output_dir / "idealized_cost_surface_executive.png",
        change_a,
        change_b,
        profit,
        peak_a,
        peak_b,
        peak_profit,
        dpi=args.dpi,
    )

    # Keep the presentation data lightweight: roughly 61 points per axis even
    # when the rendering grid is much denser.
    csv_stride = max(1, (args.grid_size - 1) // 60)
    csv_slice = np.s_[::csv_stride, ::csv_stride]
    np.savetxt(
        output_dir / "idealized_cost_surface.csv",
        np.column_stack(
            [
                change_a[csv_slice].ravel(),
                change_b[csv_slice].ravel(),
                acceptance[csv_slice].ravel(),
                revenue[csv_slice].ravel(),
                gross_loss[csv_slice].ravel(),
                cost[csv_slice].ravel(),
                profit[csv_slice].ravel(),
            ]
        ),
        delimiter=",",
        header=(
            "segment_a_price_change,segment_b_price_change,portfolio_acceptance,"
            "accepted_revenue,accepted_gross_loss,cost,expected_profit"
        ),
        comments="",
    )
    metadata = {
        "formula": "cost = sum(weight * f_accept * (gross_loss - revenue))",
        "profit_formula": "expected_profit = -cost",
        "rendering_grid_size": args.grid_size,
        "csv_grid_size": int(change_a[csv_slice].shape[0]),
        "optimum": {
            "segment_a_price_change": peak_a,
            "segment_b_price_change": peak_b,
            "expected_profit": peak_profit,
            "cost": peak_cost,
        },
        "segments": [asdict(segment) for segment in SEGMENTS],
    }
    (output_dir / "idealized_cost_surface_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Peak expected profit: ${peak_profit:,.0f} per policy")
    print(f"Segment A price change: {peak_a:+.1%}")
    print(f"Segment B price change: {peak_b:+.1%}")
    print(f"Wrote executive plots to: {output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("idealized_cost_surface"),
        help="Output directory (default: scratch/idealized_cost_surface).",
    )
    parser.add_argument("--price-min", type=float, default=-0.10)
    parser.add_argument("--price-max", type=float, default=0.30)
    parser.add_argument("--grid-size", type=int, default=241)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.price_min >= args.price_max:
        parser.error("--price-min must be below --price-max")
    if args.grid_size < 25:
        parser.error("--grid-size must be at least 25")
    return args


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    result = np.empty_like(z_arr)
    positive = z_arr >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-z_arr[positive]))
    exp_z = np.exp(z_arr[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def segment_metrics(price_change: np.ndarray, segment: Segment) -> tuple[np.ndarray, ...]:
    """Return acceptance, revenue, loss, and cost for a segment."""
    baseline_logit = np.log(segment.baseline_acceptance / (1.0 - segment.baseline_acceptance))
    acceptance = _sigmoid(baseline_logit - segment.price_elasticity * price_change)
    revenue = segment.baseline_premium * (1.0 + price_change)
    gross_loss = np.full_like(revenue, segment.gross_loss, dtype=float)
    cost = acceptance * (gross_loss - revenue)
    return acceptance, revenue, gross_loss, cost


def portfolio_cost(
    price_change_a: np.ndarray,
    price_change_b: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return weighted portfolio cost and its presentation components."""
    metrics_a = segment_metrics(price_change_a, SEGMENTS[0])
    metrics_b = segment_metrics(price_change_b, SEGMENTS[1])
    weight_a, weight_b = SEGMENTS[0].weight, SEGMENTS[1].weight

    acceptance = weight_a * metrics_a[0] + weight_b * metrics_b[0]
    accepted_revenue = weight_a * metrics_a[0] * metrics_a[1] + weight_b * metrics_b[0] * metrics_b[1]
    accepted_loss = weight_a * metrics_a[0] * metrics_a[2] + weight_b * metrics_b[0] * metrics_b[2]
    cost = weight_a * metrics_a[3] + weight_b * metrics_b[3]
    return cost, acceptance, accepted_revenue, accepted_loss


def _style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")


def _style_3d_axis(ax: plt.Axes) -> None:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID)
    ax.yaxis.pane.set_edgecolor(GRID)
    ax.zaxis.pane.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6, linestyle="-")


def _percent_axes(ax: plt.Axes) -> None:
    formatter = ticker.PercentFormatter(xmax=1.0, decimals=0)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def _slide_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.965, title, ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.055, 0.905, subtitle, ha="left", va="top", fontsize=11, color=MUTED)


def _footer(fig: plt.Figure) -> None:
    fig.text(
        0.055,
        0.025,
        "Illustrative, idealized portfolio — not calibrated to production data",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )


def plot_profit_surface(
    path: Path,
    change_a: np.ndarray,
    change_b: np.ndarray,
    profit: np.ndarray,
    peak_a: float,
    peak_b: float,
    peak_profit: float,
    *,
    dpi: int,
) -> None:
    """Plot the slide-ready expected-profit hill."""
    fig = plt.figure(figsize=(13.333, 7.5))
    _style_figure(fig)
    ax = fig.add_axes((0.04, 0.09, 0.90, 0.76), projection="3d")
    stride = max(1, profit.shape[0] // 80)
    ax.plot_surface(
        change_a,
        change_b,
        profit,
        cmap=SURFACE_CMAP,
        rstride=stride,
        cstride=stride,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    floor = float(np.min(profit))
    ax.contour(change_a, change_b, profit, levels=12, zdir="z", offset=floor, cmap=SURFACE_CMAP, linewidths=0.7)
    ax.scatter([peak_a], [peak_b], [peak_profit], s=85, color=ACCENT, edgecolor="white", linewidth=1.4, depthshade=False)
    ax.text(
        peak_a,
        peak_b,
        peak_profit + 8.0,
        f"  Peak  ${peak_profit:,.0f}",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )
    ax.set_zlim(floor, float(np.max(profit)) + 18.0)
    ax.set_xlabel("Segment A price change", labelpad=12, color=INK)
    ax.set_ylabel("Segment B price change", labelpad=12, color=INK)
    ax.set_zlabel("Expected profit / policy", labelpad=10, color=INK)
    ax.zaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    _percent_axes(ax)
    _style_3d_axis(ax)
    ax.view_init(elev=27, azim=-128)
    ax.set_box_aspect((1.45, 1.0, 0.62))
    _slide_title(
        fig,
        "Pricing has a clear economic optimum",
        r"Expected profit = $-\,f_{accept}\,(gross\ loss - revenue)$; higher is better",
    )
    _footer(fig)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_profit_contour(
    path: Path,
    change_a: np.ndarray,
    change_b: np.ndarray,
    profit: np.ndarray,
    peak_a: float,
    peak_b: float,
    peak_profit: float,
    *,
    dpi: int,
) -> None:
    """Plot a clean top-down contour of the expected-profit surface."""
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    _style_figure(fig)
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.14, top=0.82)
    filled = ax.contourf(change_a, change_b, profit, levels=18, cmap=SURFACE_CMAP)
    lines = ax.contour(change_a, change_b, profit, levels=10, colors="white", alpha=0.62, linewidths=0.8)
    ax.clabel(lines, inline=True, fontsize=8, fmt=lambda value: f"${value:,.0f}")
    ax.scatter([peak_a], [peak_b], marker="*", s=250, color=ACCENT, edgecolor="white", linewidth=1.4, zorder=4)
    ax.axvline(peak_a, color=ACCENT, linewidth=1.1, linestyle=(0, (3, 3)), alpha=0.85)
    ax.axhline(peak_b, color=ACCENT, linewidth=1.1, linestyle=(0, (3, 3)), alpha=0.85)
    ax.annotate(
        f"Peak expected profit\n${peak_profit:,.0f} per policy\nA {peak_a:+.1%}  |  B {peak_b:+.1%}",
        xy=(peak_a, peak_b),
        xytext=(-0.065, 0.225),
        ha="left",
        fontsize=11,
        color=INK,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "alpha": 0.96},
        arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 1.4},
    )
    ax.set_xlabel("Segment A price change", fontsize=11, color=INK, labelpad=10)
    ax.set_ylabel("Segment B price change", fontsize=11, color=INK, labelpad=10)
    ax.tick_params(colors=MUTED)
    _percent_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(filled, ax=ax, pad=0.035, fraction=0.035)
    colorbar.set_label("Expected profit / policy", color=INK, labelpad=10)
    colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    colorbar.ax.tick_params(colors=MUTED)
    _slide_title(
        fig,
        "The optimum is stable across both pricing levers",
        "Closed contour bands make the peak and the value trade-off easy to read",
    )
    _footer(fig)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_cost_surface(
    path: Path,
    change_a: np.ndarray,
    change_b: np.ndarray,
    cost: np.ndarray,
    optimum_a: float,
    optimum_b: float,
    minimum_cost: float,
    *,
    dpi: int,
) -> None:
    """Plot the literal minimized cost surface requested by the model formula."""
    fig = plt.figure(figsize=(13.333, 7.5))
    _style_figure(fig)
    ax = fig.add_axes((0.04, 0.09, 0.90, 0.76), projection="3d")
    stride = max(1, cost.shape[0] // 80)
    ax.plot_surface(
        change_a,
        change_b,
        cost,
        cmap=SURFACE_CMAP.reversed(),
        rstride=stride,
        cstride=stride,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax.scatter(
        [optimum_a],
        [optimum_b],
        [minimum_cost],
        s=85,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.4,
        depthshade=False,
    )
    ax.text2D(
        0.68,
        0.28,
        f"MINIMUM COST\n${minimum_cost:,.0f} / policy\nA {optimum_a:+.1%}  |  B {optimum_b:+.1%}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": GRID, "alpha": 0.94},
    )
    ax.set_xlabel("Segment A price change", labelpad=12, color=INK)
    ax.set_ylabel("Segment B price change", labelpad=12, color=INK)
    ax.set_zlabel("Cost / policy", labelpad=10, color=INK)
    ax.zaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    _percent_axes(ax)
    _style_3d_axis(ax)
    ax.view_init(elev=27, azim=-128)
    ax.set_box_aspect((1.45, 1.0, 0.62))
    _slide_title(
        fig,
        "A clear minimum in the model objective",
        r"Cost = $f_{accept}\,(gross\ loss - revenue)$; lower is better",
    )
    _footer(fig)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_executive_combined(
    path: Path,
    change_a: np.ndarray,
    change_b: np.ndarray,
    profit: np.ndarray,
    peak_a: float,
    peak_b: float,
    peak_profit: float,
    *,
    dpi: int,
) -> None:
    """Plot a 16:9 two-panel surface-and-contour executive slide."""
    fig = plt.figure(figsize=(13.333, 7.5))
    _style_figure(fig)
    grid = fig.add_gridspec(1, 2, left=0.045, right=0.96, bottom=0.12, top=0.82, width_ratios=(1.12, 0.88), wspace=0.10)
    ax_surface = fig.add_subplot(grid[0, 0], projection="3d")
    ax_contour = fig.add_subplot(grid[0, 1])

    stride = max(1, profit.shape[0] // 70)
    ax_surface.plot_surface(
        change_a,
        change_b,
        profit,
        cmap=SURFACE_CMAP,
        rstride=stride,
        cstride=stride,
        linewidth=0,
        antialiased=True,
        alpha=0.97,
    )
    ax_surface.scatter(
        [peak_a], [peak_b], [peak_profit], s=75, color=ACCENT, edgecolor="white", linewidth=1.3, depthshade=False
    )
    ax_surface.set_xlabel("Segment A", labelpad=8, color=INK)
    ax_surface.set_ylabel("Segment B", labelpad=8, color=INK)
    ax_surface.set_zlabel("Profit / policy", labelpad=7, color=INK)
    ax_surface.zaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    _percent_axes(ax_surface)
    _style_3d_axis(ax_surface)
    ax_surface.view_init(elev=28, azim=-128)
    ax_surface.set_box_aspect((1.25, 1.0, 0.64))

    filled = ax_contour.contourf(change_a, change_b, profit, levels=18, cmap=SURFACE_CMAP)
    ax_contour.contour(change_a, change_b, profit, levels=10, colors="white", alpha=0.52, linewidths=0.7)
    ax_contour.scatter([peak_a], [peak_b], marker="*", s=210, color=ACCENT, edgecolor="white", linewidth=1.2)
    ax_contour.axvline(peak_a, color=ACCENT, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.85)
    ax_contour.axhline(peak_b, color=ACCENT, linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.85)
    ax_contour.set_xlabel("Segment A price change", color=INK, labelpad=8)
    ax_contour.set_ylabel("Segment B price change", color=INK, labelpad=8)
    ax_contour.tick_params(colors=MUTED, labelsize=9)
    _percent_axes(ax_contour)
    ax_contour.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(filled, ax=ax_contour, pad=0.03, fraction=0.05)
    colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)

    fig.text(
        0.77,
        0.72,
        f"PEAK\n${peak_profit:,.0f} / policy\nA {peak_a:+.1%}  |  B {peak_b:+.1%}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": GRID, "alpha": 0.95},
    )
    _slide_title(
        fig,
        "Pricing has a clear economic optimum",
        r"Maximum expected profit, equivalent to minimizing $f_{accept}\,(gross\ loss - revenue)$",
    )
    _footer(fig)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
