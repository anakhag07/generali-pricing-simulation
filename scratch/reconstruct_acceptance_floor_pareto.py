"""Reconstruct the first-order acceptance-floor Pareto plot as a vector PDF.

The original ``acceptance_floor_sweep.csv`` is no longer available locally.  The
acceptance floors come from this repository's retired sweep driver, while the
plotted acceptance/objective coordinates were digitized from the supplied PNG.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


FRONTIER = (
    (0.500, 0.556046671, 37.094104780),
    (0.550, 0.600000294, 36.951539377),
    (0.600, 0.650171747, 35.525885348),
    (0.650, 0.699914384, 33.529969706),
    (0.700, 0.750300245, 30.393530841),
    (0.750, 0.800471697, 25.974003349),
    (0.800, 0.840565978, 20.841648842),
    (0.840, 0.871655126, 15.994425141),
    (0.870, 0.892881510, 11.574897649),
    (0.890, 0.909819736, 7.155370157),
    (0.910, 0.925685921, 2.450711859),
    (0.925, 0.939193620, -2.539077245),
    (0.940, 0.949914016, -7.813997154),
    (0.950, 0.959562372, -13.801744079),
    (0.960, 0.969210729, -21.215145033),
    (0.970, 0.977787045, -29.483938406),
    (0.978, 0.986148954, -38.750689598),
    (0.985, 0.991294744, -48.017440791),
    (0.990, 0.995154087, -58.139584402),
    (0.993, 0.997083758, -67.834031804),
    (0.995, 0.998370206, -73.679213325),
)


def write_frontier_csv(path: Path) -> None:
    """Write the reconstructed coordinates used by the figure."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("acceptance_floor", "mean_acceptance", "objective_value"))
        writer.writerows(FRONTIER)


def plot_frontier(output_dir: Path) -> None:
    """Render the reconstructed frontier to canonical PDF and PNG preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(FRONTIER, dtype=float)
    floors, mean_acceptance, objective_value = values.T

    norm = matplotlib.colors.Normalize(vmin=float(floors.min()), vmax=float(floors.max()))
    cmap = matplotlib.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mean_acceptance, objective_value, color="#1f77b4", alpha=0.25, linewidth=1.0)
    ax.scatter(
        mean_acceptance,
        objective_value,
        c=floors,
        cmap=cmap,
        norm=norm,
        marker="X",
        s=28.0,
        edgecolors="#1f77b4",
        linewidths=0.6,
        alpha=0.9,
    )

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array(floors)
    colorbar = fig.colorbar(scalar_mappable, ax=ax)
    colorbar.set_label("Acceptance Floor", fontsize=12)
    ax.set_xlabel("Mean Acceptance", fontsize=12)
    ax.set_ylabel("Expected Profit per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    colorbar.ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_objective_acceptance_first_order.pdf", format="pdf")
    fig.savefig(output_dir / "pareto_objective_acceptance_first_order.png", dpi=200)
    plt.close(fig)
    write_frontier_csv(output_dir / "digitized_frontier.csv")


if __name__ == "__main__":
    plot_frontier(Path("outputs/acceptance-floor-pareto-reconstruction"))
