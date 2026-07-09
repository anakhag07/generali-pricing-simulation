"""Extra diagnostic plots for the optax-adam noise-offset grids.

Reads the collector's ``noise_offset_grid_finals.csv`` for each family and adds
views the built-in per-estimator figure does not provide:

* single-axis slices along the **noise** axis (x = noise level, one curve per
  init offset) -- the transpose of the collector's offset-axis figure -- plus a
  matched offset-axis slice for symmetry;
* **noise x offset heatmaps** of final theta-distance-to-truth and clean-
  objective gap, annotated per cell, to showcase how the optimizer drifts from
  the clean first-order truth (i.e. exploits noise) as noise and init offset
  grow.

All metrics are means over run seeds of the clean-objective quantities already
computed by the collector (never the noisy ``final_value``). Regenerates from
saved CSVs only; never reruns optimization.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from experiments.paths import results_root  # noqa: E402


@dataclass(frozen=True)
class Family:
    key: str
    project_name: str
    axis_key: str  # CSV column for the noise level
    noise_symbol: str  # LaTeX symbol for the noise level
    noise_label: str  # axis label for the noise level


HOMO = Family(
    key="homoskedastic",
    project_name="homoskedastic-noise-offset-grid-optax",
    axis_key="noise_std",
    noise_symbol=r"\sigma",
    noise_label=r"homoskedastic noise std $\sigma$",
)
HETERO = Family(
    key="heteroskedastic",
    project_name="heteroskedastic-noise-offset-grid-optax",
    axis_key="noise_growth",
    noise_symbol=r"\gamma",
    noise_label=r"heteroskedastic noise growth $\gamma$",
)
FAMILIES = {"homoskedastic": (HOMO,), "heteroskedastic": (HETERO,), "all": (HOMO, HETERO)}

ESTIMATORS = ("finite_difference", "stein_difference")
ESTIMATOR_LABELS = {"finite_difference": "Finite difference", "stein_difference": "Stein difference"}
METRICS = (
    ("theta_distance_to_truth", r"$\|\hat{\theta}_{\mathrm{final}} - \theta^{\mathrm{FO}}_{\mathrm{clean}}\|_2$"),
    ("clean_objective_gap", r"$J_{\mathrm{clean}}(\hat{\theta}) - J_{\mathrm{clean}}(\theta^{\mathrm{FO}}_{\mathrm{clean}})$"),
)


# ---------------------------------------------------------------------------
# Data loading / aggregation
# ---------------------------------------------------------------------------

Key = tuple[float, float, str]  # (noise_level, offset, estimator)


def _load_rows(family: Family) -> list[dict[str, object]]:
    csv_path = results_root() / family.project_name / "noise_offset_grid_finals.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing finals CSV for {family.key}: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate(family: Family, rows: list[dict[str, object]]) -> dict[Key, dict[str, float]]:
    """Mean/std over run seeds per (noise level, offset, estimator)."""
    buckets: dict[Key, dict[str, list[float]]] = defaultdict(
        lambda: {"theta_distance_to_truth": [], "clean_objective_gap": []}
    )
    for row in rows:
        key = (float(row[family.axis_key]), float(row["theta_offset"]), str(row["estimator"]))
        for metric, _ in METRICS:
            buckets[key][metric].append(float(row[metric]))
    stats: dict[Key, dict[str, float]] = {}
    for key, metric_values in buckets.items():
        entry: dict[str, float] = {}
        for metric, _ in METRICS:
            values = np.asarray(metric_values[metric], dtype=float)
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=0))
        stats[key] = entry
    return stats


def _levels(stats: dict[Key, dict[str, float]], estimator: str) -> tuple[list[float], list[float]]:
    noise = sorted({key[0] for key in stats if key[2] == estimator})
    offsets = sorted({key[1] for key in stats if key[2] == estimator})
    return noise, offsets


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------


def _symlog_offset_axis(ax, offsets: list[float]) -> None:
    positive = [o for o in offsets if o > 0.0]
    linthresh = 0.5 * min(positive) if positive else 1e-3
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xticks(offsets)
    ax.set_xticklabels([f"{o:g}" for o in offsets], rotation=45, ha="right", fontsize=8)


def _symlog_noise_axis(ax, noise: list[float]) -> None:
    positive = [n for n in noise if n > 0.0]
    linthresh = 0.5 * min(positive) if positive else 1e-2
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xticks(noise)
    ax.set_xticklabels([f"{n:g}" for n in noise], fontsize=8)


def _metric_yscale(ax, values: list[float]) -> None:
    positive = [v for v in values if v > 0.0]
    if positive and len(positive) == len(values):
        ax.set_yscale("log")
    else:
        linthresh = 0.5 * min(positive) if positive else 1e-9
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(bottom=0.0)


# ---------------------------------------------------------------------------
# Single-axis slice plots
# ---------------------------------------------------------------------------


def _plot_noise_axis_slices(path: Path, family: Family, estimator: str, stats: dict[Key, dict[str, float]]) -> None:
    """x = noise level, one curve per init offset (colored by offset magnitude)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    import matplotlib.pyplot as plt

    noise, offsets = _levels(stats, estimator)
    cmap = colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for ax, (metric, y_label) in zip(axes, METRICS):
        all_vals: list[float] = []
        for i, offset in enumerate(offsets):
            color = cmap(0.85 * i / max(len(offsets) - 1, 1))
            means = [stats[(n, offset, estimator)][f"{metric}_mean"] for n in noise]
            stds = [stats[(n, offset, estimator)][f"{metric}_std"] for n in noise]
            all_vals.extend(means)
            if metric == "theta_distance_to_truth":
                yerr = np.vstack([np.minimum(stds, means), stds])
            else:
                yerr = np.vstack([stds, stds])
            ax.errorbar(
                noise, means, yerr=yerr if any(s > 0 for s in stds) else None,
                color=color, marker="o", linewidth=1.7, markersize=5, capsize=3,
                label=rf"$\delta = {offset:g}$",
            )
        _symlog_noise_axis(ax, noise)
        _metric_yscale(ax, all_vals)
        ax.set_xlabel(family.noise_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(title=r"init offset $\delta$", fontsize=8, ncol=2)
    fig.suptitle(
        f"{ESTIMATOR_LABELS[estimator]} — {family.key}: metrics vs noise level\n"
        r"curves = init offset $\delta$ (viridis: light = larger $\delta$); mean $\pm$ std over seeds",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_offset_axis_slices(path: Path, family: Family, estimator: str, stats: dict[Key, dict[str, float]]) -> None:
    """x = init offset, one curve per noise level (colored by noise magnitude)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    import matplotlib.pyplot as plt

    noise, offsets = _levels(stats, estimator)
    cmap = colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for ax, (metric, y_label) in zip(axes, METRICS):
        all_vals: list[float] = []
        for i, level in enumerate(noise):
            color = cmap(0.85 * i / max(len(noise) - 1, 1))
            means = [stats[(level, o, estimator)][f"{metric}_mean"] for o in offsets]
            stds = [stats[(level, o, estimator)][f"{metric}_std"] for o in offsets]
            all_vals.extend(means)
            label = rf"${family.noise_symbol} = {level:g}$" + (" (clean)" if level == 0.0 else "")
            if metric == "theta_distance_to_truth":
                yerr = np.vstack([np.minimum(stds, means), stds])
            else:
                yerr = np.vstack([stds, stds])
            ax.errorbar(
                offsets, means, yerr=yerr if any(s > 0 for s in stds) else None,
                color=color, marker="o", linewidth=1.7, markersize=5, capsize=3, label=label,
            )
        _symlog_offset_axis(ax, offsets)
        _metric_yscale(ax, all_vals)
        ax.set_xlabel(r"init offset $\delta$ in $\theta_0 = \theta^{\mathrm{FO}}_{\mathrm{clean}} + \delta\,\mathbf{1}$")
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(title=rf"noise ${family.noise_symbol}$", fontsize=8)
    fig.suptitle(
        f"{ESTIMATOR_LABELS[estimator]} — {family.key}: metrics vs init offset\n"
        rf"curves = noise ${family.noise_symbol}$ (viridis: light = more noise); mean $\pm$ std over seeds",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmaps (noise x offset), the optimizer-exploitation showcase
# ---------------------------------------------------------------------------


def _plot_heatmaps(path: Path, family: Family, estimator: str, stats: dict[Key, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt

    noise, offsets = _levels(stats, estimator)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))
    for ax, (metric, title) in zip(axes, METRICS):
        # rows = noise (top = most noise), cols = offset
        grid = np.array(
            [[stats[(n, o, estimator)][f"{metric}_mean"] for o in offsets] for n in noise[::-1]],
            dtype=float,
        )
        positive = grid[grid > 0.0]
        vmin = float(positive.min()) if positive.size else 1e-12
        vmax = float(grid.max()) if grid.max() > 0 else 1.0
        display = np.where(grid > 0.0, grid, vmin)
        im = ax.imshow(display, aspect="auto", cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xticks(range(len(offsets)))
        ax.set_xticklabels([f"{o:g}" for o in offsets], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(noise)))
        ax.set_yticklabels([f"{n:g}" for n in noise[::-1]], fontsize=8)
        ax.set_xlabel(r"init offset $\delta$")
        ax.set_ylabel(rf"noise ${family.noise_symbol}$")
        # annotate each cell; text color flips with luminance for readability
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid[r, c]
                rel = (np.log10(max(val, vmin)) - np.log10(vmin)) / max(np.log10(vmax) - np.log10(vmin), 1e-9)
                ax.text(
                    c, r, f"{val:.1e}", ha="center", va="center", fontsize=6.5,
                    color="white" if rel < 0.6 else "black",
                )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{ESTIMATOR_LABELS[estimator]} — {family.key}: drift from clean first-order truth over noise x offset\n"
        r"brighter = larger drift; the optimizer exploits noise as ${}$ and $\delta$ grow (mean over seeds)".format(
            family.noise_symbol
        ),
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_family_plots(family: Family) -> list[Path]:
    rows = _load_rows(family)
    stats = _aggregate(family, rows)
    out_dir = results_root() / family.project_name
    written: list[Path] = []
    for estimator in ESTIMATORS:
        if not any(key[2] == estimator for key in stats):
            continue
        targets = (
            (out_dir / f"noise_offset_slice_by_noise_{estimator}.png", _plot_noise_axis_slices),
            (out_dir / f"noise_offset_slice_by_offset_{estimator}.png", _plot_offset_axis_slices),
            (out_dir / f"noise_offset_heatmap_{estimator}.png", _plot_heatmaps),
        )
        for path, fn in targets:
            fn(path, family, estimator, stats)
            written.append(path)
            print(f"Wrote {path}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", choices=tuple(FAMILIES), default="all")
    args = parser.parse_args(argv)
    for family in FAMILIES[args.families]:
        build_family_plots(family)


if __name__ == "__main__":
    main()
