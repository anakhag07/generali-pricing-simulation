"""Plot Chebyshev train/test mean acceptance with confidence bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from plot_policy_capacity_constraint_overfitting import _summarize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chebyshev-csv", type=Path, required=True)
    parser.add_argument("--acceptance-floor", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def plot_chebyshev_mean_acceptance(
    frame: pd.DataFrame,
    *,
    acceptance_floor: float,
    output: Path,
) -> Path:
    """Write the single-panel log-capacity acceptance plot."""
    summary = _summarize(frame, acceptance_floor)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for split_name, split_label in (("train", "Train"), ("test", "Test")):
        mean = gaussian_filter1d(
            summary[f"{split_name}_acceptance_mean"].to_numpy(dtype=float),
            sigma=1.0,
            mode="nearest",
        )
        parameters = summary["parameter_count"].to_numpy(dtype=float)
        ax.plot(parameters, mean, marker="o", label=split_label)
    ax.axhline(
        acceptance_floor,
        color="C2",
        linestyle="--",
        linewidth=1.2,
        label="Acceptance floor",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.85, 0.90)
    ax.set_xlabel("Policy parameter count (log scale)", fontsize=12)
    ax.set_ylabel("Mean Acceptance Probability", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def main() -> None:
    args = _parse_args()
    plot_chebyshev_mean_acceptance(
        pd.read_csv(args.chebyshev_csv),
        acceptance_floor=float(args.acceptance_floor),
        output=args.output,
    )


if __name__ == "__main__":
    main()
