"""Plot saved full-polynomial and MLP train/test mean acceptance results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.paths import results_root  # noqa: E402


FULL_POLYNOMIAL_SWEEP = (
    "policy-capacity-xgb-u-0-0p16-full-polynomial-degree-3"
    "/sweeps/20260828_full_polynomial_degree3"
)
MLP_SWEEP = (
    "policy-mlp-two-layer-glm-xgb-u-0-0p16"
    "/sweeps/20260902_mlp_two_layer"
)
ACCEPTANCE_FLOOR = 0.8787745289312372


def _parse_args() -> argparse.Namespace:
    root = results_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-polynomial-summary",
        type=Path,
        default=root / FULL_POLYNOMIAL_SWEEP / "capacity_summary.csv",
    )
    parser.add_argument(
        "--mlp-summary",
        type=Path,
        default=root / MLP_SWEEP / "mlp_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / FULL_POLYNOMIAL_SWEEP / "mean_acceptance_full_polynomial_with_mlp.pdf",
    )
    return parser.parse_args()


def plot_acceptance(
    full_polynomial_summary: Path,
    mlp_summary: Path,
    output: Path,
) -> Path:
    """Replay saved summaries and write the standalone acceptance PDF."""
    polynomial = pd.read_csv(full_polynomial_summary)
    polynomial = polynomial.loc[
        (polynomial["optimize_model"] == "xgb")
        & (polynomial["evaluate_model"] == "xgb")
    ].sort_values("degree")
    if polynomial.empty:
        raise ValueError("No matched XGB rows found in the full-polynomial summary.")

    mlp = pd.read_csv(mlp_summary)
    mlp_rows = mlp.loc[mlp["model"] == "xgb"]
    if mlp_rows.shape[0] != 1:
        raise ValueError("Expected exactly one XGB row in the MLP summary.")
    mlp_row = mlp_rows.iloc[0]

    polynomial_positions = np.arange(polynomial.shape[0], dtype=float)
    mlp_position = float(polynomial.shape[0])
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for color, split_name, split_label in (
        ("C0", "train", "Train"),
        ("C1", "test", "Test"),
    ):
        ax.errorbar(
            polynomial_positions,
            polynomial[f"{split_name}_acceptance_mean"],
            yerr=polynomial[f"{split_name}_acceptance_ci95"],
            color=color,
            marker="o",
            capsize=3,
            label=split_label,
        )
        ax.errorbar(
            [mlp_position],
            [float(mlp_row[f"{split_name}_acceptance_mean"])],
            yerr=[float(mlp_row[f"{split_name}_acceptance_ci95"])],
            color=color,
            marker="D",
            markersize=6,
            capsize=3,
        )
    ax.axhline(
        ACCEPTANCE_FLOOR,
        color="C2",
        linestyle="--",
        linewidth=1.2,
        label="Acceptance floor",
    )

    polynomial_labels = [
        f"Degree {int(row.degree)}\n({int(row.parameter_count):,})"
        for row in polynomial.itertuples(index=False)
    ]
    mlp_parameter_count = int(mlp_row["parameter_count"])
    ax.set_xticks(
        np.arange(polynomial.shape[0] + 1, dtype=float),
        labels=[*polynomial_labels, f"MLP\n({mlp_parameter_count:,})"],
    )
    ax.set_title("Mean acceptance: full polynomial expansion and MLP", fontsize=14)
    ax.set_xlabel("Decision rule (parameter count)", fontsize=12)
    ax.set_ylabel("Mean acceptance probability", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def main() -> None:
    args = _parse_args()
    print(plot_acceptance(args.full_polynomial_summary, args.mlp_summary, args.output))


if __name__ == "__main__":
    main()
