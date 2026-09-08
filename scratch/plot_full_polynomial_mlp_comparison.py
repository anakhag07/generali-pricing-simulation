"""Add the saved two-hidden-layer MLP result to the full-polynomial XGB plot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.paths import results_root  # noqa: E402
from reporting.visualization import plot_policy_capacity_objective  # noqa: E402


FULL_POLYNOMIAL_SWEEP = (
    "policy-capacity-xgb-u-0-0p16-full-polynomial-degree-3"
    "/sweeps/20260828_full_polynomial_degree3"
)
MLP_SWEEP = (
    "policy-mlp-two-layer-glm-xgb-u-0-0p16"
    "/sweeps/20260902_mlp_two_layer"
)


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
        "--output-dir",
        type=Path,
        default=root / FULL_POLYNOMIAL_SWEEP,
    )
    return parser.parse_args()


def plot_comparison(
    full_polynomial_summary: Path,
    mlp_summary: Path,
    output_dir: Path,
) -> Path:
    """Replay saved optimizer summaries and write the comparison PDF."""
    polynomial = pd.read_csv(full_polynomial_summary)
    mlp = pd.read_csv(mlp_summary)
    xgb_rows = mlp.loc[mlp["model"] == "xgb"]
    if xgb_rows.shape[0] != 1:
        raise ValueError("Expected exactly one XGB row in the MLP summary.")
    row = xgb_rows.iloc[0]
    parameter_count = int(row["parameter_count"])
    return plot_policy_capacity_objective(
        polynomial,
        output_dir,
        family="xgb",
        train_size=100,
        comparison_points=(
            {
                "label": f"MLP ({parameter_count} parameters)",
                "parameter_count": parameter_count,
                "train_profit_mean": row["train_profit_mean"],
                "train_profit_ci95": row["train_profit_ci95"],
                "test_profit_mean": row["test_profit_mean"],
                "test_profit_ci95": row["test_profit_ci95"],
            },
        ),
        degree_label="Total polynomial degree",
        output_stem="objective_vs_policy_capacity_xgb_with_mlp",
        append_comparisons=True,
    )


def main() -> None:
    args = _parse_args()
    path = plot_comparison(
        args.full_polynomial_summary,
        args.mlp_summary,
        args.output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
