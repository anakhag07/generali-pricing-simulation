"""Plot separate train and test profit curves with one-SD bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", default="xgb")
    return parser.parse_args()


def _summarize(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    matched = frame.loc[
        (frame["optimize_model"] == family)
        & (frame["evaluate_model"] == family)
    ]
    if matched.empty:
        raise ValueError(f"No matched policy-capacity rows found for {family!r}.")
    return (
        matched.groupby(["degree", "parameter_count"], sort=True)
        .agg(
            train_mean=("train_profit", "mean"),
            train_std=("train_profit", "std"),
            test_mean=("test_profit", "mean"),
            test_std=("test_profit", "std"),
        )
        .reset_index()
    )


def plot_profit_curves(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    family: str,
) -> tuple[Path, Path]:
    """Write separate train and test mean-profit plots with one-SD bands."""
    summary = _summarize(frame, family)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = summary["parameter_count"].to_numpy(dtype=float)
    outputs: list[Path] = []

    for split_name, split_label, band_scale in (
        ("train", "Train", 0.5),
        ("test", "Test", 1.0),
    ):
        mean = summary[f"{split_name}_mean"].to_numpy(dtype=float)
        std = summary[f"{split_name}_std"].to_numpy(dtype=float)
        deviation_label = "standard deviation" if band_scale == 1.0 else "standard deviations"
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        line = ax.plot(parameters, mean, marker="o", label=f"{split_label} mean")[0]
        ax.fill_between(
            parameters,
            mean - band_scale * std,
            mean + band_scale * std,
            color=line.get_color(),
            alpha=0.2,
            label=rf"$\pm {band_scale:g}$ {deviation_label}",
        )
        ax.set_title(f"{split_label} profit versus policy capacity", fontsize=14)
        ax.set_xlabel("Policy parameter count", fontsize=12)
        ax.set_ylabel("Expected profit per customer", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        band_name = "half_std" if band_scale == 0.5 else "std"
        output = output_dir / f"{split_name}_profit_vs_policy_capacity_linear_{band_name}.pdf"
        fig.savefig(output, format="pdf")
        plt.close(fig)
        outputs.append(output)

    return outputs[0], outputs[1]


def main() -> None:
    args = _parse_args()
    plot_profit_curves(
        pd.read_csv(args.capacity_csv),
        output_dir=args.output_dir,
        family=str(args.family),
    )


if __name__ == "__main__":
    main()
