"""Plot train/test mean profit against Chebyshev policy capacity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-csv", type=Path, required=True)
    parser.add_argument("--model", choices=("glm", "xgb"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _summarize(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    matched = frame.loc[
        (frame["optimize_model"] == model)
        & (frame["evaluate_model"] == model)
    ].copy()
    if matched.empty:
        raise ValueError(f"No matching {model!r} optimize/evaluate rows found.")
    return (
        matched.groupby(["degree", "parameter_count"], sort=True)
        .agg(
            train_profit_mean=("train_profit", "mean"),
            test_profit_mean=("test_profit", "mean"),
        )
        .reset_index()
    )


def plot_mean_profit(
    frame: pd.DataFrame,
    *,
    model: str,
    output: Path,
) -> Path:
    """Write a train/test mean-profit plot without uncertainty bands."""
    summary = _summarize(frame, model)
    parameters = summary["parameter_count"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for split_name, split_label in (("train", "Train"), ("test", "Test")):
        ax.plot(
            parameters,
            summary[f"{split_name}_profit_mean"].to_numpy(dtype=float),
            marker="o",
            label=split_label,
        )
    ax.set_xscale("log")
    ax.set_title(
        "Mean Expected Profit per Customer vs. Decision Rule Parameter Count",
        fontsize=14,
    )
    ax.set_xlabel("Decision Rule Parameter Count (log scale)", fontsize=12)
    ax.set_ylabel("Mean Expected Profit per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def main() -> None:
    args = _parse_args()
    plot_mean_profit(
        pd.read_csv(args.capacity_csv),
        model=args.model,
        output=args.output,
    )


if __name__ == "__main__":
    main()
