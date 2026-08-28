"""Compare train/test acceptance constraint generalization across policy bases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chebyshev-csv", type=Path, required=True)
    parser.add_argument("--full-polynomial-csv", type=Path, required=True)
    parser.add_argument("--acceptance-floor", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean_and_ci(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if array.size <= 1:
        return mean, 0.0
    half_width = float(
        student_t.ppf(0.975, array.size - 1)
        * np.std(array, ddof=1)
        / np.sqrt(array.size)
    )
    return mean, half_width


def _summarize(frame: pd.DataFrame, acceptance_floor: float) -> pd.DataFrame:
    matched = frame.loc[frame["optimize_model"] == frame["evaluate_model"]].copy()
    rows: list[dict[str, float | int]] = []
    for (degree, parameter_count), group in matched.groupby(
        ["degree", "parameter_count"],
        sort=True,
    ):
        train_violation = np.maximum(
            0.0,
            acceptance_floor - group["train_acceptance"].to_numpy(dtype=float),
        )
        test_violation = np.maximum(
            0.0,
            acceptance_floor - group["test_acceptance"].to_numpy(dtype=float),
        )
        row: dict[str, float | int] = {
            "degree": int(degree),
            "parameter_count": int(parameter_count),
        }
        for name, values in (
            ("train_acceptance", group["train_acceptance"]),
            ("test_acceptance", group["test_acceptance"]),
            ("train_violation", pd.Series(train_violation)),
            ("test_violation", pd.Series(test_violation)),
        ):
            mean, ci95 = _mean_and_ci(values)
            row[f"{name}_mean"] = mean
            row[f"{name}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def plot_constraint_overfitting(
    chebyshev: pd.DataFrame,
    full_polynomial: pd.DataFrame,
    *,
    acceptance_floor: float,
    output: Path,
) -> Path:
    """Write a two-basis train/test acceptance and floor-shortfall comparison."""
    summaries = (
        ("Additive Chebyshev (1–609 parameters)", _summarize(chebyshev, acceptance_floor)),
        (
            "Full polynomial interactions (1–1,540 parameters)",
            _summarize(full_polynomial, acceptance_floor),
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row_index, (label, summary) in enumerate(summaries):
        acceptance_ax = axes[row_index, 0]
        violation_ax = axes[row_index, 1]
        for split_name, split_label in (("train", "Train"), ("test", "Test")):
            acceptance_ax.errorbar(
                summary["parameter_count"],
                summary[f"{split_name}_acceptance_mean"],
                yerr=summary[f"{split_name}_acceptance_ci95"],
                marker="o",
                capsize=3,
                label=split_label,
            )
            violation_ax.errorbar(
                summary["parameter_count"],
                summary[f"{split_name}_violation_mean"],
                yerr=summary[f"{split_name}_violation_ci95"],
                marker="o",
                capsize=3,
                label=split_label,
            )
        acceptance_ax.axhline(
            acceptance_floor,
            color="C2",
            linestyle="--",
            linewidth=1.2,
            label="Acceptance floor",
        )
        violation_ax.axhline(0.0, color="C2", linestyle="--", linewidth=1.2)
        acceptance_ax.set_title(label, fontsize=14)
        violation_ax.set_title("Acceptance-floor shortfall", fontsize=14)
        acceptance_ax.set_xlabel("Policy parameter count", fontsize=12)
        violation_ax.set_xlabel("Policy parameter count", fontsize=12)
        acceptance_ax.set_ylabel("Mean acceptance probability", fontsize=12)
        violation_ax.set_ylabel("Mean positive floor violation", fontsize=12)
        for ax in (acceptance_ax, violation_ax):
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
    fig.suptitle("Constraint overfitting under increasing policy capacity", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    return output


def main() -> None:
    args = _parse_args()
    plot_constraint_overfitting(
        pd.read_csv(args.chebyshev_csv),
        pd.read_csv(args.full_polynomial_csv),
        acceptance_floor=float(args.acceptance_floor),
        output=args.output,
    )


if __name__ == "__main__":
    main()
