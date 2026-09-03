"""Merge completed policy-capacity sweeps and regenerate profit plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.policy_capacity import summarize_policy_capacity
from reporting.visualization import (
    plot_policy_capacity_baseline_adjusted_gains,
    plot_policy_capacity_objective,
    plot_policy_capacity_penalized_gains,
)
from plot_policy_capacity_profit_std import plot_profit_curves


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--extension-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", default="xgb")
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--acceptance-floor", type=float, default=0.8787745289312372)
    parser.add_argument("--penalty-weight", type=float, default=1_000_000.0)
    parser.add_argument("--penalty-temperature", type=float, default=0.001)
    return parser.parse_args()


def combine_sweeps(
    base: pd.DataFrame,
    extension: pd.DataFrame,
    *,
    output_dir: Path,
    family: str,
    train_size: int,
    acceptance_floor: float,
    penalty_weight: float,
    penalty_temperature: float,
) -> tuple[Path, Path]:
    """Merge split rows and write combined summaries and profit figures."""
    key = ["split_seed", "optimize_model", "evaluate_model", "degree"]
    combined = (
        pd.concat([base, extension], ignore_index=True)
        .drop_duplicates(key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )
    for split_name in ("train", "test"):
        scaled_gap = (
            acceptance_floor - combined[f"{split_name}_acceptance"].to_numpy(dtype=float)
        ) / penalty_temperature
        soft_gap = penalty_temperature * np.logaddexp(0.0, scaled_gap)
        penalty = penalty_weight * soft_gap * soft_gap
        combined[f"{split_name}_acceptance_penalty"] = penalty
        combined[f"{split_name}_penalized_profit"] = (
            combined[f"{split_name}_profit"].to_numpy(dtype=float) - penalty
        )
    matched = combined.loc[
        (combined["optimize_model"] == family)
        & (combined["evaluate_model"] == family)
    ]
    counts = matched.groupby("degree")["split_seed"].nunique()
    if counts.empty or counts.nunique() != 1:
        raise ValueError("Every combined degree must contain the same number of split seeds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "capacity_per_split.csv"
    summary_path = output_dir / "capacity_summary.csv"
    combined.to_csv(rows_path, index=False)
    summary = summarize_policy_capacity(combined)
    summary.to_csv(summary_path, index=False)
    plot_policy_capacity_objective(
        summary,
        output_dir,
        family=family,
        train_size=train_size,
    )
    plot_policy_capacity_baseline_adjusted_gains(combined, output_dir, family=family)
    plot_policy_capacity_penalized_gains(combined, output_dir, family=family)
    plot_profit_curves(combined, output_dir=output_dir, family=family)
    return rows_path, summary_path


def main() -> None:
    args = _parse_args()
    combine_sweeps(
        pd.read_csv(args.base_csv),
        pd.read_csv(args.extension_csv),
        output_dir=args.output_dir,
        family=str(args.family),
        train_size=int(args.train_size),
        acceptance_floor=float(args.acceptance_floor),
        penalty_weight=float(args.penalty_weight),
        penalty_temperature=float(args.penalty_temperature),
    )


if __name__ == "__main__":
    main()
