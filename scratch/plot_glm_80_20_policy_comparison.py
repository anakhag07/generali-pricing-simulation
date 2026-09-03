"""Compare an 80/20 GLM first-order policy with historical actions.

The saved policy is evaluated separately on its bound training and test rows.
Historical actions are evaluated on all selected rows.  Every plotted point is
the row-level mean and every whisker is one population standard deviation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from experiments.policy_artifacts import load_policy_artifact
from scripts.evaluate_historical_policy_objective import (
    evaluate_historical_policy_objective,
    evaluate_historical_u_objective,
    evaluate_model_policy_objective,
    load_historical_rows,
)


DEFAULT_RUN_DIR = Path(
    "/home/anakhag/projects/generali-pricing/results/"
    "glm-softmax-80-20-first-order/glm-softmax-80-20-first-order"
)
DEFAULT_POLICY = (
    DEFAULT_RUN_DIR
    / "seeds"
    / "seed-8"
    / "policies"
    / "first_order"
    / "policy.json"
)
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "summary-seed-8.json"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "policy-comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-artifact", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-png", type=Path, default=None)
    return parser.parse_args()


def mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return row-level mean and population standard deviation."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty finite array.")
    return float(np.mean(array)), float(np.std(array, ddof=0))


def summarize_policy_split(artifact: Any, split: str) -> dict[str, Any]:
    row_indices = artifact.row_indices(split)
    historical_rows = load_historical_rows(row_indices)
    model_eval = evaluate_model_policy_objective(artifact=artifact, split=split)
    observed_eval = evaluate_historical_policy_objective(
        config=SimpleNamespace(
            objective=artifact.build_objective(),
            x_fixed=artifact.load_x(split=split),
        ),
        theta=artifact.theta,
        row_indices=row_indices,
        historical_rows=historical_rows,
        estimator=artifact.estimator,
        split=split,
    )
    action_mean, action_std = mean_std(model_eval.policy_u)
    model_mean, model_std = mean_std(model_eval.objective_contribution)
    observed_mean, observed_std = mean_std(observed_eval.objective_contribution)
    return {
        "category": f"{split.title()} policy",
        "split": split,
        "action_source": "optimized policy",
        "n_rows": int(row_indices.size),
        "mean_u": action_mean,
        "std_u": action_std,
        "model_objective_mean": model_mean,
        "model_objective_std": model_std,
        "model_objective_sum": float(np.sum(model_eval.objective_contribution)),
        "observed_objective_mean": observed_mean,
        "observed_objective_std": observed_std,
        "observed_objective_sum": float(np.sum(observed_eval.objective_contribution)),
        "model_acceptance_mean": float(np.mean(model_eval.model_acceptance)),
        "historical_acceptance_mean": float(np.mean(observed_eval.historical_acceptance)),
    }


def summarize_historical_all(artifact: Any) -> dict[str, Any]:
    split = "all"
    row_indices = artifact.row_indices(split)
    historical_rows = load_historical_rows(row_indices)
    objective = artifact.build_objective()
    x_eval = artifact.load_x(split=split)
    model_eval = evaluate_historical_u_objective(
        model_type=str(artifact.objective.model_type),
        row_indices=row_indices,
        historical_rows=historical_rows,
        acceptance_source="model",
        technical_price_source="model",
        objective=objective,
        x_eval=x_eval,
        split=split,
    )
    observed_eval = evaluate_historical_u_objective(
        model_type=str(artifact.objective.model_type),
        row_indices=row_indices,
        historical_rows=historical_rows,
        acceptance_source="historical",
        technical_price_source="historical",
        split=split,
    )
    action_mean, action_std = mean_std(observed_eval.historical_u)
    model_mean, model_std = mean_std(model_eval.objective_contribution)
    observed_mean, observed_std = mean_std(observed_eval.objective_contribution)
    return {
        "category": "Historical actions (all)",
        "split": split,
        "action_source": "historical U",
        "n_rows": int(row_indices.size),
        "mean_u": action_mean,
        "std_u": action_std,
        "model_objective_mean": model_mean,
        "model_objective_std": model_std,
        "model_objective_sum": float(np.sum(model_eval.objective_contribution)),
        "observed_objective_mean": observed_mean,
        "observed_objective_std": observed_std,
        "observed_objective_sum": float(np.sum(observed_eval.objective_contribution)),
        "model_acceptance_mean": float(np.mean(model_eval.selected_acceptance)),
        "historical_acceptance_mean": float(np.mean(observed_eval.selected_acceptance)),
    }


def validate_run(artifact: Any, run_summary: dict[str, Any]) -> None:
    config = run_summary["config"]
    if artifact.estimator != "first_order":
        raise ValueError(f"Expected first_order, got {artifact.estimator!r}.")
    if not np.isclose(float(config["train_fraction"]), 0.8):
        raise ValueError("Run is not an 80% training split.")
    if not np.isclose(float(config["test_fraction"]), 0.2):
        raise ValueError("Run is not a 20% test split.")
    action_bounds = (artifact.policy_head.action_low, artifact.policy_head.action_high)
    if not np.allclose(action_bounds, (-0.1, 0.2)):
        raise ValueError("Run does not use policy bounds [-0.1, 0.2].")


def plot_comparison(
    frame: pd.DataFrame,
    output_path: Path,
    preview_path: Path | None = None,
) -> None:
    labels = frame["category"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (
        ("mean_u", "std_u", "Proposed Price Change", "Price Change (%)"),
        (
            "model_objective_mean",
            "model_objective_std",
            "GLM-Estimated Objective",
            "Mean Objective (lower is better)",
        ),
        (
            "observed_objective_mean",
            "observed_objective_std",
            "Observed-Outcome Diagnostic",
            "Mean Objective (lower is better)",
        ),
    )
    for ax, (mean_col, std_col, title, ylabel) in zip(axes, panels, strict=True):
        means = frame[mean_col].to_numpy(dtype=float)
        stds = frame[std_col].to_numpy(dtype=float)
        if mean_col == "mean_u":
            means = 100.0 * means
            stds = 100.0 * stds
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
        ax.errorbar(x, means, yerr=stds, fmt="o", capsize=5)
        ax.axhline(0.0, color="0.7", linewidth=1.0)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Evaluation Cohort", fontsize=12)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelrotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("80/20 First-Order Policy: Train, Test, and Historical Comparison", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact = load_policy_artifact(args.policy_artifact)
    with args.run_summary.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)
    validate_run(artifact, run_summary)

    records = [
        summarize_policy_split(artifact, "train"),
        summarize_policy_split(artifact, "test"),
        summarize_historical_all(artifact),
    ]
    frame = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "train_test_historical_comparison.csv"
    json_path = args.output_dir / "train_test_historical_comparison.json"
    pdf_path = args.output_dir / "plots" / "first_order_train_test_historical_comparison.pdf"
    frame.to_csv(csv_path, index=False)

    payload = {
        "policy_artifact": str(args.policy_artifact.resolve()),
        "run_summary": str(args.run_summary.resolve()),
        "aggregation": {
            "unit": "selected customer-policy row",
            "center": "arithmetic mean over rows",
            "spread": "population standard deviation over rows (ddof=0)",
            "whisker": "mean plus or minus one population standard deviation",
        },
        "objective_definition": {
            "sign": "acceptance * (technical price - revenue); lower is better",
            "model": "GLM predicted acceptance and predicted claims/loss",
            "observed": "historical acceptance and observed claims/loss",
            "policy_observed_caveat": (
                "Observed outcomes are held fixed while the learned policy price replaces historical U; "
                "this is a descriptive diagnostic, not a causal off-policy estimate."
            ),
        },
        "records": records,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    plot_comparison(frame, pdf_path, args.preview_png)

    print(frame.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
