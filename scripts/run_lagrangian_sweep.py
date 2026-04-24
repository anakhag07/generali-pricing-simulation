"""Run a lagrangian-lambda sweep and plot the resulting frontiers."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from data.loader import load_mean_observed_acceptance
from experiments.sweep_utils import run_preset_sweep
from reporting.visualization import (
    plot_lagrangian_lambda_tradeoffs,
    plot_lagrangian_pareto_frontier,
)

BASE_PRESET = "real_data_glm_softmax_policy_base"
PROJECT_NAME = "glm-softmax-lagrangian-sweep"
DISPLAY_KEYS = ("lagrangian_lambda",)
LAMBDA_VALUES = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

OVERRIDE_GRID = {
    "acceptance_floor": [load_mean_observed_acceptance("glm")],
    "lagrangian_lambda": list(LAMBDA_VALUES),
    "enabled_estimators": [
        (
            "first_order",
            "finite_difference",
            "spsa",
            "stein_difference",
        )
    ],
    "plot": [True],
    "verbose": [True],
    "wandb_enabled": [True],
}


def _collect_rows(results):
    rows: list[dict[str, float | str]] = []
    for run_name, result in results:
        lambda_value = result.config.lagrangian_lambda
        if lambda_value is None:
            continue
        for estimator, estimator_result in result.results.items():
            if estimator_result.mean_acceptance is None:
                continue
            rows.append(
                {
                    "run_name": run_name,
                    "estimator": estimator,
                    "lambda": float(lambda_value),
                    "u": float(estimator_result.u),
                    "mean_acceptance": float(estimator_result.mean_acceptance),
                    "value": float(estimator_result.value),
                }
            )
    return rows


def _write_rows(rows, output_dir: Path) -> None:
    csv_path = output_dir / "lagrangian_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_name", "estimator", "lambda", "u", "mean_acceptance", "value"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    rows = _collect_rows(results)
    if not rows:
        raise ValueError("No lagrangian sweep rows were produced. Check lagrangian_lambda overrides.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"lagrangian_frontier_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir)

    plot_dir = str(output_dir)
    plot_lagrangian_lambda_tradeoffs(rows, plot_dir)
    plot_lagrangian_pareto_frontier(
        rows,
        plot_dir,
        y_key="value",
        y_label="Final objective value",
        filename="pareto_objective_acceptance.png",
    )
    plot_lagrangian_pareto_frontier(
        rows,
        plot_dir,
        y_key="u",
        y_label="Final u",
        filename="pareto_u_acceptance.png",
    )
    print(f"Completed {len(results)} lagrangian sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
