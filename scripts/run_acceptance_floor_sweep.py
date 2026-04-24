"""Run a trust-constrained acceptance-floor sweep and plot the frontier."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from experiments.sweep_utils import run_preset_sweep
from reporting.visualization import _plot_sweep_pareto_frontier, _plot_sweep_tradeoffs

BASE_PRESET = "real_data_glm_softmax_policy_trust_region_constr"
PROJECT_NAME = "glm-softmax-acceptance-floor-sweep"
DISPLAY_KEYS = ("acceptance_floor",)
C_VALUES = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.84,
    0.87,
    0.89,
    0.91,
    0.925,
    0.94,
    0.95,
    0.96,
    0.97,
    0.978,
    0.985,
    0.99,
    0.993,
    0.995,
)

OVERRIDE_GRID = {
    "acceptance_floor": list(C_VALUES),
    "enabled_estimators": [
        (
            "first_order",
            "finite_difference",
            "spsa",
            "stein_difference",
        )
    ],
    # The sweep already writes aggregate frontier plots at the end, so disable
    # per-run plotting and W&B logging to keep dense trust-constr sweeps tractable.
    "plot": [False],
    "verbose": [False],
    "wandb_enabled": [False],
}


def _collect_rows(results):
    rows: list[dict[str, float | str]] = []
    for run_name, result in results:
        acceptance_floor = result.config.acceptance_floor
        if acceptance_floor is None:
            continue
        for estimator, estimator_result in result.results.items():
            if estimator_result.mean_acceptance is None:
                continue
            rows.append(
                {
                    "run_name": run_name,
                    "estimator": estimator,
                    "c": float(acceptance_floor),
                    "u": float(estimator_result.u),
                    "mean_acceptance": float(estimator_result.mean_acceptance),
                    "value": float(estimator_result.value),
                    "constraint_violation": (
                        float(estimator_result.constraint_violation)
                        if estimator_result.constraint_violation is not None
                        else ""
                    ),
                }
            )
    return rows


def _write_rows(rows, output_dir: Path) -> None:
    csv_path = output_dir / "acceptance_floor_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "estimator",
                "c",
                "u",
                "mean_acceptance",
                "value",
                "constraint_violation",
            ],
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
        raise ValueError("No acceptance-floor sweep rows were produced. Check acceptance_floor overrides.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"acceptance_floor_frontier_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir)

    plot_dir = str(output_dir)
    _plot_sweep_tradeoffs(
        rows,
        plot_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        filename="c_vs_u_acceptance.png",
    )
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="value",
        y_label="Final objective value",
        filename="pareto_objective_acceptance.png",
    )
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="u",
        y_label="Final u",
        filename="pareto_u_acceptance.png",
    )
    print(f"Completed {len(results)} acceptance-floor sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
