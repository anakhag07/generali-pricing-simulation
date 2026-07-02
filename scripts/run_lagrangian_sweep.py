"""Run a lagrangian-lambda sweep and plot the resulting frontiers."""

from __future__ import annotations

from data.loader import load_mean_observed_acceptance
from experiments.sweep_reporting import (
    collect_config_sweep_final_rows,
    timestamped_sweep_output_dir,
    write_rows_csv,
    write_sweep_frontier_plots,
)
from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-softmax-lagrangian-sweep"
DISPLAY_KEYS = ("lagrangian_lambda",)
LAMBDA_VALUES = (0.0, 10.0, 50.0, 100.0, 250.0, 300.0,  325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0)

OVERRIDE_GRID = {
    "acceptance_floor": [load_mean_observed_acceptance("glm")],
    "policy_kind": ["softmax"],
    "lagrangian_lambda": list(LAMBDA_VALUES),
    "enabled_estimators": [
        (
            "first_order",
            # "finite_difference",
            # "spsa",
            # "stein_difference",
        )
    ],
    "plot": [True],
    "verbose": [True],
    "wandb_enabled": [True],
}

def main() -> None:
    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    rows = collect_config_sweep_final_rows(
        results,
        config_attr="lagrangian_lambda",
        sweep_key="lambda",
    )
    if not rows:
        raise ValueError("No lagrangian sweep rows were produced. Check lagrangian_lambda overrides.")

    output_dir = timestamped_sweep_output_dir(
        project_name=PROJECT_NAME,
        dirname_prefix="lagrangian_frontier",
    )
    write_rows_csv(
        output_dir / "lagrangian_sweep.csv",
        rows,
        fieldnames=["run_name", "estimator", "lambda", "u", "mean_acceptance", "value"],
    )
    write_sweep_frontier_plots(
        rows,
        output_dir,
        sweep_key="lambda",
        sweep_label="Lagrangian lambda",
        tradeoff_filename="lambda_vs_u_acceptance.png",
    )
    print(f"Completed {len(results)} lagrangian sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
