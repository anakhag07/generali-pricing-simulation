"""Run a trust-constrained acceptance-floor sweep and plot the frontier."""

from __future__ import annotations

from experiments.sweep_reporting import (
    collect_config_sweep_final_rows,
    timestamped_sweep_output_dir,
    write_rows_csv,
    write_sweep_frontier_plots,
)
from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "real_data_glm_base"
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
    "policy_kind": ["softmax"],
    "constraint_mode": ["trust_constr"],
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

def main() -> None:
    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    rows = collect_config_sweep_final_rows(
        results,
        config_attr="acceptance_floor",
        sweep_key="c",
        include_constraint_violation=True,
    )
    if not rows:
        raise ValueError("No acceptance-floor sweep rows were produced. Check acceptance_floor overrides.")

    output_dir = timestamped_sweep_output_dir(
        project_name=PROJECT_NAME,
        dirname_prefix="acceptance_floor_frontier",
    )
    write_rows_csv(
        output_dir / "acceptance_floor_sweep.csv",
        rows,
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
    write_sweep_frontier_plots(
        rows,
        output_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        tradeoff_filename="c_vs_u_acceptance.png",
    )
    print(f"Completed {len(results)} acceptance-floor sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
