"""Run a small preset sweep with top-level config overrides."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from experiments.slurm import assert_jax_gpu_available, submit_to_slurm_if_needed
from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "feature-order-sweep"
DISPLAY_KEYS = ("feature_order",)

OVERRIDE_GRID = {
    "policy_kind": ["softmax"],
    "softmax_action_bounds": [(-0.1, 0.2)],
    "initial_u": [0.0],
    "policy_preprocessing": ["no_pca"],
    "feature_order": ["linear", "quadratic", "cubic"],
    "constraint_mode": ["trust_constr"],
    "n_samples": [100],
    "train_fraction": [0.8] , 
    "test_fraction": [0.2],
    "n_grad_samples": [8],
    "t_steps": [100],
    "enabled_estimators": [("first_order",)],
    "wandb_enabled": [False],
    "wandb_project": ["jax-move-scipy-opt-demo"],
    "compute_backend": ["jax"],
    "seed": [8],
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured preset sweep.")
    parser.add_argument(
        "--no-sbatch",
        action="store_true",
        help="Run in the current process instead of auto-submitting to ORCD Slurm.",
    )
    return parser.parse_args(argv)


def _override_grid_requires_jax() -> bool:
    values = OVERRIDE_GRID.get("compute_backend", [])
    if isinstance(values, str):
        return values == "jax"
    return any(value == "jax" for value in values)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    requires_jax = _override_grid_requires_jax()

    submission = submit_to_slurm_if_needed(
        requires_jax=requires_jax,
        no_sbatch=args.no_sbatch,
        argv=original_argv,
    )
    if submission is not None:
        print(
            f"Submitted {submission.profile.name} Slurm job {submission.job_id}; "
            f"logs: {submission.profile.output}"
        )
        return

    if requires_jax:
        jax_status = assert_jax_gpu_available([SimpleNamespace(compute_backend="jax")])
        if jax_status is not None:
            print(jax_status)

    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    print(f"Completed {len(results)} sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
