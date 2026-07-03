"""Run a small optimization demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.seeds import SeedSetup
from experiments.slurm import (
    assert_jax_gpu_available,
    run_specs_require_jax,
    submit_to_slurm_if_needed,
)

RUN_CONFIGS: list[str | tuple[str, dict[str, Any]]] = [
    (
        "planted_logistic_base",
        {
            "seed_setup": SeedSetup(
                run_seed=7,
                data_seed=7,
                split_seed=7,
                theta_seed=7,
                noise_seed=101,
                optimizer_seed=7,
            ),
            "enabled_estimators": ("first_order",),
            "correctness": CorrectnessSpec(gradient_source="exact"),
            "perturbation_space": "u",
            "step_rule": "l-bfgs-b",
            "t_steps": 1000,
            "step_size": 0.001,
            "n_samples": 1000,
            "sigma": 0.05,
            "n_grad_samples": 8,
            "plot": True,
            "verbose": True,
            "wandb_enabled": False,
        },
    )
]

# RUN_CONFIGS: list[str | tuple[str, dict[str, Any]]] = [
#     (
#         "real_data_glm_base",
#         {
#             "policy_kind": "softmax",
#             "softmax_action_bounds": (-0.1, 0.2),
#             "initial_u": 0.0,
#             "policy_preprocessing": "no_pca",
#             "feature_order": "cubic",
#             "constraint_mode": "trust_constr",
#             # "n_samples": 700000,
#             "n_samples": None, 
#             "train_fraction": 0.8, 
#             "test_fraction": 0.2,
#             "n_grad_samples": 8,
#             "t_steps": 100,
#             "enabled_estimators": ("first_order", "finite_difference", "stein_difference"),
#             "wandb_enabled": False,
#             "wandb_project": "jax-move-scipy-opt-demo",
#             "compute_backend": "jax",
#             "seed": 8,
#         },
#     )
# ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured pricing experiments.")
    parser.add_argument(
        "--no-sbatch",
        action="store_true",
        help="Run in the current process instead of auto-submitting to ORCD Slurm.",
    )
    return parser.parse_args(argv)


def _resolve_configs() -> list[tuple[str, Any, dict[str, Any]]]:
    configs: list[tuple[str, Any, dict[str, Any]]] = []
    for run_spec in RUN_CONFIGS:
        if isinstance(run_spec, tuple):
            config_name, overrides = run_spec
        else:
            config_name, overrides = run_spec, {}
        config = get_config(config_name, overrides=overrides)
        configs.append((config_name, config, dict(overrides)))
    return configs


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    requires_jax = run_specs_require_jax(RUN_CONFIGS)
    submission = submit_to_slurm_if_needed(
        requires_jax=requires_jax,
        no_sbatch=args.no_sbatch,
        argv=original_argv,
        cwd=_REPO_ROOT,
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

    resolved_configs = _resolve_configs()
    if not requires_jax:
        jax_status = assert_jax_gpu_available([config for _, config, _ in resolved_configs])
        if jax_status is not None:
            print(jax_status)

    for config_name, config, overrides in resolved_configs:
        execute_experiment_run(
            config_name,
            config,
            run_metadata={"preset_name": config_name, "overrides": overrides},
        )


if __name__ == "__main__":
    main()
