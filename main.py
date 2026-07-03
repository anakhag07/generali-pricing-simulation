"""Run a small optimization demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.seeds import SeedSetup
from experiments.slurm import assert_jax_gpu_available, run_specs_require_jax

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
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


def _normalize_run_spec(index: int) -> tuple[str, dict[str, Any]]:
    run_spec = RUN_CONFIGS[index]
    if isinstance(run_spec, tuple):
        config_name, overrides = run_spec
        return config_name, dict(overrides)
    return run_spec, {}


def _run_name(index: int, config_name: str) -> str:
    if len(RUN_CONFIGS) == 1:
        return config_name
    return f"{config_name}__task_{index:03d}"


def _run_config_task(index: int, context: LaunchContext) -> dict[str, object]:
    del context
    config_name, overrides = _normalize_run_spec(index)
    config = get_config(config_name, overrides=overrides)
    jax_status = assert_jax_gpu_available([config])
    if jax_status is not None:
        print(jax_status)
    executed = execute_experiment_run(
        _run_name(index, config_name),
        config,
        run_metadata={"preset_name": config_name, "overrides": overrides},
    )
    return {
        "config_name": config_name,
        "run_name": executed.name,
        "run_dir": str(executed.run_context.run_dir),
    }


def _run_all_configs(context: LaunchContext) -> None:
    for index in range(len(RUN_CONFIGS)):
        _run_config_task(index, context)


def _build_launch_plan() -> LaunchPlan:
    return LaunchPlan(
        name="main",
        task_count=len(RUN_CONFIGS),
        requires_jax=run_specs_require_jax(RUN_CONFIGS),
        run_task=_run_config_task,
        run_all=_run_all_configs,
        default_launch="auto",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(), args=args, argv=original_argv, cwd=_REPO_ROOT)


if __name__ == "__main__":
    main()
