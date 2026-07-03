"""Run a planted-logistic homoskedastic-noise sweep."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.seeds import SeedSetup
from experiments.slurm import assert_jax_gpu_available, submit_to_slurm_if_needed
from experiments.sweep_utils import run_preset_sweep
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective

BASE_PRESET = "planted_logistic_base"
PROJECT_NAME = "homoskedastic-theta-offset-sweep"
DISPLAY_KEYS: tuple[str, ...] = ()
NOISE_STD = 0.5
BASE_THETA = np.asarray(
    [
        0.4054882808450241,
        0.00012799868045781167,
        -4.657524122982136e-05,
        6.221922280809605e-05,
    ],
    dtype=float,
)
THETA_OFFSETS = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

_PLANTED_BASE = get_config(BASE_PRESET)
_SEED_SETUP = SeedSetup(
    run_seed=7,
    data_seed=7,
    split_seed=7,
    theta_seed=7,
    noise_seed=101,
    optimizer_seed=7,
)


def _theta_offset_label(offset: float) -> str:
    return f"theta-offset-{float(offset):g}"


def _theta0(offset: float) -> np.ndarray:
    return BASE_THETA + float(offset)


def _noisy_objective() -> NoisyObjective:
    return NoisyObjective(
        base_objective=_PLANTED_BASE.objective,
        noise=HomoskedasticGaussianNoise(std=NOISE_STD),
    )


OVERRIDE_LIST = [
    {
        "_run_name": _theta_offset_label(offset),
        "objective": _noisy_objective(),
        "theta0": _theta0(offset),
        "seed_setup": _SEED_SETUP,
        "enabled_estimators": ("finite_difference", "stein_difference"),
        "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
        "perturbation_space": "u",
        "step_rule": "l-bfgs-b",
        "t_steps": 1000,
        "step_size": 0.001,
        "n_samples": 1000,
        "sigma": 0.05,
        "n_grad_samples": 8,
        "plot": True,
        "verbose": False,
        "wandb_enabled": False,
    }
    for offset in THETA_OFFSETS
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured preset sweep.")
    parser.add_argument(
        "--no-sbatch",
        action="store_true",
        help="Run in the current process instead of auto-submitting to ORCD Slurm.",
    )
    return parser.parse_args(argv)


def _override_list_requires_jax() -> bool:
    return any(overrides.get("compute_backend") == "jax" for overrides in OVERRIDE_LIST)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    requires_jax = _override_list_requires_jax()

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
        override_list=OVERRIDE_LIST,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    print(f"Completed {len(results)} sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
