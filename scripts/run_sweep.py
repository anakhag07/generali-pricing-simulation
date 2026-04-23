"""Run a small preset sweep with top-level config overrides."""

from __future__ import annotations

from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "real_data_glm_softmax_policy_trust_region_constr"
PROJECT_NAME = "glm-softmax-policy-trust-region-constr-initial-penalty-sweep"
DISPLAY_KEYS = ("initial_constr_penalty",)

OVERRIDE_GRID = {
    # "sigma": [0.01], #[0.0001, 0.001, 0.01, 0.05, 0.1],
    # "n_grad_samples": [2, 4, 64, 128, 256, 1024, 2048, 1024 * 8],
    # "n_samples": [100], # [100, 500, 1000],
    # "t_steps": [10000],
    "seed": [8],
    "plot": [True],
    "initial_constr_penalty": [0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
    "wandb_enabled": [True],
    "wandb_project": [PROJECT_NAME],
    # "wandb_group": [PROJECT_NAME],
}


def main() -> None:
    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    print(f"Completed {len(results)} sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
