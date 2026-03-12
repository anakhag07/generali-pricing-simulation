"""Run a small preset sweep with top-level config overrides."""

from __future__ import annotations

from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "fixed_regression_base"

OVERRIDE_GRID = {
    "sigma": [0.01, 0.03, 0.05],
    "n_grad_samples": [2, 4, 64, 128, 256, 1024],
    "n_samples": [100, 500, 1000],
    "t_steps": [10000],
    "plot": [True],
    "wandb_enabled": [True],
    "wandb_project": ["generali_pricing_simulation_hyperparameter_sweep"],
}


def main() -> None:
    results = run_preset_sweep(base_preset=BASE_PRESET, override_grid=OVERRIDE_GRID)
    print(f"Completed {len(results)} sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
