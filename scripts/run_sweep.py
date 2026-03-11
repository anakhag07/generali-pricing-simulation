"""Run a small preset sweep with top-level config overrides."""

from __future__ import annotations

from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "fixed_regression_base"

OVERRIDE_GRID = {
    "seed": [7, 11],
    "sigma": [0.03, 0.05],
    "n_grad_samples": [64],
    "t_steps": [250],
    "lbfgs_maxiter": [250],
    "plot": [False],
    "wandb_enabled": [False],
}


def main() -> None:
    results = run_preset_sweep(base_preset=BASE_PRESET, override_grid=OVERRIDE_GRID)
    print(f"Completed {len(results)} sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
