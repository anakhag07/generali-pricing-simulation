"""Run a small optimization demo."""

from __future__ import annotations

from experiments.configs import get_config
from experiments.run import run_experiment

RUN_CONFIGS = ["planted_logistic"]


def main() -> None:
    for config_name in RUN_CONFIGS:
        config = get_config(config_name)
        print(f"\n=== Running experiment: {config_name} ===")
        run_experiment(config)


if __name__ == "__main__":
    main()
