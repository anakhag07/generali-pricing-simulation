"""Run a small optimization demo."""

from __future__ import annotations

from typing import Any

from experiments.configs import get_config
from experiments.reporters import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    ReporterStack,
    WandbReporter,
    create_run_context,
)
from experiments.run import run_experiment

RUN_CONFIGS: list[str | tuple[str, dict[str, Any]]] = [
    (
        "real_data_glm_base",
        {
            "policy_kind": "softmax",
            "policy_preprocessing": "no_pca",
            "feature_order": "linear", 
            "constraint_mode": "trust_constr",
            "n_samples": 190000,
            "n_grad_samples": 8,
            "t_steps": 20,
            "enabled_estimators": ("first_order", "finite_difference", "stein_difference"),
            "wandb_enabled": False,
        },
    )
]

def main() -> None:
    for run_spec in RUN_CONFIGS:
        if isinstance(run_spec, tuple):
            config_name, overrides = run_spec
        else:
            config_name, overrides = run_spec, {}
        config = get_config(config_name, overrides=overrides)
        run_context = create_run_context(config_name, runs_root="outputs")
        reporter_list = [
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            JsonReporter(),
            PlotReporter(),
        ]
        if config.wandb_enabled:
            reporter_list.append(WandbReporter())
        reporters = ReporterStack(reporter_list)
        reporters.on_start(run_context, config)
        result = run_experiment(config, step_reporter=reporters)
        reporters.on_end(run_context, result)


if __name__ == "__main__":
    main()
