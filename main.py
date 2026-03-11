"""Run a small optimization demo."""

from __future__ import annotations

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

RUN_CONFIGS = ["fixed_regression_base"]

def main() -> None:
    for config_name in RUN_CONFIGS:
        config = get_config(config_name)
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
