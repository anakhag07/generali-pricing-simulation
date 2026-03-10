"""Run a small optimization demo."""

from __future__ import annotations

from experiments.configs import get_config
from experiments.reporters import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    ReporterStack,
    create_run_context,
)
from experiments.run import run_experiment

# RUN_CONFIGS = ["planted_logistic", "custom"]
RUN_CONFIGS = ["custom"]

def main() -> None:
    for config_name in RUN_CONFIGS:
        config = get_config(config_name)
        run_context = create_run_context(config_name, runs_root="outputs")
        reporters = ReporterStack([
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            JsonReporter(),
            PlotReporter(),
        ])
        reporters.on_start(run_context, config)
        result = run_experiment(config, step_reporter=reporters)
        reporters.on_end(run_context, result)


if __name__ == "__main__":
    main()
