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
        run_context = create_run_context(config_name, runs_root="runs")

        # Build reporter stack with configurable step file logging to avoid
        # unnecessary per-step I/O and large steps.csv files on long runs.
        reporter_list = [
            ConsoleReporter(verbose=config.verbose),
            JsonReporter(),
            PlotReporter(),
        ]

        # Enable file step logging only when configured. Prefer an explicit
        # `log_steps_to_file` flag on the config if present; otherwise, fall
        # back to `verbose` as a reasonable default for detailed logging.
        if getattr(config, "log_steps_to_file", config.verbose):
            reporter_list.insert(1, FileStepLogger())

        reporters = ReporterStack(reporter_list)
        reporters.on_start(run_context, config)
        result = run_experiment(config, step_reporter=reporters)
        reporters.on_end(run_context, result)


if __name__ == "__main__":
    main()
