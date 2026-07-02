"""Run-lifecycle helpers for experiment execution with reporters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext, create_run_context
from experiments.results import ExperimentResult
from experiments.run import run_experiment

if TYPE_CHECKING:
    from experiments.reporting.base import ReporterStack


@dataclass(frozen=True)
class ExecutedRun:
    """Completed experiment run plus its output context."""

    name: str
    config: ExperimentConfig
    result: ExperimentResult
    run_context: RunContext


ReporterStackFactory = Callable[[ExperimentConfig], "ReporterStack"]


def default_reporter_stack(config: ExperimentConfig) -> "ReporterStack":
    """Build the default reporter stack in its required execution order."""
    from experiments.reporting.artifacts import PolicyArtifactReporter
    from experiments.reporting.base import ReporterStack
    from experiments.reporting.console import ConsoleReporter
    from experiments.reporting.json_summary import JsonReporter
    from experiments.reporting.plots import PlotReporter
    from experiments.reporting.step_logger import FileStepLogger
    from experiments.reporting.wandb import WandbReporter

    reporter_list = [
        ConsoleReporter(verbose=config.verbose),
        FileStepLogger(),
        PolicyArtifactReporter(),
        JsonReporter(),
        PlotReporter(),
    ]
    if config.wandb_enabled:
        reporter_list.append(WandbReporter())
    return ReporterStack(reporter_list)


def execute_experiment_run(
    name: str,
    config: ExperimentConfig,
    *,
    runs_root: str = "outputs",
    reporter_stack_factory: ReporterStackFactory = default_reporter_stack,
) -> ExecutedRun:
    """Create output context, run one experiment, and finalize reporters."""
    run_context = create_run_context(name, runs_root=runs_root)
    reporters = reporter_stack_factory(config)
    reporters.on_start(run_context, config)
    result = run_experiment(config, step_reporter=reporters)
    reporters.on_end(run_context, result)
    return ExecutedRun(
        name=name,
        config=config,
        result=result,
        run_context=run_context,
    )


__all__ = [
    "ExecutedRun",
    "ReporterStackFactory",
    "default_reporter_stack",
    "execute_experiment_run",
]
