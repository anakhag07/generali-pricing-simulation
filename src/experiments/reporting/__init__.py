"""Experiment run reporting interfaces and reporter implementations."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ConsoleReporter": "experiments.reporting.console",
    "FileStepLogger": "experiments.reporting.step_logger",
    "JsonReporter": "experiments.reporting.json_summary",
    "PlotReporter": "experiments.reporting.plots",
    "PolicyArtifactReporter": "experiments.reporting.artifacts",
    "Reporter": "experiments.reporting.base",
    "ReporterStack": "experiments.reporting.base",
    "RunContext": "experiments.reporting.context",
    "StepReporter": "experiments.reporting.base",
    "WandbReporter": "experiments.reporting.wandb",
    "build_summary_payload": "experiments.reporting.json_summary",
    "create_run_context": "experiments.reporting.context",
}

__all__ = [
    "ConsoleReporter",
    "FileStepLogger",
    "JsonReporter",
    "PlotReporter",
    "PolicyArtifactReporter",
    "Reporter",
    "ReporterStack",
    "RunContext",
    "StepReporter",
    "WandbReporter",
    "build_summary_payload",
    "create_run_context",
]


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'experiments.reporting' has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
