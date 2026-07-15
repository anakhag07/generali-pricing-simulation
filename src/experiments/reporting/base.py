"""Reporter interfaces and composition."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult


@runtime_checkable
class StepReporter(Protocol):
    """Protocol for per-step metric logging during optimization."""

    def log_step(
        self,
        method: str,
        step: int,
        u: float,
        value: float,
        grad_norm: float | None = None,
        step_size: float | None = None,
        mean_acceptance: float | None = None,
        projected_loss: float | None = None,
        projected_revenue: float | None = None,
        proximal_penalty: float | None = None,
        support_penalty: float | None = None,
    ) -> None:
        ...


@runtime_checkable
class Reporter(Protocol):
    """Lifecycle hooks for run reporters."""

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        ...

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        ...


class ReporterStack:
    """Composite reporter that also forwards per-step logs."""

    def __init__(self, reporters: Sequence[Reporter]) -> None:
        self._reporters = list(reporters)

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        for reporter in self._reporters:
            reporter.on_start(run_context, config)

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        for reporter in self._reporters:
            reporter.on_end(run_context, result)

    def log_step(
        self,
        method: str,
        step: int,
        u: float,
        value: float,
        grad_norm: float | None = None,
        step_size: float | None = None,
        mean_acceptance: float | None = None,
        projected_loss: float | None = None,
        projected_revenue: float | None = None,
        proximal_penalty: float | None = None,
        support_penalty: float | None = None,
    ) -> None:
        for reporter in self._reporters:
            if isinstance(reporter, StepReporter):
                reporter.log_step(
                    method,
                    step,
                    u,
                    value,
                    grad_norm,
                    step_size,
                    mean_acceptance,
                    projected_loss,
                    projected_revenue,
                    proximal_penalty,
                    support_penalty,
                )


__all__ = ["Reporter", "ReporterStack", "StepReporter"]
