"""Console reporter."""

from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult
from reporting.logging import log_step, log_summary


class ConsoleReporter:
    """Reporter that prints to terminal. Verbose mode controls per-step output."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        del config
        print(f"\n=== Running experiment: {run_context.experiment_name} ===")

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        del run_context
        log_summary(result)

    def log_step(
        self,
        method: str,
        step: int,
        u: float | None,
        value: float,
        grad_norm: float | None = None,
        step_size: float | None = None,
        mean_acceptance: float | None = None,
        projected_loss: float | None = None,
        projected_revenue: float | None = None,
    ) -> None:
        if self._verbose:
            log_step(
                method,
                step,
                u,
                value,
                grad_norm,
                step_size,
                mean_acceptance,
                projected_loss,
                projected_revenue,
            )


__all__ = ["ConsoleReporter"]
