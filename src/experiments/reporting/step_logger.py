"""CSV step logger reporter."""

from __future__ import annotations

from typing import IO

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult


class FileStepLogger:
    """Writes per-step metrics to a CSV file in the run directory."""

    def __init__(self) -> None:
        self._file: IO[str] | None = None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        del config
        optimization_dir = run_context.plots_dir / "optimization"
        optimization_dir.mkdir(parents=True, exist_ok=True)
        self._file = (optimization_dir / "steps.csv").open("w", encoding="utf-8")
        self._file.write(
            "method,step,u,value,grad_norm,step_size,mean_acceptance,projected_loss,projected_revenue\n"
        )

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        del run_context, result
        if self._file is not None:
            self._file.close()
            self._file = None

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
    ) -> None:
        if self._file is None:
            return
        grad_str = f"{grad_norm:.6f}" if grad_norm is not None else ""
        step_str = f"{step_size:.6f}" if step_size is not None else ""
        acceptance_str = f"{mean_acceptance:.6f}" if mean_acceptance is not None else ""
        loss_str = f"{projected_loss:.6f}" if projected_loss is not None else ""
        revenue_str = f"{projected_revenue:.6f}" if projected_revenue is not None else ""
        self._file.write(
            f"{method},{step},{u:.6f},{value:.6f},{grad_str},{step_str},"
            f"{acceptance_str},{loss_str},{revenue_str}\n"
        )


__all__ = ["FileStepLogger"]
