"""Policy-artifact reporter."""

from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.policy_artifacts import save_policy_artifacts
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult


class PolicyArtifactReporter:
    """Reporter that writes reloadable trained-policy artifacts per estimator."""

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        del run_context, config

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        objective = result.config.objective
        if not hasattr(objective, "acceptance_model") or result.train_row_indices is None:
            return
        save_policy_artifacts(result, run_context.run_dir / "policies")


__all__ = ["PolicyArtifactReporter"]
