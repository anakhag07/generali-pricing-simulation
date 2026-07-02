"""Weights & Biases reporter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult


class WandbReporter:
    """Logs one wandb run per estimator so overlay charts share a 0-based step axis."""

    def __init__(self) -> None:
        self._enabled = False
        self._allowlist: set[str] | None = None
        self._wandb: object | None = None
        self._plots_enabled = True
        self._plots_dir: Path | None = None
        self._config: ExperimentConfig | None = None
        self._run_context: RunContext | None = None
        self._config_payload: dict | None = None
        self._current_run: object | None = None
        self._current_method: str | None = None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        self._enabled = bool(config.wandb_enabled)
        self._plots_enabled = bool(config.wandb_log_plots)
        self._plots_dir = run_context.plots_dir
        self._current_run = None
        self._current_method = None
        self._allowlist = None if config.wandb_estimator_allowlist is None else set(config.wandb_estimator_allowlist)
        if not self._enabled:
            self._wandb = None
            return
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb_enabled=True but wandb is not installed. Install wandb or disable W&B."
            ) from exc
        self._wandb = wandb
        self._config = config
        self._run_context = run_context
        self._config_payload = config.to_dict()

    def _ensure_run(self, method: str) -> None:
        """Start a new wandb run when the estimator method changes."""
        if method == self._current_method:
            return
        if self._current_run is not None:
            self._current_run.finish()
        self._current_method = method
        rc = self._run_context
        cfg = self._config
        if rc is None or cfg is None or self._wandb is None or self._config_payload is None:
            return
        group = cfg.wandb_group or f"{rc.experiment_name}-{rc.run_id}"
        self._current_run = self._wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=group,
            job_type=method,
            tags=list(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            name=f"{rc.experiment_name}-{rc.run_id}-{method}",
            config={
                "experiment_name": rc.experiment_name,
                "run_id": rc.run_id,
                "estimator": method,
                **self._config_payload,
            },
            reinit=True,
        )

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        del run_context
        if not self._enabled or self._wandb is None:
            return
        if self._current_run is not None:
            self._current_run.finish()
            self._current_run = None

        rc = self._run_context
        cfg = self._config
        if rc is None or cfg is None or self._config_payload is None:
            return
        group = cfg.wandb_group or f"{rc.experiment_name}-{rc.run_id}"
        summary_run = self._wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=group,
            job_type="summary",
            tags=list(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            name=f"{rc.experiment_name}-{rc.run_id}-summary",
            config={"experiment_name": rc.experiment_name, "run_id": rc.run_id, **self._config_payload},
            reinit=True,
        )
        final_payload = {}
        for name, estimator_result in result.results.items():
            if self._allowlist is not None and name not in self._allowlist:
                continue
            final_payload[f"final/{name}/u"] = float(estimator_result.u)
            final_payload[f"final/{name}/value"] = float(estimator_result.value)
            final_payload[f"final/{name}/runtime_sec"] = float(estimator_result.time)
            final_payload[f"final/{name}/theta_l2_norm"] = float(np.linalg.norm(estimator_result.theta))
            if estimator_result.mean_acceptance is not None:
                final_payload[f"final/{name}/mean_acceptance"] = float(estimator_result.mean_acceptance)
            if estimator_result.constraint_violation is not None:
                final_payload[f"final/{name}/constraint_violation"] = float(estimator_result.constraint_violation)
            if estimator_result.acceptance_multiplier is not None:
                final_payload[f"final/{name}/acceptance_multiplier"] = float(estimator_result.acceptance_multiplier)
        if final_payload:
            summary_run.log(final_payload)

        if self._plots_enabled and self._plots_dir is not None and self._plots_dir.exists():
            plot_payload = {}
            for plot_path in sorted(self._plots_dir.rglob("*.png")):
                relative_key = plot_path.relative_to(self._plots_dir).with_suffix("").as_posix()
                plot_payload[f"plots/{relative_key}"] = self._wandb.Image(str(plot_path))
            if plot_payload:
                summary_run.log(plot_payload)
        summary_run.finish()

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
        if not self._enabled or self._wandb is None:
            return
        if self._allowlist is not None and method not in self._allowlist:
            return
        self._ensure_run(method)
        payload = {"u": float(u), "objective": float(value)}
        if grad_norm is not None:
            payload["theta_grad_norm"] = float(grad_norm)
        if step_size is not None:
            payload["step_size"] = float(step_size)
        if mean_acceptance is not None:
            payload["mean_acceptance"] = float(mean_acceptance)
        if projected_loss is not None:
            payload["projected_loss"] = float(projected_loss)
        if projected_revenue is not None:
            payload["projected_revenue"] = float(projected_revenue)
        self._current_run.log(payload, step=int(step))


__all__ = ["WandbReporter"]
