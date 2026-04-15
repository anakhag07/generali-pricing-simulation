"""Reporting utilities for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import IO, Protocol, runtime_checkable, Sequence

import numpy as np

from data.loader import (
    FEATURE_COLS_GLM,
    FEATURE_COLS_XGB,
    _load_observed_u_array,
    extract_model_based_coefficients,
)
from objective.objectives import FixedRegressionObjective
from experiments.config import ExperimentConfig
from reporting.logging import log_step, log_summary
from experiments.results import ExperimentResult
from reporting.visualization import (
    _plot_policy_u_histograms,
    _plot_policy_u_vs_objective,
    ESTIMATOR_STYLES,
    plot_gradient_norms,
    plot_loss_curves,
    plot_objective_u_slice,
    plot_step_sizes,
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
)
@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    run_id: str
    run_dir: Path
    plots_dir: Path
    started_at: datetime


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", cleaned)
    return cleaned or "run"


def create_run_context(
    experiment_name: str,
    runs_root: str = "outputs",
    started_at: datetime | None = None,
) -> RunContext:
    timestamp = started_at or datetime.now()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_name(experiment_name)
    run_dir = Path(runs_root) / safe_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    return RunContext(
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        plots_dir=plots_dir,
        started_at=timestamp,
    )


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
    ) -> None:
        ...


@runtime_checkable
class Reporter(Protocol):
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        ...

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        ...


class ReporterStack:
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
                )


class ConsoleReporter:
    """Reporter that prints to terminal. Verbose mode controls per-step output."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        print(f"\n=== Running experiment: {run_context.experiment_name} ===")

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        log_summary(result)

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


class JsonReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        payload = _build_summary_payload(run_context, result)
        summary_path = run_context.run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)


class WandbReporter:
    def __init__(self) -> None:
        self._enabled = False
        self._allowlist: set[str] | None = None
        self._wandb: object | None = None
        self._run: object | None = None
        self._global_step = 0
        self._plots_enabled = True
        self._plots_dir: Path | None = None
        self._tracked_estimators: tuple[str, ...] = ()

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        self._global_step = 0
        self._enabled = bool(config.wandb_enabled)
        self._plots_enabled = bool(config.wandb_log_plots)
        self._plots_dir = run_context.plots_dir
        if config.wandb_estimator_allowlist is None:
            self._allowlist = None
            self._tracked_estimators = tuple(config.enabled_estimators)
        else:
            self._allowlist = set(config.wandb_estimator_allowlist)
            self._tracked_estimators = tuple(config.wandb_estimator_allowlist)
        if not self._enabled:
            self._wandb = None
            self._run = None
            return
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "wandb_enabled=True but wandb is not installed. Install wandb or disable W&B."
            ) from exc

        config_payload = config.to_dict()
        self._wandb = wandb
        self._run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            job_type=config.wandb_job_type,
            tags=list(config.wandb_tags),
            mode=config.wandb_mode,
            name=f"{run_context.experiment_name}-{run_context.run_id}",
            config={
                "experiment_name": run_context.experiment_name,
                "run_id": run_context.run_id,
                **config_payload,
            },
        )
        self._define_curve_metrics()

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        if not self._enabled or self._wandb is None:
            return
        wandb_api = self._wandb
        final_payload = {}
        for name, estimator_result in result.results.items():
            if self._allowlist is not None and name not in self._allowlist:
                continue
            theta_l2_norm = float(np.linalg.norm(estimator_result.theta))
            final_payload[f"final/{name}/u"] = float(estimator_result.u)
            final_payload[f"final/{name}/value"] = float(estimator_result.value)
            final_payload[f"final/{name}/runtime_sec"] = float(estimator_result.time)
            final_payload[f"final/{name}/theta_l2_norm"] = theta_l2_norm
            if estimator_result.mean_acceptance is not None:
                final_payload[f"final/{name}/mean_acceptance"] = float(estimator_result.mean_acceptance)
        if final_payload:
            wandb_api.log(final_payload, step=self._global_step)

        if self._plots_enabled and self._plots_dir is not None and self._plots_dir.exists():
            plot_payload = {}
            for plot_path in sorted(self._plots_dir.glob("*.png")):
                plot_payload[f"plots/{plot_path.stem}"] = wandb_api.Image(str(plot_path))
            if plot_payload:
                wandb_api.log(plot_payload, step=self._global_step)
        wandb_api.finish()
        self._run = None

    def _define_curve_metrics(self) -> None:
        if self._wandb is None:
            return
        for method in self._tracked_estimators:
            step_metric = f"curve/{method}/step"
            self._wandb.define_metric(step_metric, hidden=True, summary="none")
            self._wandb.define_metric(f"curve/{method}/u", step_metric=step_metric)
            self._wandb.define_metric(f"curve/{method}/objective", step_metric=step_metric)
            self._wandb.define_metric(
                f"curve/{method}/theta_grad_norm",
                step_metric=step_metric,
            )
            self._wandb.define_metric(f"curve/{method}/step_size", step_metric=step_metric)
            self._wandb.define_metric(f"curve/{method}/mean_acceptance", step_metric=step_metric)
            self._wandb.define_metric(f"curve/{method}/projected_loss", step_metric=step_metric)
            self._wandb.define_metric(f"curve/{method}/projected_revenue", step_metric=step_metric)

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
        payload = {
            f"curve/{method}/step": int(step),
            f"curve/{method}/u": float(u),
            f"curve/{method}/objective": float(value),
        }
        if grad_norm is not None:
            payload[f"curve/{method}/theta_grad_norm"] = float(grad_norm)
        if step_size is not None:
            payload[f"curve/{method}/step_size"] = float(step_size)
        if mean_acceptance is not None:
            payload[f"curve/{method}/mean_acceptance"] = float(mean_acceptance)
        if projected_loss is not None:
            payload[f"curve/{method}/projected_loss"] = float(projected_loss)
        if projected_revenue is not None:
            payload[f"curve/{method}/projected_revenue"] = float(projected_revenue)
        self._global_step += 1
        self._wandb.log(payload)


class PlotReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        config = result.config
        if not config.plot or not result.traces:
            return
        run_context.plots_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = str(run_context.plots_dir)
        objective = config.objective
        action_objective = getattr(objective, "action_objective", None)
        traces = result.traces
        u_star_plot = _u_star_for_plot(action_objective, result.u_star)
        plot_loss_curves(
            traces,
            plot_dir,
            u_star=u_star_plot,
        )
        plot_gradient_norms(traces, plot_dir)
        observed_u = _observed_u_reference(result)
        if observed_u is not None:
            theta_by_estimator = {
                name: estimator_result.theta for name, estimator_result in result.results.items()
            }
            _plot_policy_u_histograms(
                observed_u,
                result.x_samples,
                objective,
                theta_by_estimator,
                plot_dir,
            )
            _plot_policy_u_vs_objective(
                observed_u,
                result.x_samples,
                objective,
                theta_by_estimator,
                plot_dir,
            )
        if action_objective is not None:
            plot_objective_u_slice(
                result.x_samples,
                action_objective,
                traces,
                plot_dir,
                u_star=u_star_plot,
            )
        if any(trace.step_sizes is not None for trace in traces.values()):
            plot_step_sizes(traces, plot_dir)
        if config.theta0.size >= 2:
            axis_indices = (0, 1)
            axis_labels = None
            theta_path_points = [config.theta0]
            for trace in traces.values():
                if trace.theta_values:
                    theta_path_points.extend(trace.theta_values)
            if config.theta0.size > 2 and theta_path_points:
                axis_indices = select_theta_axes_max_variance(theta_path_points)
                axis_labels = (
                    f"theta[{axis_indices[0]}] (max-var axis)",
                    f"theta[{axis_indices[1]}] (max-var axis)",
                )
            ordered_results = [
                (name, result.results[name])
                for name in config.enabled_estimators
                if name in result.results
            ]
            theta_refs = [config.theta0]
            theta_points = [(config.theta0, "initial", "#636363", "o")]
            for name, estimator_result in ordered_results:
                theta_refs.append(estimator_result.theta)
                style = ESTIMATOR_STYLES[name]
                theta_points.append(
                    (
                        estimator_result.theta,
                        style["label"],
                        style["color"],
                        style["marker"],
                    )
                )
            plot_theta_objective_contours(
                result.x_samples,
                objective,
                config.theta0,
                plot_dir,
                axis_indices=axis_indices,
                axis_labels=axis_labels,
                theta_refs=theta_refs,
                theta_points=theta_points,
                traces=traces,
            )


class FileStepLogger:
    """Writes per-step metrics to a CSV file in the run directory."""

    def __init__(self) -> None:
        self._file: IO[str] | None = None
        self._path: Path | None = None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        self._path = run_context.run_dir / "steps.csv"
        self._file = self._path.open("w", encoding="utf-8")
        self._file.write(
            "method,step,u,value,grad_norm,step_size,mean_acceptance,projected_loss,projected_revenue\n"
        )  # type: ignore[union-attr]

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
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


def _u_star_for_plot(
    action_objective: object,
    u_star: float | None,
) -> float | None:
    if isinstance(action_objective, FixedRegressionObjective):
        return None
    if u_star is not None:
        return u_star
    return None


def _observed_u_reference(result: ExperimentResult) -> np.ndarray | None:
    objective = result.config.objective
    if result.config.x_fixed is None:
        return None
    if not hasattr(objective, "acceptance_model") or not hasattr(objective, "loss_model"):
        return None
    state_dim = int(result.x_samples.shape[1])
    if state_dim == len(FEATURE_COLS_GLM):
        model_type = "glm"
    elif state_dim == len(FEATURE_COLS_XGB):
        model_type = "xgb"
    else:
        return None
    return _load_observed_u_array(model_type, n_rows=result.x_samples.shape[0])


def _build_summary_payload(run_context: RunContext, result: ExperimentResult) -> dict:
    estimators: dict[str, dict] = {}
    for name, estimator_result in result.results.items():
        trace = result.traces.get(name)
        theta_l2_norm = float(np.linalg.norm(estimator_result.theta))
        theta_delta_l2_norm = float(
            np.linalg.norm(estimator_result.theta - result.config.theta0)
        )
        estimator_payload = {
            "final_u": float(estimator_result.u),
            "final_value": float(estimator_result.value),
            "runtime_sec": float(estimator_result.time),
            "theta": _as_list(estimator_result.theta),
            "theta_l2_norm": theta_l2_norm,
            "theta_delta_l2_norm": theta_delta_l2_norm,
        }
        if estimator_result.mean_acceptance is not None:
            estimator_payload["mean_acceptance"] = float(estimator_result.mean_acceptance)
        if trace is not None:
            estimator_payload["optimizer_status"] = trace.optimizer_status
            estimator_payload["optimizer_message"] = trace.optimizer_message
        estimators[name] = estimator_payload

    trace_summary: dict[str, dict] = {}
    for name, trace in result.traces.items():
        if trace.objective_values:
            trace_summary[name] = {
                "steps": len(trace.steps),
                "final_objective": float(trace.objective_values[-1]),
                "min_objective": float(np.min(trace.objective_values)),
            }

    payload = {
        "run": {
            "experiment_name": run_context.experiment_name,
            "run_id": run_context.run_id,
            "started_at": run_context.started_at.isoformat(),
            "run_dir": str(run_context.run_dir),
        },
        "config": result.config.to_dict(),
        "initial_value": float(result.initial_value),
        "initial_mean_acceptance": float(result.initial_mean_acceptance)
        if result.initial_mean_acceptance is not None
        else None,
        "u_star": float(result.u_star) if result.u_star is not None else None,
        "value_at_u_star": float(result.value_at_u_star)
        if result.value_at_u_star is not None
        else None,
        "estimators": estimators,
        "trace_summary": trace_summary,
    }
    coeffs = extract_model_based_coefficients(
        result.config.objective.acceptance_model,
        result.config.objective.loss_model,
    ) if hasattr(result.config.objective, "acceptance_model") and hasattr(result.config.objective, "loss_model") else None
    if coeffs is not None:
        payload["model_formulas"] = {
            "objective": "f(u; x) = p_acc(x, u) * (loss_hat(x) - u * premium(x))",
            "churn": "p_churn(x, u) = sigmoid(beta_0 + beta_x^T x_acc + beta_u * u)",
            "acceptance": "p_acc(x, u) = 1 - p_churn(x, u)",
            "loss": "loss_hat(x) = gamma_0 + gamma_x^T x_loss",
        }
        payload["model_coefficients"] = coeffs
    return payload


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


__all__ = [
    "ConsoleReporter",
    "FileStepLogger",
    "JsonReporter",
    "PlotReporter",
    "Reporter",
    "ReporterStack",
    "RunContext",
    "StepReporter",
    "WandbReporter",
    "create_run_context",
]
