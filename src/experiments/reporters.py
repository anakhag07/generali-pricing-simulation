"""Reporting utilities for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
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
from experiments.policy_artifacts import save_policy_artifacts
from reporting.logging import log_step, log_summary
from experiments.results import ExperimentResult, PolicyEvaluation
from reporting.visualization import (
    _plot_policy_acceptance_histograms,
    _plot_policy_delta_u_by_elasticity,
    _plot_policy_delta_u_histograms,
    _plot_policy_final_summary_metrics,
    _plot_policy_u_acceptance_histograms,
    _plot_policy_u_histograms,
    plot_gradient_norms,
    plot_loss_curves,
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


class PolicyArtifactReporter:
    """Reporter that writes reloadable trained-policy artifacts per estimator."""

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        objective = result.config.objective
        if not hasattr(objective, "acceptance_model") or result.train_row_indices is None:
            return None
        save_policy_artifacts(result, run_context.run_dir / "policies")


class WandbReporter:
    """Logs one wandb run per estimator so overlay charts share a 0-based step axis."""

    def __init__(self) -> None:
        self._enabled = False
        self._allowlist: set[str] | None = None
        self._wandb: object | None = None
        self._plots_enabled = True
        self._plots_dir: Path | None = None
        # Deferred — stored in on_start, used by _ensure_run
        self._config: ExperimentConfig | None = None
        self._run_context: RunContext | None = None
        self._config_payload: dict | None = None
        # Per-estimator run tracking
        self._current_run: object | None = None
        self._current_method: str | None = None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        self._enabled = bool(config.wandb_enabled)
        self._plots_enabled = bool(config.wandb_log_plots)
        self._plots_dir = run_context.plots_dir
        self._current_run = None
        self._current_method = None
        if config.wandb_estimator_allowlist is None:
            self._allowlist = None
        else:
            self._allowlist = set(config.wandb_estimator_allowlist)
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
        if not self._enabled or self._wandb is None:
            return
        # Finish the last estimator run
        if self._current_run is not None:
            self._current_run.finish()
            self._current_run = None

        # Summary run for final metrics and plots
        rc = self._run_context
        cfg = self._config
        group = cfg.wandb_group or f"{rc.experiment_name}-{rc.run_id}"
        summary_run = self._wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=group,
            job_type="summary",
            tags=list(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            name=f"{rc.experiment_name}-{rc.run_id}-summary",
            config={
                "experiment_name": rc.experiment_name,
                "run_id": rc.run_id,
                **self._config_payload,
            },
            reinit=True,
        )
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
        payload = {
            "u": float(u),
            "objective": float(value),
        }
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


class PlotReporter:
    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        return None

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        config = result.config
        if not config.plot or not result.traces:
            return
        run_context.plots_dir.mkdir(parents=True, exist_ok=True)
        optimization_dir = run_context.plots_dir / "optimization"
        policy_train_dir = run_context.plots_dir / "policy_train"
        policy_test_dir = run_context.plots_dir / "policy_test"
        optimization_dir.mkdir(parents=True, exist_ok=True)
        policy_train_dir.mkdir(parents=True, exist_ok=True)
        policy_test_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = str(optimization_dir)
        objective = config.objective
        traces = result.traces
        timings: dict[str, float] = {}

        def timed(name: str, fn: object, *args: object, **kwargs: object) -> None:
            start = time.perf_counter()
            try:
                fn(*args, **kwargs)  # type: ignore[misc]
            finally:
                timings[name] = time.perf_counter() - start

        u_star_plot = _u_star_for_plot(objective, result.u_star)
        timed(
            "loss_curves",
            plot_loss_curves,
            traces,
            plot_dir,
            u_star=u_star_plot,
            constant_u_baselines=result.constant_u_baselines,
        )
        timed("gradient_norms", plot_gradient_norms, traces, plot_dir)
        theta_by_estimator = {
            name: estimator_result.theta for name, estimator_result in result.results.items()
        }
        runtime_by_estimator = {
            name: estimator_result.time for name, estimator_result in result.results.items()
        }
        _plot_policy_diagnostics(
            result,
            result.x_samples,
            result.train_row_indices,
            str(policy_train_dir),
            "policy_train",
            theta_by_estimator,
            runtime_by_estimator,
            timed,
        )
        if result.x_test is not None:
            _plot_policy_diagnostics(
                result,
                result.x_test,
                result.test_row_indices,
                str(policy_test_dir),
                "policy_test",
                theta_by_estimator,
                runtime_by_estimator,
                timed,
            )
        if any(trace.step_sizes is not None for trace in traces.values()):
            timed("step_sizes", plot_step_sizes, traces, plot_dir)
        if config.theta0.size >= 2:
            axis_indices = (0, 1)
            axis_labels = None
            theta_path_points = [config.theta0]
            for trace in traces.values():
                if trace.theta_values:
                    theta_path_points.extend(
                        theta for theta in trace.theta_values if theta.size == config.theta0.size
                    )
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
            theta_refs = list(theta_path_points)
            theta_points = [(config.theta0, "initial")]
            for name, estimator_result in ordered_results:
                if estimator_result.theta.size == config.theta0.size:
                    theta_refs.append(estimator_result.theta)
            first_order_result = result.results.get("first_order")
            if first_order_result is not None:
                theta_points.append((first_order_result.theta, "first-order final point"))
            contour_x = _contour_x_samples(result.x_samples, objective)
            timed(
                "theta_objective_contours",
                plot_theta_objective_contours,
                contour_x,
                objective,
                config.theta0,
                plot_dir,
                axis_indices=axis_indices,
                axis_labels=axis_labels,
                theta_refs=theta_refs,
                theta_points=theta_points,
                traces=traces,
                grid_size=_contour_grid_size(objective),
            )
        with (run_context.plots_dir / "plot_timings.json").open("w", encoding="utf-8") as handle:
            json.dump({key: float(value) for key, value in timings.items()}, handle, indent=2, sort_keys=True)


def _contour_x_samples(x_samples: object, objective: object, max_rows: int = 200) -> object:
    """Return deterministic contour-evaluation rows, subsampling costly model objectives."""
    if hasattr(x_samples, "iloc"):
        x_arr = x_samples.reset_index(drop=True)
    else:
        x_arr = np.asarray(x_samples, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_samples must be a 2D array/DataFrame.")
    is_model_based = hasattr(objective, "acceptance_model") and hasattr(objective, "loss_model")
    if not is_model_based or x_arr.shape[0] <= max_rows:
        return x_arr
    indices = np.linspace(0, x_arr.shape[0] - 1, max_rows, dtype=int)
    if hasattr(x_arr, "iloc"):
        return x_arr.iloc[indices].reset_index(drop=True)
    return x_arr[indices]


def _contour_grid_size(objective: object, default_grid_size: int = 60, model_based_grid_size: int = 20) -> int:
    """Return contour grid resolution, lowering only costly model-based diagnostics."""
    is_model_based = hasattr(objective, "acceptance_model") and hasattr(objective, "loss_model")
    return model_based_grid_size if is_model_based else default_grid_size


class FileStepLogger:
    """Writes per-step metrics to a CSV file in the run directory."""

    def __init__(self) -> None:
        self._file: IO[str] | None = None
        self._path: Path | None = None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        optimization_dir = run_context.plots_dir / "optimization"
        optimization_dir.mkdir(parents=True, exist_ok=True)
        self._path = optimization_dir / "steps.csv"
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


def _observed_u_reference(
    result: ExperimentResult,
    x_samples: object,
    row_indices: np.ndarray | None,
) -> np.ndarray | None:
    objective = result.config.objective
    if result.config.x_fixed is None:
        return None
    if not hasattr(objective, "acceptance_model") or not hasattr(objective, "loss_model"):
        return None
    state_dim = int(result.config.state_dim)
    if state_dim == len(FEATURE_COLS_GLM):
        model_type = "glm"
    elif state_dim == len(FEATURE_COLS_XGB):
        model_type = "xgb"
    else:
        return None
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=int)
        if row_indices.shape != (_x_sample_count(x_samples),):
            return None
    return _load_observed_u_array(
        model_type,
        n_rows=_x_sample_count(x_samples),
        row_indices=row_indices,
        seed=result.config.seed,
    )


def _x_sample_count(x_samples: object) -> int:
    return int(x_samples.shape[0])


def _plot_policy_diagnostics(
    result: ExperimentResult,
    x_samples: object,
    row_indices: np.ndarray | None,
    plot_dir: str,
    timing_prefix: str,
    theta_by_estimator: dict[str, np.ndarray],
    runtime_by_estimator: dict[str, float],
    timed: object,
) -> None:
    observed_u = _observed_u_reference(result, x_samples, row_indices)
    if observed_u is None:
        return
    objective = result.config.objective
    timed(
        f"{timing_prefix}_policy_u_histograms",
        _plot_policy_u_histograms,
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir,
    )
    timed(
        f"{timing_prefix}_policy_acceptance_histograms",
        _plot_policy_acceptance_histograms,
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir,
    )
    timed(
        f"{timing_prefix}_policy_final_summary_metrics",
        _plot_policy_final_summary_metrics,
        x_samples,
        objective,
        theta_by_estimator,
        runtime_by_estimator,
        plot_dir,
    )
    timed(
        f"{timing_prefix}_policy_delta_u_histograms",
        _plot_policy_delta_u_histograms,
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir,
    )
    timed(
        f"{timing_prefix}_policy_delta_u_by_elasticity",
        _plot_policy_delta_u_by_elasticity,
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir,
    )
    timed(
        f"{timing_prefix}_policy_u_acceptance_histograms",
        _plot_policy_u_acceptance_histograms,
        x_samples,
        objective,
        theta_by_estimator,
        str(Path(plot_dir) / "u_acceptance"),
        acceptance_floor=result.config.acceptance_floor,
    )


def _build_summary_payload(run_context: RunContext, result: ExperimentResult) -> dict:
    estimators: dict[str, dict] = {}
    n_objective_terms = int(result.x_samples.shape[0])
    for name, estimator_result in result.results.items():
        trace = result.traces.get(name)
        theta_l2_norm = float(np.linalg.norm(estimator_result.theta))
        theta_delta_l2_norm = (
            float(np.linalg.norm(estimator_result.theta - result.config.theta0))
            if estimator_result.theta.size == result.config.theta0.size
            else None
        )
        estimator_payload = {
            "final_u": float(estimator_result.u),
            "final_value": float(estimator_result.value),
            "final_objective_sum": n_objective_terms * float(estimator_result.value),
            "runtime_sec": float(estimator_result.time),
            "theta": _as_list(estimator_result.theta),
            "theta_l2_norm": theta_l2_norm,
            "theta_delta_l2_norm": theta_delta_l2_norm,
        }
        if estimator_result.mean_acceptance is not None:
            estimator_payload["mean_acceptance"] = float(estimator_result.mean_acceptance)
        if estimator_result.constraint_violation is not None:
            estimator_payload["constraint_violation"] = float(estimator_result.constraint_violation)
        if estimator_result.acceptance_multiplier is not None:
            estimator_payload["acceptance_multiplier"] = float(estimator_result.acceptance_multiplier)
        if estimator_result.constraint_penalty is not None:
            estimator_payload["constraint_penalty"] = float(estimator_result.constraint_penalty)
        if name in result.train_metrics:
            estimator_payload["train"] = _policy_evaluation_to_dict(result.train_metrics[name])
        if name in result.test_metrics:
            estimator_payload["test"] = _policy_evaluation_to_dict(result.test_metrics[name])
        if trace is not None:
            estimator_payload["optimizer_success"] = trace.optimizer_success
            if trace.optimizer_optimality is not None:
                estimator_payload["optimizer_optimality"] = float(trace.optimizer_optimality)
            if trace.optimizer_lagrangian_grad is not None:
                estimator_payload["optimizer_lagrangian_grad"] = _as_list(
                    trace.optimizer_lagrangian_grad
                )
            estimator_payload["optimizer_status"] = trace.optimizer_status
            estimator_payload["optimizer_message"] = trace.optimizer_message
            lagrangian_diag = _final_lagrangian_diagnostics(result, estimator_result.theta, trace)
            estimator_payload.update(lagrangian_diag)
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
        "split": {
            "train_fraction": float(result.config.train_fraction),
            "test_fraction": float(result.config.test_fraction),
            "train_n_samples": int(result.x_samples.shape[0]),
            "test_n_samples": int(result.x_test.shape[0]) if result.x_test is not None else 0,
            "train_indices_head": [int(idx) for idx in result.train_indices[:10]]
            if result.train_indices is not None
            else None,
            "test_indices_head": [int(idx) for idx in result.test_indices[:10]]
            if result.test_indices is not None
            else None,
        },
    }
    if result.constant_u_baselines:
        constant_baselines_payload = [
            {
                "u": float(baseline.u),
                "value": float(baseline.value),
                "mean_acceptance": float(baseline.mean_acceptance)
                if baseline.mean_acceptance is not None
                else None,
            }
            for baseline in result.constant_u_baselines
        ]
        payload["constant_u_baselines"] = constant_baselines_payload
        best_baseline = min(result.constant_u_baselines, key=lambda baseline: baseline.value)
        payload["best_constant_u_baseline"] = {
            "u": float(best_baseline.u),
            "value": float(best_baseline.value),
            "mean_acceptance": float(best_baseline.mean_acceptance)
            if best_baseline.mean_acceptance is not None
            else None,
        }
    coeffs = extract_model_based_coefficients(
        result.config.objective.acceptance_model,
        result.config.objective.loss_model,
    ) if hasattr(result.config.objective, "acceptance_model") and hasattr(result.config.objective, "loss_model") else None
    if coeffs is not None:
        coeffs = {
            "acceptance": dict(coeffs["acceptance"]),
            "loss": dict(coeffs["loss"]),
        }
        objective = result.config.objective
        effective_u_coef = getattr(objective, "u_coef", None)
        if effective_u_coef is not None:
            artifact_u_coef = float(coeffs["acceptance"]["u_coef"])
            effective_u_coef = float(effective_u_coef)
            coeffs["acceptance"]["artifact_u_coef"] = artifact_u_coef
            coeffs["acceptance"]["effective_u_coef"] = effective_u_coef
            coeffs["acceptance"]["u_coef"] = effective_u_coef
            coeffs["acceptance"]["u_coef_is_overridden"] = bool(
                not np.isclose(effective_u_coef, artifact_u_coef)
            )
        payload["model_formulas"] = {
            "objective": "f(u; x) = p_acc(x, u) * (loss_hat(x) - (u + 1) * premium(x))",
            "acceptance": "p_acc(x, u) = sigmoid(beta_0 + beta_x^T x_acc + beta_u * u)",
            "loss": "loss_hat(x) = gamma_0 + gamma_x^T x_loss",
        }
        payload["model_coefficients"] = coeffs
    policy_artifacts = _policy_artifact_paths(run_context, result)
    if policy_artifacts:
        payload["policy_artifacts"] = policy_artifacts
    return payload


def _policy_artifact_paths(run_context: RunContext, result: ExperimentResult) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in result.results:
        policy_json = run_context.run_dir / "policies" / name / "policy.json"
        if policy_json.exists():
            paths[name] = str(policy_json.relative_to(run_context.run_dir))
    return paths


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


def _policy_evaluation_to_dict(evaluation: PolicyEvaluation) -> dict[str, float | int | None]:
    return {
        "n_samples": int(evaluation.n_samples),
        "objective_value": float(evaluation.objective_value),
        "objective_sum": float(evaluation.objective_sum),
        "mean_u": float(evaluation.mean_u),
        "u_q25": float(evaluation.u_q25),
        "u_q75": float(evaluation.u_q75),
        "mean_acceptance": float(evaluation.mean_acceptance)
        if evaluation.mean_acceptance is not None
        else None,
        "projected_loss": float(evaluation.projected_loss)
        if evaluation.projected_loss is not None
        else None,
        "projected_revenue": float(evaluation.projected_revenue)
        if evaluation.projected_revenue is not None
        else None,
    }


def _final_lagrangian_diagnostics(result: ExperimentResult, theta: np.ndarray, trace: object) -> dict:
    acceptance_multiplier = getattr(trace, "acceptance_multiplier", None)
    if acceptance_multiplier is None:
        return {}
    mean_acceptance_grad_fn = getattr(result.config.objective, "mean_acceptance_grad", None)
    if not callable(mean_acceptance_grad_fn):
        return {}
    if theta.size != result.config.theta0.size:
        return {}
    objective_grad = np.asarray(result.config.objective.grad(theta, result.x_samples), dtype=float)
    constraint_grad = np.asarray(mean_acceptance_grad_fn(theta, result.x_samples), dtype=float)
    lagrangian_grad = objective_grad - float(acceptance_multiplier) * constraint_grad
    return {
        "final_lagrangian_grad": _as_list(lagrangian_grad),
        "final_lagrangian_grad_inf_norm": float(np.linalg.norm(lagrangian_grad, ord=np.inf)),
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
    "create_run_context",
]
