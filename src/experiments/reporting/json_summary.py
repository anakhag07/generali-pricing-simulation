"""JSON summary reporter and serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from data.loader import extract_model_based_coefficients
from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult, PolicyEvaluation
from objective.noise import NoisyObjective


class JsonReporter:
    """Writes the run summary JSON artifact.

    ``summary_name`` names the file (seed sweeps pass ``summary-seed-<seed>.json``);
    ``summary_dir`` overrides where it is written (defaulting to the run directory)
    so per-seed summaries can share one variant-level folder.
    """

    def __init__(self, summary_name: str = "summary.json", summary_dir: Path | None = None) -> None:
        self._summary_name = summary_name
        self._summary_dir = Path(summary_dir) if summary_dir is not None else None

    def on_start(self, run_context: RunContext, config: ExperimentConfig) -> None:
        del run_context, config

    def on_end(self, run_context: RunContext, result: ExperimentResult) -> None:
        summary_dir = self._summary_dir if self._summary_dir is not None else run_context.run_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
        payload = build_summary_payload(run_context, result, summary_dir=summary_dir)
        with (summary_dir / self._summary_name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)


def build_summary_payload(
    run_context: RunContext,
    result: ExperimentResult,
    summary_dir: Path | None = None,
) -> dict:
    """Build the serializable summary payload for one completed run.

    ``summary_dir`` is the directory the summary file is written to; policy
    artifact paths are recorded relative to it so they resolve against the
    summary's own location. Defaults to ``run_context.run_dir``.
    """
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
                estimator_payload["optimizer_lagrangian_grad"] = _as_list(trace.optimizer_lagrangian_grad)
            estimator_payload["optimizer_status"] = trace.optimizer_status
            estimator_payload["optimizer_message"] = trace.optimizer_message
            estimator_payload.update(_final_lagrangian_diagnostics(result, estimator_result.theta, trace))
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
        "initial_mean_acceptance": (
            float(result.initial_mean_acceptance) if result.initial_mean_acceptance is not None else None
        ),
        "u_star": float(result.u_star) if result.u_star is not None else None,
        "value_at_u_star": float(result.value_at_u_star) if result.value_at_u_star is not None else None,
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
    run_metadata = getattr(run_context, "run_metadata", None)
    if run_metadata:
        payload["preset"] = _preset_payload(run_metadata)
    if result.constant_u_baselines:
        payload["constant_u_baselines"] = [
            {
                "u": float(baseline.u),
                "value": float(baseline.value),
                "mean_acceptance": (
                    float(baseline.mean_acceptance) if baseline.mean_acceptance is not None else None
                ),
            }
            for baseline in result.constant_u_baselines
        ]
        best_baseline = min(result.constant_u_baselines, key=lambda baseline: baseline.value)
        payload["best_constant_u_baseline"] = {
            "u": float(best_baseline.u),
            "value": float(best_baseline.value),
            "mean_acceptance": (
                float(best_baseline.mean_acceptance) if best_baseline.mean_acceptance is not None else None
            ),
        }
    coeffs = (
        extract_model_based_coefficients(
            result.config.objective.acceptance_model,
            result.config.objective.loss_model,
        )
        if hasattr(result.config.objective, "acceptance_model")
        and hasattr(result.config.objective, "loss_model")
        else None
    )
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
    policy_artifacts = _policy_artifact_paths(run_context, result, summary_dir)
    if policy_artifacts:
        payload["policy_artifacts"] = policy_artifacts
    return payload


def _policy_artifact_paths(
    run_context: RunContext,
    result: ExperimentResult,
    summary_dir: Path | None = None,
) -> dict[str, str]:
    # Artifacts always live under run_dir, but paths are recorded relative to
    # summary_dir (where summary.json is written) so they resolve against it.
    base_dir = run_context.run_dir if summary_dir is None else summary_dir
    paths: dict[str, str] = {}
    for name in result.results:
        policy_json = run_context.run_dir / "policies" / name / "policy.json"
        if policy_json.exists():
            paths[name] = os.path.relpath(policy_json, base_dir)
    return paths


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


def _preset_payload(run_metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "preset_name" in run_metadata:
        payload["preset_name"] = _serialize_metadata_value(run_metadata["preset_name"])
    if "overrides" in run_metadata:
        payload["overrides"] = _serialize_overrides(run_metadata["overrides"])
    for key, value in run_metadata.items():
        key_text = str(key)
        if key_text in {"preset_name", "overrides"}:
            continue
        payload[key_text] = _serialize_metadata_value(value)
    return payload


def _serialize_overrides(overrides: object) -> dict[str, Any]:
    if not isinstance(overrides, Mapping):
        return {}
    return {
        str(key): _serialize_metadata_value(value)
        for key, value in overrides.items()
        if str(key) != "_run_name"
    }


def _serialize_metadata_value(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _serialize_metadata_value(value.tolist())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _serialize_metadata_value(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_metadata_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_metadata_value(item)
            for key, item in value.items()
            if str(key) != "_run_name"
        }
    if isinstance(value, tuple):
        return [_serialize_metadata_value(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_metadata_value(item) for item in value]
    return str(value)


def _policy_evaluation_to_dict(evaluation: PolicyEvaluation) -> dict[str, float | int | None]:
    return {
        "n_samples": int(evaluation.n_samples),
        "objective_value": float(evaluation.objective_value),
        "objective_sum": float(evaluation.objective_sum),
        "mean_u": float(evaluation.mean_u),
        "u_q25": float(evaluation.u_q25),
        "u_q75": float(evaluation.u_q75),
        "mean_acceptance": float(evaluation.mean_acceptance) if evaluation.mean_acceptance is not None else None,
        "projected_loss": float(evaluation.projected_loss) if evaluation.projected_loss is not None else None,
        "projected_revenue": float(evaluation.projected_revenue) if evaluation.projected_revenue is not None else None,
    }


def _final_lagrangian_diagnostics(result: ExperimentResult, theta: np.ndarray, trace: object) -> dict:
    acceptance_multiplier = getattr(trace, "acceptance_multiplier", None)
    if acceptance_multiplier is None:
        return {}
    if isinstance(result.config.objective, NoisyObjective):
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


# Private alias retained only for tests that inspect summary payload internals.
_build_summary_payload = build_summary_payload


__all__ = ["JsonReporter", "build_summary_payload"]
