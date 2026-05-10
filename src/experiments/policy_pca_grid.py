"""Policy PCA-dimension by policy-class experiment utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_GLM,
    LOSS_FEATURE_COLS,
    extract_glm_u_coef,
    load_model_artifacts,
    load_x_array,
    sample_csv_row_indices,
)
from experiments.config import CorrectnessSpec, ExperimentConfig, make_model_based_objective
from experiments.results import ExperimentResult
from experiments.run import run_experiment
from objective.policy import (
    ConstantPolicy,
    CubicFeatureMap,
    IdentityFeatureMap,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    mlp_init_theta,
)
from objective.policy_preprocessing import PolicyFeaturePreprocessor, fit_policy_feature_preprocessor
from reporting.visualization import plot_policy_pca_final_objective, plot_policy_pca_richness_gap


PCA_DIMS: tuple[int | None, ...] = (2, 4, 6, 9, None)
POLICY_CLASSES: tuple[str, ...] = (
    "constant",
    "linear",
    "quadratic",
    "third_order",
    "fourth_order",
    "mlp",
)


@dataclass(frozen=True)
class PolicyPcaGridSpec:
    """Configuration for the policy PCA-dimension experiment grid."""

    pca_dims: tuple[int | None, ...] = PCA_DIMS
    policy_classes: tuple[str, ...] = POLICY_CLASSES
    seeds: tuple[int, ...] = (42, 43, 44)
    n_samples: int = 5000
    data_seed: int = 42
    estimator: str = "first_order"
    step_rule: str = "l-bfgs-b"
    t_steps: int = 1000
    step_size: float = 0.01
    sigma: float = 0.05
    n_grad_samples: int = 50
    grad_norm_tol: float | None = 1e-6
    ftol: float | None = None
    standardize: bool = True
    sphere: bool = True
    output_root: str = "outputs"
    project_name: str = "policy-pca-grid"
    verbose: bool = True


@dataclass(frozen=True)
class PolicyPcaCondition:
    """One resolved policy/PCA/seed condition."""

    policy_class: str
    pca_dim: int | None
    seed: int
    config: ExperimentConfig
    policy_preprocessor: PolicyFeaturePreprocessor


@dataclass(frozen=True)
class PolicyPcaGridOutput:
    """Aggregate output paths and rows for a completed policy PCA grid."""

    output_dir: Path
    final_rows: list[dict[str, object]]
    trace_rows: list[dict[str, object]]


def run_policy_pca_grid(spec: PolicyPcaGridSpec | None = None) -> PolicyPcaGridOutput:
    """Run the unconstrained policy PCA grid and write aggregate outputs."""
    spec = spec or PolicyPcaGridSpec()
    output_dir = _grid_output_dir(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    acceptance_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acceptance_model)
    row_indices = sample_csv_row_indices("glm", n_rows=spec.n_samples, seed=spec.data_seed)
    x_fixed = load_x_array("glm", row_indices=row_indices)
    x_policy = _policy_raw_x(x_fixed)
    preprocessors = {
        pca_dim: fit_policy_feature_preprocessor(
            x_policy,
            standardize=spec.standardize,
            sphere=spec.sphere,
            pca_dim=pca_dim,
        )
        for pca_dim in spec.pca_dims
    }

    final_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    total_conditions = len(spec.pca_dims) * len(spec.policy_classes) * len(spec.seeds)
    condition_index = 0
    for pca_dim in spec.pca_dims:
        policy_preprocessor = preprocessors[pca_dim]
        for policy_class in spec.policy_classes:
            for seed in spec.seeds:
                condition_index += 1
                condition = build_policy_pca_condition(
                    spec=spec,
                    policy_class=policy_class,
                    pca_dim=pca_dim,
                    seed=seed,
                    x_fixed=x_fixed,
                    row_indices=row_indices,
                    acceptance_model=acceptance_model,
                    loss_model=loss_model,
                    u_coef=u_coef,
                    policy_preprocessor=policy_preprocessor,
                )
                _reset_eval_counts(condition.config.objective)
                start_time = time.perf_counter()
                if spec.verbose:
                    print(
                        "[policy-pca-grid] "
                        f"{condition_index}/{total_conditions} start "
                        f"pca_dim={_pca_dim_value(pca_dim)} "
                        f"policy={policy_class} seed={seed} "
                        f"dim_theta={condition.config.objective.policy_theta_dim()}",
                        flush=True,
                    )
                try:
                    result = run_experiment(condition.config)
                except Exception as exc:  # noqa: BLE001 - grid should keep failed conditions.
                    final_rows.append(_failure_row(condition, spec, exc))
                    if spec.verbose:
                        elapsed = time.perf_counter() - start_time
                        print(
                            "[policy-pca-grid] "
                            f"{condition_index}/{total_conditions} failed "
                            f"pca_dim={_pca_dim_value(pca_dim)} "
                            f"policy={policy_class} seed={seed} "
                            f"runtime_sec={elapsed:.2f} error={exc}",
                            flush=True,
                        )
                    continue
                final_rows.extend(_final_rows(condition, result, spec))
                trace_rows.extend(_trace_rows(condition, result, spec))
                if spec.verbose:
                    elapsed = time.perf_counter() - start_time
                    final_value = _first_final_value(result)
                    final_value_text = "" if final_value is None else f" final_value={final_value:.6f}"
                    print(
                        "[policy-pca-grid] "
                        f"{condition_index}/{total_conditions} done "
                        f"pca_dim={_pca_dim_value(pca_dim)} "
                        f"policy={policy_class} seed={seed} "
                        f"runtime_sec={elapsed:.2f}{final_value_text}",
                        flush=True,
                    )

    write_policy_pca_outputs(final_rows, trace_rows, output_dir)
    if spec.verbose:
        print(f"[policy-pca-grid] wrote outputs to {output_dir}", flush=True)
    return PolicyPcaGridOutput(output_dir=output_dir, final_rows=final_rows, trace_rows=trace_rows)


def build_policy_pca_condition(
    *,
    spec: PolicyPcaGridSpec,
    policy_class: str,
    pca_dim: int | None,
    seed: int,
    x_fixed: np.ndarray,
    row_indices: np.ndarray,
    acceptance_model: object,
    loss_model: object,
    u_coef: float,
    policy_preprocessor: PolicyFeaturePreprocessor,
) -> PolicyPcaCondition:
    """Build one runnable policy/PCA/seed condition."""
    policy = _policy_from_class(policy_class)
    theta0 = _theta0_for_policy(policy, policy_preprocessor.output_dim_, seed)
    objective = make_model_based_objective(
        policy=policy,
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
        policy_preprocessor=policy_preprocessor,
        policy_feature_cols=tuple(ACCEPTANCE_STATE_COLS),
    )
    config = ExperimentConfig(
        state_dim=len(FEATURE_COLS_GLM),
        n_samples=int(spec.n_samples),
        step_rule=spec.step_rule,
        objective=objective,
        perturbation_space="u",
        theta0=theta0,
        seed=int(seed),
        t_steps=int(spec.t_steps),
        step_size=float(spec.step_size),
        grad_norm_tol=spec.grad_norm_tol,
        ftol=spec.ftol,
        sigma=float(spec.sigma),
        n_grad_samples=int(spec.n_grad_samples),
        verbose=bool(spec.verbose),
        plot=False,
        enabled_estimators=(spec.estimator,),
        wandb_enabled=False,
        correctness=CorrectnessSpec(gradient_source="none"),
        x_fixed=x_fixed,
        x_fixed_row_indices=row_indices,
    )
    return PolicyPcaCondition(
        policy_class=policy_class,
        pca_dim=pca_dim,
        seed=seed,
        config=config,
        policy_preprocessor=policy_preprocessor,
    )


def write_policy_pca_outputs(
    final_rows: Sequence[Mapping[str, object]],
    trace_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Write aggregate CSV, markdown, and plots for a policy PCA grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "policy_pca_finals.csv", final_rows, _FINAL_FIELDNAMES)
    _write_rows(output_dir / "policy_pca_traces.csv", trace_rows, _TRACE_FIELDNAMES)
    _write_summary_markdown(output_dir / "policy_pca_summary.md", final_rows)
    plot_policy_pca_final_objective(final_rows, str(output_dir))
    plot_policy_pca_richness_gap(final_rows, str(output_dir))


def _policy_from_class(policy_class: str) -> object:
    if policy_class == "constant":
        return ConstantPolicy()
    if policy_class == "linear":
        return LinearPolicy(feature_map=IdentityFeatureMap())
    if policy_class == "quadratic":
        return LinearPolicy(feature_map=QuadraticFeatureMap())
    if policy_class == "third_order":
        return LinearPolicy(feature_map=CubicFeatureMap())
    if policy_class == "fourth_order":
        return LinearPolicy(feature_map=QuarticFeatureMap())
    if policy_class == "mlp":
        return MLPPolicy(feature_map=IdentityFeatureMap())
    raise ValueError(f"Unknown policy_class '{policy_class}'.")


def _theta0_for_policy(policy: object, input_dim: int, seed: int) -> np.ndarray | None:
    if isinstance(policy, ConstantPolicy):
        return np.zeros(1, dtype=float)
    if isinstance(policy, MLPPolicy):
        return mlp_init_theta(np.random.default_rng(seed), d_in=input_dim, hidden=policy.hidden)
    return None


def _policy_raw_x(x_fixed: np.ndarray) -> np.ndarray:
    return np.asarray(x_fixed, dtype=float)[:, : len(ACCEPTANCE_STATE_COLS)]


def _reset_eval_counts(objective: object) -> None:
    reset = getattr(objective, "reset_eval_counts", None)
    if callable(reset):
        reset()


def _eval_counts(objective: object) -> dict[str, int]:
    counts = getattr(objective, "eval_counts", None)
    if callable(counts):
        return dict(counts())
    return {}


def _condition_metadata(condition: PolicyPcaCondition, spec: PolicyPcaGridSpec) -> dict[str, object]:
    preprocessor = condition.policy_preprocessor
    return {
        "policy_class": condition.policy_class,
        "pca_dim": _pca_dim_value(condition.pca_dim),
        "standardize": bool(spec.standardize),
        "sphere": bool(spec.sphere),
        "seed": int(condition.seed),
        "estimator": spec.estimator,
        "n_samples": int(spec.n_samples),
        "dim_policy_input": int(preprocessor.output_dim_),
        "dim_theta": int(condition.config.theta0.size) if condition.config.theta0 is not None else int(condition.config.objective.policy_theta_dim()),
    }


def _final_rows(
    condition: PolicyPcaCondition,
    result: ExperimentResult,
    spec: PolicyPcaGridSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata = _condition_metadata(condition, spec)
    counts = _eval_counts(result.config.objective)
    estimated_m_evals = max(
        counts.get("acceptance_predict_calls_rows", 0),
        counts.get("loss_predict_calls_rows", 0),
    )
    for estimator, estimator_result in result.results.items():
        trace = result.traces.get(estimator)
        theta_delta = estimator_result.theta - np.asarray(result.config.theta0, dtype=float)
        rows.append(
            {
                **metadata,
                "estimator": estimator,
                "final_u": float(estimator_result.u),
                "final_value": float(estimator_result.value),
                "final_objective_sum": float(estimator_result.value) * int(result.x_samples.shape[0]),
                "runtime_sec": float(estimator_result.time),
                "mean_acceptance": _optional_float(estimator_result.mean_acceptance),
                "theta_l2_norm": float(np.linalg.norm(estimator_result.theta)),
                "theta_delta_l2_norm": float(np.linalg.norm(theta_delta)),
                "objective_value_calls": counts.get("objective_value_calls", ""),
                "estimated_m_evals": estimated_m_evals,
                "optimizer_success": bool(trace.optimizer_success) if trace is not None else "",
                "optimizer_status": trace.optimizer_status if trace is not None else "",
                "optimizer_message": trace.optimizer_message if trace is not None else "",
                "converged": bool(trace.optimizer_success) if trace is not None else False,
                "error": "",
            }
        )
    return rows


def _trace_rows(
    condition: PolicyPcaCondition,
    result: ExperimentResult,
    spec: PolicyPcaGridSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata = _condition_metadata(condition, spec)
    for estimator, trace in result.traces.items():
        for index, step in enumerate(trace.steps):
            rows.append(
                {
                    **metadata,
                    "estimator": estimator,
                    "step": int(step),
                    "u": _sequence_value(trace.u_values, index),
                    "objective": _sequence_value(trace.objective_values, index),
                    "theta_grad_norm": _optional_sequence_value(trace.theta_grad_norms, index),
                    "mean_acceptance": _optional_sequence_value(trace.mean_acceptance_values, index),
                    "step_size": _optional_sequence_value(trace.step_sizes, index),
                }
            )
    return rows


def _failure_row(condition: PolicyPcaCondition, spec: PolicyPcaGridSpec, exc: Exception) -> dict[str, object]:
    return {
        **_condition_metadata(condition, spec),
        "final_u": "",
        "final_value": "",
        "final_objective_sum": "",
        "runtime_sec": "",
        "mean_acceptance": "",
        "theta_l2_norm": "",
        "theta_delta_l2_norm": "",
        "objective_value_calls": "",
        "estimated_m_evals": "",
        "optimizer_success": False,
        "optimizer_status": "",
        "optimizer_message": "",
        "converged": False,
        "error": str(exc),
    }


def _grid_output_dir(spec: PolicyPcaGridSpec) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(spec.output_root) / spec.project_name / f"policy_pca_grid_{timestamp}"


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_summary_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| pca_dim | policy_class | seed | estimator | dim_theta | final_value | mean_acceptance | converged |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                "| {pca_dim} | {policy_class} | {seed} | {estimator} | {dim_theta} | {final_value} | {mean_acceptance} | {converged} |\n".format(
                    pca_dim=row.get("pca_dim", ""),
                    policy_class=row.get("policy_class", ""),
                    seed=row.get("seed", ""),
                    estimator=row.get("estimator", ""),
                    dim_theta=row.get("dim_theta", ""),
                    final_value=_format_float(row.get("final_value")),
                    mean_acceptance=_format_float(row.get("mean_acceptance")),
                    converged=row.get("converged", ""),
                )
            )


def _pca_dim_value(pca_dim: int | None) -> int | str:
    return "none" if pca_dim is None else int(pca_dim)


def _sequence_value(values: Sequence[float], index: int) -> float:
    return float(values[index])


def _optional_sequence_value(values: Sequence[float] | None, index: int) -> float | str:
    if values is None or index >= len(values):
        return ""
    return float(values[index])


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _format_float(value: object) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _first_final_value(result: ExperimentResult) -> float | None:
    for estimator_result in result.results.values():
        return float(estimator_result.value)
    return None


_FINAL_FIELDNAMES = [
    "pca_dim",
    "standardize",
    "sphere",
    "seed",
    "policy_class",
    "estimator",
    "n_samples",
    "dim_policy_input",
    "dim_theta",
    "final_u",
    "final_value",
    "final_objective_sum",
    "runtime_sec",
    "mean_acceptance",
    "theta_l2_norm",
    "theta_delta_l2_norm",
    "objective_value_calls",
    "estimated_m_evals",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "converged",
    "error",
]

_TRACE_FIELDNAMES = [
    "pca_dim",
    "standardize",
    "sphere",
    "seed",
    "policy_class",
    "estimator",
    "n_samples",
    "dim_policy_input",
    "dim_theta",
    "step",
    "u",
    "objective",
    "theta_grad_norm",
    "mean_acceptance",
    "step_size",
]


__all__ = [
    "PCA_DIMS",
    "POLICY_CLASSES",
    "PolicyPcaCondition",
    "PolicyPcaGridOutput",
    "PolicyPcaGridSpec",
    "build_policy_pca_condition",
    "run_policy_pca_grid",
    "write_policy_pca_outputs",
]
