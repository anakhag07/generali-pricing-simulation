"""GLM/XGBoost policy-capacity experiment on one shared customer cohort."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from data.dataset_metadata import DATASET_PATH
from data.loader import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
    load_acceptance_artifact,
    load_loss_artifact,
    load_x_frame,
)
from data.monotone_spline_xgb import (
    MonotoneSplineXGBAcceptance,
    fit_monotone_spline_artifact,
    load_monotone_spline_artifact,
    save_monotone_spline_artifact,
)
from experiments.config import CorrectnessSpec, ExperimentConfig, make_model_based_objective
from experiments.launch import LaunchContext, LaunchPlan
from experiments.policy_validation import evaluate_policy
from experiments.run import run_experiment
from experiments.slurm import SlurmProfile
from objective.policy import AdditiveChebyshevFeatureMap, SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from reporting.visualization import (
    plot_policy_capacity_endpoint_slices,
    plot_policy_capacity_generalization_gap,
    plot_policy_capacity_model_transfer,
    plot_policy_capacity_objective,
)


POLICY_CAPACITY_MANIFEST_KIND = "policy_capacity"
POLICY_INPUT_PREFIX = "__policy_input_"
MODEL_FAMILIES = ("glm", "xgb")


@dataclass(frozen=True)
class PolicyCapacityLaunchSpec:
    """Launch settings for one-fit-per-task condition arrays."""

    mode: str
    array: str
    array_max_parallel: int | None = None
    time: str = "02:00:00"
    cpus_per_task: int = 2
    memory: str = "16G"


@dataclass(frozen=True)
class PolicyCapacityManifest:
    """Resolved manifest for the GLM/XGBoost policy-capacity experiment."""

    name: str
    models: tuple[str, ...]
    degrees: tuple[int, ...]
    action_bounds: tuple[float, float]
    initial_u: float
    clip_scale: float
    acceptance_floor: float
    acceptance_penalty_weight: float
    acceptance_penalty_temperature: float
    split_seeds: tuple[int, ...]
    cohort_size: int
    train_size: int
    optimizer: dict[str, Any]
    curve_action_grid: tuple[float, ...]
    curve_dense_grid_size: int
    objective_modifications: tuple[dict[str, Any], ...]
    launch: PolicyCapacityLaunchSpec
    source_path: Path | None = None

    @property
    def state_dim(self) -> int:
        return len(ACCEPTANCE_STATE_COLS)

    def parameter_count(self, degree: int) -> int:
        return 1 + self.state_dim * int(degree)


@dataclass(frozen=True)
class _TaskResources:
    frame: pd.DataFrame
    source_row_indices: np.ndarray
    policy_cols: tuple[str, ...]
    acceptance_models: dict[str, Any]
    loss_models: dict[str, Any]


@dataclass(frozen=True)
class PolicyCapacityTask:
    """One independent optimization fit and its two evaluator replays."""

    split_seed: int
    optimize_model: str
    degree: int


def load_policy_capacity_manifest(path: str | Path) -> PolicyCapacityManifest:
    """Load and validate a policy-capacity JSON manifest."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Policy-capacity manifest must be a JSON object.")
    if payload.get("kind") != POLICY_CAPACITY_MANIFEST_KIND:
        raise ValueError(f"Expected manifest kind {POLICY_CAPACITY_MANIFEST_KIND!r}.")

    policy = _mapping(payload.get("policy"), "policy")
    acceptance = _mapping(payload.get("acceptance"), "acceptance")
    seeds = _mapping(payload.get("seeds"), "seeds")
    cohort = _mapping(payload.get("cohort"), "cohort")
    optimizer = dict(_mapping(payload.get("optimizer"), "optimizer"))
    curve = _mapping(payload.get("curve_cache"), "curve_cache")
    launch_payload = _mapping(payload.get("launch"), "launch")
    resources_payload = _mapping(launch_payload.get("resources"), "launch.resources")

    models = tuple(str(value) for value in _sequence(payload.get("models"), "models"))
    if models != MODEL_FAMILIES:
        raise ValueError("models must be exactly ['glm', 'xgb'] in that order.")
    degrees = tuple(int(value) for value in _sequence(policy.get("degrees"), "policy.degrees"))
    if not degrees or any(value < 0 for value in degrees) or tuple(sorted(set(degrees))) != degrees:
        raise ValueError("policy.degrees must be unique, sorted, and non-negative.")
    bounds = tuple(float(value) for value in _sequence(policy.get("action_bounds"), "policy.action_bounds"))
    if len(bounds) != 2 or not bounds[0] < bounds[1]:
        raise ValueError("policy.action_bounds must contain increasing lower/upper bounds.")
    initial_u = float(policy.get("initial_u"))
    if not bounds[0] < initial_u < bounds[1]:
        raise ValueError("policy.initial_u must lie strictly inside action_bounds.")
    clip_scale = float(policy.get("clip_scale"))
    if not np.isfinite(clip_scale) or clip_scale <= 0.0:
        raise ValueError("policy.clip_scale must be finite and positive.")

    split_seeds = tuple(int(value) for value in _sequence(seeds.get("split_seeds"), "seeds.split_seeds"))
    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("seeds.split_seeds must contain unique values.")
    cohort_size = int(cohort.get("size"))
    train_size = int(cohort.get("train_size"))
    if cohort.get("source") != "monotone_spline_xgb_covered":
        raise ValueError("cohort.source must be 'monotone_spline_xgb_covered'.")
    if cohort_size <= 1 or not 0 < train_size < cohort_size:
        raise ValueError("cohort sizes must define non-empty train and test splits.")

    action_grid = tuple(float(value) for value in _sequence(curve.get("action_grid"), "curve_cache.action_grid"))
    if len(action_grid) < 2 or np.any(np.diff(action_grid) <= 0.0):
        raise ValueError("curve_cache.action_grid must be strictly increasing.")
    if not np.isclose(action_grid[0], bounds[0]) or not np.isclose(action_grid[-1], bounds[1]):
        raise ValueError("curve-cache endpoints must equal the policy action bounds.")
    dense_grid_size = int(curve.get("dense_grid_size"))
    if dense_grid_size < 2:
        raise ValueError("curve_cache.dense_grid_size must be at least 2.")

    expected_optimizer = {
        "estimator": "first_order",
        "step_rule": "l-bfgs-b",
        "t_steps": 1000,
        "grad_norm_tol": 1e-6,
    }
    for key, expected in expected_optimizer.items():
        if optimizer.get(key) != expected:
            raise ValueError(f"optimizer.{key} must be {expected!r}.")

    objective_modifications = tuple(
        dict(value) for value in _sequence(payload.get("objective_modifications"), "objective_modifications", allow_empty=True)
    )
    if objective_modifications:
        raise ValueError("This experiment requires objective_modifications=[] for model comparability.")

    launch = PolicyCapacityLaunchSpec(
        mode=str(launch_payload.get("mode")),
        array=str(launch_payload.get("array")),
        array_max_parallel=(
            int(launch_payload["array_max_parallel"])
            if launch_payload.get("array_max_parallel") is not None
            else None
        ),
        time=str(resources_payload.get("time")),
        cpus_per_task=int(resources_payload.get("cpus_per_task")),
        memory=str(resources_payload.get("memory")),
    )
    if launch.mode not in {"auto", "local", "slurm"} or launch.array != "condition":
        raise ValueError("launch must use a valid mode and array='condition'.")
    if launch.cpus_per_task <= 0 or not launch.memory or not launch.time:
        raise ValueError("launch.resources must specify positive CPUs, memory, and time.")

    return PolicyCapacityManifest(
        name=str(payload["name"]),
        models=models,
        degrees=degrees,
        action_bounds=(bounds[0], bounds[1]),
        initial_u=initial_u,
        clip_scale=clip_scale,
        acceptance_floor=float(acceptance["floor"]),
        acceptance_penalty_weight=float(acceptance["penalty_weight"]),
        acceptance_penalty_temperature=float(acceptance["penalty_temperature"]),
        split_seeds=split_seeds,
        cohort_size=cohort_size,
        train_size=train_size,
        optimizer=optimizer,
        curve_action_grid=action_grid,
        curve_dense_grid_size=dense_grid_size,
        objective_modifications=objective_modifications,
        launch=launch,
        source_path=manifest_path.resolve(),
    )


def initial_theta(manifest: PolicyCapacityManifest, degree: int) -> np.ndarray:
    """Return the exact intercept-only initialization representing ``initial_u``."""
    low, high = manifest.action_bounds
    probability = (manifest.initial_u - low) / (high - low)
    theta = np.zeros(manifest.parameter_count(degree), dtype=float)
    theta[0] = math.log(probability / (1.0 - probability))
    return theta


def split_positions(manifest: PolicyCapacityManifest, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic 100/100-style train and test positions."""
    permutation = np.random.default_rng(int(seed)).permutation(manifest.cohort_size)
    train = permutation[: manifest.train_size]
    test = permutation[manifest.train_size :]
    return train, test


def policy_capacity_tasks(manifest: PolicyCapacityManifest) -> tuple[PolicyCapacityTask, ...]:
    """Return the deterministic split/model/degree task expansion."""
    return tuple(
        PolicyCapacityTask(split_seed=seed, optimize_model=model, degree=degree)
        for seed in manifest.split_seeds
        for model in manifest.models
        for degree in manifest.degrees
    )


def build_policy_capacity_launch_plan(
    manifest: PolicyCapacityManifest,
    *,
    runs_root: str | None = None,
    force: bool = False,
) -> LaunchPlan:
    """Build a one-fit-per-task launch plan for the shared manifest runner."""
    tasks = policy_capacity_tasks(manifest)

    def run_task(index: int, context: LaunchContext) -> Mapping[str, Any]:
        return run_policy_capacity_task(manifest, index, context, force=force)

    def run_all(context: LaunchContext) -> None:
        for index in range(len(tasks)):
            run_policy_capacity_task(manifest, index, context, force=force)
        collect_policy_capacity_outputs(manifest, context)

    profile = SlurmProfile(
        name="policy-capacity-cpu",
        partition="mit_normal",
        time=manifest.launch.time,
        nodes=1,
        ntasks=1,
        cpus_per_task=manifest.launch.cpus_per_task,
        memory=manifest.launch.memory,
        job_name="policy-capacity",
        output="outputs/slurm/%x-%j.out",
    )

    return LaunchPlan(
        name=manifest.name,
        task_count=len(tasks),
        requires_jax=False,
        run_task=run_task,
        run_all=run_all,
        collect=lambda context: collect_policy_capacity_outputs(manifest, context),
        runs_root=runs_root,
        default_launch=manifest.launch.mode,
        default_array=True,
        slurm_profile=profile,
    )


def run_policy_capacity_task(
    manifest: PolicyCapacityManifest,
    task_index: int,
    context: LaunchContext,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run one split/model/degree fit and replay it under both evaluators."""
    tasks = policy_capacity_tasks(manifest)
    if not 0 <= int(task_index) < len(tasks):
        raise IndexError("Policy-capacity task index is out of range.")
    task = tasks[int(task_index)]
    seed = task.split_seed
    split_dir = context.sweep_dir / "splits" / f"seed-{seed:02d}"
    condition_dir = (
        split_dir
        / "conditions"
        / task.optimize_model
        / f"degree-{task.degree:02d}"
    )
    output_csv = condition_dir / "capacity_rows.csv"
    if output_csv.exists() and not force:
        return {
            "split_seed": seed,
            "optimize_model": task.optimize_model,
            "degree": task.degree,
            "rows_csv": str(output_csv),
            "skipped": True,
        }
    condition_dir.mkdir(parents=True, exist_ok=True)

    resources = _load_task_resources(manifest, context.sweep_dir / "cache")
    train_positions, test_positions = split_positions(manifest, seed)
    train_raw = resources.frame.iloc[train_positions].reset_index(drop=True)
    test_raw = resources.frame.iloc[test_positions].reset_index(drop=True)
    train_rows = resources.source_row_indices[train_positions]
    test_rows = resources.source_row_indices[test_positions]
    policy_preprocessor = PolicyFeaturePreprocessor(
        standardize=True,
        sphere=False,
        pca_dim=None,
    ).fit(train_raw.loc[:, list(resources.policy_cols)].to_numpy(dtype=float))

    optimize_model = task.optimize_model
    degree = task.degree
    train_frame = _model_frame(train_raw, optimize_model, resources.policy_cols)
    policy = SoftmaxPolicy(
        feature_map=AdditiveChebyshevFeatureMap(
            max_degree=degree,
            clip_scale=manifest.clip_scale,
        ),
        action_low=manifest.action_bounds[0],
        action_high=manifest.action_bounds[1],
    )
    objective = _build_objective(
        manifest,
        optimize_model,
        policy,
        policy_preprocessor,
        resources,
    )
    config = ExperimentConfig(
        state_dim=manifest.state_dim,
        n_samples=manifest.train_size,
        train_fraction=1.0,
        test_fraction=0.0,
        step_rule=str(manifest.optimizer["step_rule"]),
        objective=objective,
        perturbation_space="u",
        objective_modifications=manifest.objective_modifications,
        theta0=initial_theta(manifest, degree),
        seed=seed,
        t_steps=int(manifest.optimizer["t_steps"]),
        grad_norm_tol=float(manifest.optimizer["grad_norm_tol"]),
        sigma=0.05,
        n_grad_samples=1,
        enabled_estimators=(str(manifest.optimizer["estimator"]),),
        acceptance_floor=manifest.acceptance_floor,
        acceptance_penalty_weight=manifest.acceptance_penalty_weight,
        acceptance_penalty_temperature=manifest.acceptance_penalty_temperature,
        correctness=CorrectnessSpec(gradient_source="none"),
        verbose=False,
        plot=False,
        wandb_enabled=False,
        x_fixed=train_frame,
        x_fixed_row_indices=train_rows,
    )
    result = run_experiment(config)
    estimator = result.results["first_order"]
    trace = result.traces["first_order"]
    policy_path = _save_policy_state(
        split_dir,
        optimize_model=optimize_model,
        degree=degree,
        theta=estimator.theta,
        preprocessor=policy_preprocessor,
        manifest=manifest,
        train_rows=train_rows,
        test_rows=test_rows,
    )

    rows: list[dict[str, Any]] = []
    for evaluate_model in manifest.models:
        evaluator = _build_objective(
            manifest,
            evaluate_model,
            policy,
            policy_preprocessor,
            resources,
        )
        eval_train = _model_frame(train_raw, evaluate_model, resources.policy_cols)
        eval_test = _model_frame(test_raw, evaluate_model, resources.policy_cols)
        train_metrics = evaluate_policy(evaluator, estimator.theta, eval_train)
        test_metrics = evaluate_policy(evaluator, estimator.theta, eval_test)
        rows.append(
            {
                "split_seed": seed,
                "optimize_model": optimize_model,
                "evaluate_model": evaluate_model,
                "degree": degree,
                "parameter_count": manifest.parameter_count(degree),
                "train_objective": train_metrics.objective_value,
                "test_objective": test_metrics.objective_value,
                "train_profit": -train_metrics.objective_value,
                "test_profit": -test_metrics.objective_value,
                "generalization_gap_profit": -test_metrics.objective_value
                + train_metrics.objective_value,
                "train_acceptance": train_metrics.mean_acceptance,
                "test_acceptance": test_metrics.mean_acceptance,
                "train_mean_u": train_metrics.mean_u,
                "test_mean_u": test_metrics.mean_u,
                "train_acceptance_violation": max(
                    0.0,
                    manifest.acceptance_floor - float(train_metrics.mean_acceptance),
                ),
                "optimizer_runtime_sec": estimator.time,
                "optimizer_success": trace.optimizer_success,
                "optimizer_status": trace.optimizer_status,
                "optimizer_message": trace.optimizer_message,
                "optimizer_iterations": len(trace.steps),
                "theta_l2": float(np.linalg.norm(estimator.theta)),
                "policy_state": str(policy_path.relative_to(context.sweep_dir)),
            }
        )

    _write_rows_csv(output_csv, rows)
    metadata = {
        "split_seed": seed,
        "optimize_model": optimize_model,
        "degree": degree,
        "train_source_rows": train_rows.tolist(),
        "test_source_rows": test_rows.tolist(),
        "n_fits": 1,
        "n_evaluations": len(rows),
    }
    _write_json(condition_dir / "condition_metadata.json", metadata)
    return {
        "split_seed": seed,
        "optimize_model": optimize_model,
        "degree": degree,
        "rows_csv": str(output_csv),
        "skipped": False,
    }


def collect_policy_capacity_outputs(
    manifest: PolicyCapacityManifest,
    context: LaunchContext,
) -> dict[str, Any]:
    """Collect all split outputs, compute 95% intervals, and write canonical PDFs."""
    paths = [
        context.sweep_dir
        / "splits"
        / f"seed-{task.split_seed:02d}"
        / "conditions"
        / task.optimize_model
        / f"degree-{task.degree:02d}"
        / "capacity_rows.csv"
        for task in policy_capacity_tasks(manifest)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot collect policy-capacity sweep; missing {len(missing)} split files.")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    rows_path = context.sweep_dir / "capacity_per_split.csv"
    frame.to_csv(rows_path, index=False)
    summary = summarize_policy_capacity(frame)
    summary_path = context.sweep_dir / "capacity_summary.csv"
    summary.to_csv(summary_path, index=False)

    endpoint_records = _endpoint_records(context.sweep_dir, manifest)
    for family in manifest.models:
        plot_policy_capacity_objective(summary, context.sweep_dir, family=family)
        plot_policy_capacity_generalization_gap(summary, context.sweep_dir, family=family)
        plot_policy_capacity_model_transfer(summary, context.sweep_dir, family=family)
        plot_policy_capacity_endpoint_slices(
            endpoint_records,
            context.sweep_dir,
            family=family,
        )
    _write_experiment_markdown(context.sweep_dir / "EXPERIMENT.md", manifest, frame)
    return {
        "rows_csv": str(rows_path),
        "summary_csv": str(summary_path),
        "n_rows": int(frame.shape[0]),
        "n_fits": len(manifest.models) * len(manifest.degrees) * len(manifest.split_seeds),
    }


def summarize_policy_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level metrics with Student-t 95% confidence intervals."""
    keys = ["optimize_model", "evaluate_model", "degree", "parameter_count"]
    metrics = [
        "train_objective",
        "test_objective",
        "train_profit",
        "test_profit",
        "generalization_gap_profit",
        "train_acceptance",
        "test_acceptance",
        "train_acceptance_violation",
        "optimizer_runtime_sec",
    ]
    records: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(keys, sort=True):
        record = dict(zip(keys, group_values, strict=True))
        record["n_splits"] = int(group.shape[0])
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            if values.size > 1:
                half_width = float(
                    student_t.ppf(0.975, values.size - 1)
                    * np.std(values, ddof=1)
                    / np.sqrt(values.size)
                )
            else:
                half_width = 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = half_width
        records.append(record)
    return pd.DataFrame.from_records(records)


def _load_task_resources(manifest: PolicyCapacityManifest, cache_dir: Path) -> _TaskResources:
    canonical_curves = load_acceptance_artifact("monotone_spline_xgb")
    row_indices = np.asarray(canonical_curves.covered_row_indices(), dtype=int)
    if row_indices.size != manifest.cohort_size:
        raise ValueError(
            f"Manifest cohort_size={manifest.cohort_size}, but canonical XGB curves cover {row_indices.size} rows."
        )
    frame = load_x_frame("monotone_spline_xgb", row_indices=row_indices).reset_index(drop=True)
    xgb_acceptance = load_acceptance_artifact("xgb")
    processed = xgb_acceptance.preprocessor.transform(
        frame.loc[:, list(xgb_acceptance.x_feature_cols)].copy()
    )
    processed_array = np.asarray(processed, dtype=float)
    if processed_array.shape != (manifest.cohort_size, manifest.state_dim):
        raise ValueError(
            "Shared XGB policy encoder must produce exactly one input per canonical state feature."
        )
    policy_cols = tuple(f"{POLICY_INPUT_PREFIX}{index:02d}" for index in range(manifest.state_dim))
    for index, column in enumerate(policy_cols):
        frame[column] = processed_array[:, index]

    widened_path = _ensure_widened_curve_cache(
        manifest,
        cache_dir,
        xgb_acceptance,
        row_indices,
    )
    widened = MonotoneSplineXGBAcceptance(
        load_monotone_spline_artifact(widened_path),
        xgb_acceptance,
        artifact_path=widened_path,
    )
    return _TaskResources(
        frame=frame,
        source_row_indices=row_indices,
        policy_cols=policy_cols,
        acceptance_models={
            "glm": load_acceptance_artifact("linear"),
            "xgb": widened,
        },
        loss_models={
            "glm": load_loss_artifact("linear"),
            "xgb": load_loss_artifact("xgb"),
        },
    )


def _ensure_widened_curve_cache(
    manifest: PolicyCapacityManifest,
    cache_dir: Path,
    base_xgb: Any,
    row_indices: np.ndarray,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "monotone-spline-xgb-u-minus-0.10-to-0.20.npz"
    lock_path = cache_path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            artifact = load_monotone_spline_artifact(cache_path)
            if (
                np.isclose(artifact.u_min, manifest.action_bounds[0])
                and np.isclose(artifact.u_max, manifest.action_bounds[1])
                and np.array_equal(artifact.row_indices, row_indices)
            ):
                return cache_path
            raise ValueError("Existing experiment XGB curve cache does not match the manifest.")

        dataset = pd.read_csv(DATASET_PATH, sep=";", dtype={"id": "string"})
        artifact = fit_monotone_spline_artifact(
            base_xgb,
            dataset,
            row_indices,
            base_artifact_path=str(base_xgb.artifact_path),
            action_grid=manifest.curve_action_grid,
            dense_grid_size=manifest.curve_dense_grid_size,
        )
        temporary = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
        save_monotone_spline_artifact(artifact, temporary, overwrite=True)
        os.replace(temporary, cache_path)
    return cache_path


def _build_objective(
    manifest: PolicyCapacityManifest,
    family: str,
    policy: SoftmaxPolicy,
    preprocessor: PolicyFeaturePreprocessor,
    resources: _TaskResources,
) -> Any:
    return make_model_based_objective(
        policy=policy,
        acceptance_model=resources.acceptance_models[family],
        loss_model=resources.loss_models[family],
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        u_bounds=manifest.action_bounds,
        acceptance_floor=manifest.acceptance_floor,
        acceptance_penalty_weight=manifest.acceptance_penalty_weight,
        acceptance_penalty_temperature=manifest.acceptance_penalty_temperature,
        policy_preprocessor=preprocessor,
        policy_feature_cols=resources.policy_cols,
    )


def _model_frame(frame: pd.DataFrame, family: str, policy_cols: Sequence[str]) -> pd.DataFrame:
    columns = list(ACCEPTANCE_STATE_COLS)
    if family == "xgb":
        columns.insert(0, "id")
    columns.extend(policy_cols)
    return frame.loc[:, columns].copy()


def _save_policy_state(
    split_dir: Path,
    *,
    optimize_model: str,
    degree: int,
    theta: np.ndarray,
    preprocessor: PolicyFeaturePreprocessor,
    manifest: PolicyCapacityManifest,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
) -> Path:
    policy_dir = split_dir / "policies" / optimize_model / f"degree-{degree:02d}"
    policy_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = policy_dir / "policy_arrays.npz"
    state = preprocessor.to_state()
    arrays = state["arrays"]
    np.savez_compressed(
        arrays_path,
        theta=np.asarray(theta, dtype=float),
        preprocessor_mean=np.asarray(arrays["mean"], dtype=float),
        preprocessor_scale=np.asarray(arrays["scale"], dtype=float),
        preprocessor_transform=np.asarray(arrays["transform_matrix"], dtype=float),
        train_source_rows=np.asarray(train_rows, dtype=int),
        test_source_rows=np.asarray(test_rows, dtype=int),
    )
    _write_json(
        policy_dir / "policy.json",
        {
            "optimize_model": optimize_model,
            "degree": degree,
            "parameter_count": manifest.parameter_count(degree),
            "feature_map": {
                "type": "AdditiveChebyshevFeatureMap",
                "max_degree": degree,
                "clip_scale": manifest.clip_scale,
                "ordering": "degree-major",
                "interactions": False,
            },
            "policy": {
                "type": "SoftmaxPolicy",
                "action_bounds": list(manifest.action_bounds),
            },
            "policy_preprocessor": state["metadata"],
            "arrays": arrays_path.name,
        },
    )
    return policy_dir / "policy.json"


def _endpoint_records(sweep_dir: Path, manifest: PolicyCapacityManifest) -> list[dict[str, Any]]:
    seed = manifest.split_seeds[0]
    records: list[dict[str, Any]] = []
    selected_degrees = tuple(value for value in (0, 5, 10) if value in manifest.degrees)
    for family in manifest.models:
        for degree in selected_degrees:
            arrays_path = (
                sweep_dir
                / "splits"
                / f"seed-{seed:02d}"
                / "policies"
                / family
                / f"degree-{degree:02d}"
                / "policy_arrays.npz"
            )
            with np.load(arrays_path, allow_pickle=False) as loaded:
                records.append(
                    {
                        "model": family,
                        "degree": degree,
                        "theta": loaded["theta"].copy(),
                        "state_dim": manifest.state_dim,
                        "clip_scale": manifest.clip_scale,
                        "action_low": manifest.action_bounds[0],
                        "action_high": manifest.action_bounds[1],
                    }
                )
    return records


def _write_experiment_markdown(
    path: Path,
    manifest: PolicyCapacityManifest,
    frame: pd.DataFrame,
) -> None:
    runtimes = (
        frame.drop_duplicates(["split_seed", "optimize_model", "degree"])["optimizer_runtime_sec"]
    )
    text = f"""# GLM/XGBoost policy-capacity sweep

- Policy action bounds: `{manifest.action_bounds}`
- Degrees: `{list(manifest.degrees)}`
- Parameter counts: `{[manifest.parameter_count(d) for d in manifest.degrees]}`
- Split seeds: `{list(manifest.split_seeds)}`
- Fits: `{len(manifest.models) * len(manifest.degrees) * len(manifest.split_seeds)}`
- Fixed acceptance floor: `{manifest.acceptance_floor}` (diagnostic/penalty only; not swept)
- Mean optimizer runtime per fit: `{float(runtimes.mean()):.3f}` seconds

The primary results are `objective_vs_policy_capacity_glm.pdf` and
`objective_vs_policy_capacity_xgb.pdf`. Open markers are train profit and filled
markers are held-out test profit. Acceptance is retained only as a CSV diagnostic
and is not a sweep axis or plot axis. All plots are emitted as PDF only.
"""
    path.write_text(text, encoding="utf-8")


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty policy-capacity result table.")
    fieldnames = list(rows[0].keys())
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(temporary, path)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _sequence(value: object, name: str, *, allow_empty: bool = False) -> Sequence[Any]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a{' possibly empty' if allow_empty else ' non-empty'} JSON list.")
    return value


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


__all__ = [
    "MODEL_FAMILIES",
    "POLICY_CAPACITY_MANIFEST_KIND",
    "PolicyCapacityManifest",
    "PolicyCapacityTask",
    "build_policy_capacity_launch_plan",
    "collect_policy_capacity_outputs",
    "initial_theta",
    "load_policy_capacity_manifest",
    "policy_capacity_tasks",
    "run_policy_capacity_task",
    "split_positions",
    "summarize_policy_capacity",
]
