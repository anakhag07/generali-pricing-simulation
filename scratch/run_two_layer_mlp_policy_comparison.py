"""Compare the existing two-hidden-layer MLP policy on fixed GLM and XGB objectives.

The native :class:`MLPPolicy` maps to ``(-0.5, 0.5)``.  This experiment applies
the affine transformation ``u = low + (high - low) * (mlp(x) + 0.5)`` so that
the policy smoothly spans the same ``[0, 0.16]`` action range as the capacity
sweeps.  Each model is optimized and evaluated on the same 20 deterministic
100/100 train/test splits used by the original capacity experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.config import CorrectnessSpec, ExperimentConfig  # noqa: E402
from experiments.launch import (  # noqa: E402
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    run_launch_plan,
)
from experiments.policy_capacity import (  # noqa: E402
    _build_objective,
    _load_task_resources,
    _model_frame,
    load_policy_capacity_manifest,
    split_positions,
)
from experiments.policy_validation import evaluate_policy, policy_u_values  # noqa: E402
from experiments.run import run_experiment  # noqa: E402
from experiments.slurm import SlurmProfile  # noqa: E402
from objective.base import Policy  # noqa: E402
from objective.policy import IdentityFeatureMap, MLPPolicy, mlp_init_theta  # noqa: E402
from objective.policy_preprocessing import PolicyFeaturePreprocessor  # noqa: E402


REFERENCE_MANIFEST = REPO_ROOT / "manifests" / "policy_capacity_xgb_u_0_0p16_degree_32.json"
PLAN_NAME = "policy-mlp-two-layer-glm-xgb-u-0-0p16"
MODEL_FAMILIES = ("glm", "xgb")
HIDDEN_WIDTH = 16


@dataclass(frozen=True)
class BoundedPricingMLPPolicy(Policy):
    """Affine wrapper mapping the repo's existing MLP policy to action bounds."""

    action_low: float = 0.0
    action_high: float = 0.16
    hidden: int = HIDDEN_WIDTH
    base: MLPPolicy = field(init=False, repr=False)
    kind: str = "bounded_pricing_mlp"

    def __post_init__(self) -> None:
        if not float(self.action_low) < float(self.action_high):
            raise ValueError("action_low must be smaller than action_high.")
        object.__setattr__(
            self,
            "base",
            MLPPolicy(feature_map=IdentityFeatureMap(), hidden=int(self.hidden)),
        )

    @property
    def action_span(self) -> float:
        return float(self.action_high - self.action_low)

    def theta_dim(self, state_dim: int) -> int:
        return self.base.theta_dim(state_dim)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        native = self.base.value(theta, x_batch)
        return self.action_low + self.action_span * (native + 0.5)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self.action_span * self.base.grad(theta, x_batch)

    def weighted_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        return self.action_span * self.base.weighted_grad(theta, x_batch, weights)


def _tasks(split_seeds: tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((seed, family) for seed in split_seeds for family in MODEL_FAMILIES)


def _penalty(acceptance: float, *, floor: float, weight: float, temperature: float) -> float:
    scaled_gap = (floor - acceptance) / temperature
    soft_gap = temperature * np.logaddexp(0.0, scaled_gap)
    return float(weight * soft_gap * soft_gap)


def _run_task(
    task_index: int,
    context: LaunchContext,
    *,
    force: bool,
) -> dict[str, Any]:
    manifest = load_policy_capacity_manifest(REFERENCE_MANIFEST)
    tasks = _tasks(manifest.split_seeds)
    seed, family = tasks[int(task_index)]
    condition_dir = context.sweep_dir / "splits" / f"seed-{seed:02d}" / family
    metrics_path = condition_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return {"split_seed": seed, "model": family, "metrics": str(metrics_path), "skipped": True}
    condition_dir.mkdir(parents=True, exist_ok=True)

    resources = _load_task_resources(manifest, context.sweep_dir / "cache")
    train_positions, test_positions = split_positions(manifest, seed)
    train_raw = resources.frame.iloc[train_positions].reset_index(drop=True)
    test_raw = resources.frame.iloc[test_positions].reset_index(drop=True)
    train_rows = resources.source_row_indices[train_positions]
    test_rows = resources.source_row_indices[test_positions]
    preprocessor = PolicyFeaturePreprocessor(
        standardize=True,
        sphere=False,
        pca_dim=None,
    ).fit(train_raw.loc[:, list(resources.policy_cols)].to_numpy(dtype=float))

    policy = BoundedPricingMLPPolicy(
        action_low=manifest.action_bounds[0],
        action_high=manifest.action_bounds[1],
        hidden=HIDDEN_WIDTH,
    )
    theta0 = mlp_init_theta(
        np.random.default_rng(seed),
        d_in=manifest.state_dim,
        hidden=HIDDEN_WIDTH,
    )
    train_frame = _model_frame(train_raw, family, resources.policy_cols)
    objective = _build_objective(manifest, family, policy, preprocessor, resources)
    config = ExperimentConfig(
        state_dim=manifest.state_dim,
        n_samples=manifest.train_size,
        train_fraction=1.0,
        test_fraction=0.0,
        step_rule=str(manifest.optimizer["step_rule"]),
        objective=objective,
        perturbation_space="u",
        objective_modifications=manifest.objective_modifications,
        theta0=theta0,
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

    test_frame = _model_frame(test_raw, family, resources.policy_cols)
    train_metrics = evaluate_policy(objective, estimator.theta, train_frame)
    test_metrics = evaluate_policy(objective, estimator.theta, test_frame)
    train_u = policy_u_values(objective, estimator.theta, train_frame)
    test_u = policy_u_values(objective, estimator.theta, test_frame)
    train_profit = -float(train_metrics.objective_value)
    test_profit = -float(test_metrics.objective_value)
    train_acceptance = float(train_metrics.mean_acceptance)
    test_acceptance = float(test_metrics.mean_acceptance)
    train_penalty = _penalty(
        train_acceptance,
        floor=manifest.acceptance_floor,
        weight=manifest.acceptance_penalty_weight,
        temperature=manifest.acceptance_penalty_temperature,
    )
    test_penalty = _penalty(
        test_acceptance,
        floor=manifest.acceptance_floor,
        weight=manifest.acceptance_penalty_weight,
        temperature=manifest.acceptance_penalty_temperature,
    )
    metrics: dict[str, Any] = {
        "split_seed": seed,
        "model": family,
        "policy": "two_hidden_layer_mlp",
        "hidden_width": HIDDEN_WIDTH,
        "parameter_count": policy.theta_dim(manifest.state_dim),
        "action_low": manifest.action_bounds[0],
        "action_high": manifest.action_bounds[1],
        "acceptance_floor": manifest.acceptance_floor,
        "acceptance_penalty_weight": manifest.acceptance_penalty_weight,
        "acceptance_penalty_temperature": manifest.acceptance_penalty_temperature,
        "train_profit": train_profit,
        "test_profit": test_profit,
        "generalization_gap_profit": train_profit - test_profit,
        "train_acceptance": train_acceptance,
        "test_acceptance": test_acceptance,
        "train_acceptance_violation": max(0.0, manifest.acceptance_floor - train_acceptance),
        "test_acceptance_violation": max(0.0, manifest.acceptance_floor - test_acceptance),
        "train_acceptance_penalty": train_penalty,
        "test_acceptance_penalty": test_penalty,
        "train_penalized_profit": train_profit - train_penalty,
        "test_penalized_profit": test_profit - test_penalty,
        "train_mean_u": float(np.mean(train_u)),
        "test_mean_u": float(np.mean(test_u)),
        "train_u_std": float(np.std(train_u, ddof=0)),
        "test_u_std": float(np.std(test_u, ddof=0)),
        "optimizer_runtime_sec": float(estimator.time),
        "optimizer_success": trace.optimizer_success,
        "optimizer_status": trace.optimizer_status,
        "optimizer_message": trace.optimizer_message,
        "optimizer_iterations": len(trace.steps),
        "final_gradient_norm": (
            float(trace.theta_grad_norms[-1]) if trace.theta_grad_norms else None
        ),
        "theta_l2": float(np.linalg.norm(estimator.theta)),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    state = preprocessor.to_state()
    arrays = state["arrays"]
    np.savez_compressed(
        condition_dir / "policy_arrays.npz",
        theta=np.asarray(estimator.theta, dtype=float),
        initial_theta=np.asarray(theta0, dtype=float),
        preprocessor_mean=np.asarray(arrays["mean"], dtype=float),
        preprocessor_scale=np.asarray(arrays["scale"], dtype=float),
        preprocessor_transform=np.asarray(arrays["transform_matrix"], dtype=float),
        train_source_rows=np.asarray(train_rows, dtype=int),
        test_source_rows=np.asarray(test_rows, dtype=int),
    )
    return {"split_seed": seed, "model": family, "metrics": str(metrics_path), "skipped": False}


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    ci95 = (
        float(student_t.ppf(0.975, array.size - 1) * std / np.sqrt(array.size))
        if array.size > 1
        else 0.0
    )
    return mean, std, ci95


def _collect(context: LaunchContext) -> dict[str, Any]:
    manifest = load_policy_capacity_manifest(REFERENCE_MANIFEST)
    paths = [
        context.sweep_dir / "splits" / f"seed-{seed:02d}" / family / "metrics.json"
        for seed, family in _tasks(manifest.split_seeds)
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot collect MLP comparison; missing {len(missing)} task outputs.")
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    frame = pd.DataFrame.from_records(records).sort_values(["model", "split_seed"])
    frame.to_csv(context.sweep_dir / "mlp_per_split.csv", index=False)

    summary_metrics = (
        "train_profit",
        "test_profit",
        "generalization_gap_profit",
        "train_acceptance",
        "test_acceptance",
        "train_acceptance_violation",
        "test_acceptance_violation",
        "train_acceptance_penalty",
        "test_acceptance_penalty",
        "train_penalized_profit",
        "test_penalized_profit",
        "train_mean_u",
        "test_mean_u",
        "train_u_std",
        "test_u_std",
        "optimizer_runtime_sec",
        "optimizer_iterations",
    )
    summary_records: list[dict[str, Any]] = []
    for family, group in frame.groupby("model", sort=True):
        row: dict[str, Any] = {
            "model": family,
            "n_splits": int(group.shape[0]),
            "parameter_count": int(group["parameter_count"].iloc[0]),
            "acceptance_floor": float(group["acceptance_floor"].iloc[0]),
            "optimizer_success_count": int(group["optimizer_success"].fillna(False).sum()),
            "train_acceptance_violation_count": int((group["train_acceptance_violation"] > 0).sum()),
            "test_acceptance_violation_count": int((group["test_acceptance_violation"] > 0).sum()),
        }
        for metric in summary_metrics:
            mean, std, ci95 = _mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        summary_records.append(row)
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(context.sweep_dir / "mlp_summary.csv", index=False)
    metadata = {
        "reference_manifest": str(REFERENCE_MANIFEST),
        "policy": {
            "type": "MLPPolicy",
            "hidden_layers": [HIDDEN_WIDTH, HIDDEN_WIDTH],
            "activation": "tanh",
            "native_output": "0.5 - sigmoid(z)",
            "bounded_output": "action_low + action_span * (native_output + 0.5)",
            "parameter_count": int(frame["parameter_count"].iloc[0]),
        },
        "n_splits": len(manifest.split_seeds),
        "models": list(MODEL_FAMILIES),
        "n_fits": len(paths),
    }
    with (context.sweep_dir / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return {
        "per_split_csv": str(context.sweep_dir / "mlp_per_split.csv"),
        "summary_csv": str(context.sweep_dir / "mlp_summary.csv"),
        "n_fits": len(paths),
    }


def _build_plan(*, force: bool) -> LaunchPlan:
    manifest = load_policy_capacity_manifest(REFERENCE_MANIFEST)

    def run_task(index: int, context: LaunchContext) -> Mapping[str, Any]:
        return _run_task(index, context, force=force)

    def run_all(context: LaunchContext) -> None:
        for index in range(len(_tasks(manifest.split_seeds))):
            _run_task(index, context, force=force)
        _collect(context)

    profile = SlurmProfile(
        name="policy-mlp-cpu",
        partition="mit_normal",
        time="02:00:00",
        nodes=1,
        ntasks=1,
        cpus_per_task=2,
        memory="16G",
        job_name="policy-mlp",
        output="outputs/slurm/%x-%j.out",
    )
    return LaunchPlan(
        name=PLAN_NAME,
        task_count=len(_tasks(manifest.split_seeds)),
        requires_jax=False,
        run_task=run_task,
        run_all=run_all,
        collect=_collect,
        default_launch="slurm",
        default_array=True,
        slurm_profile=profile,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing task outputs.")
    add_launch_args(parser, default_launch="slurm", default_array=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_launch_plan(
        _build_plan(force=args.force),
        args=args,
        argv=[sys.argv[0], *sys.argv[1:]],
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
