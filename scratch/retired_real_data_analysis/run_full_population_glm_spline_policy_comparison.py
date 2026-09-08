#!/usr/bin/env python3
"""Run the seed-8 GLM policy setup with exact full-cache spline acceptance.

The comparison preserves the saved GLM run's train/test rows, bounded
softmax-linear policy, acceptance floor, initialization, and repository
first-order trust-constr optimizer. Only the response models change: exact
per-customer monotone-spline XGBoost acceptance and XGBoost financial loss.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.full_monotone_spline_cache import ShardedMonotoneSplineCache
from data.loader import load_model_artifact_pair, load_x_frame
from experiments.paths import results_root
from experiments.policy_artifacts import load_policy_artifact
from experiments.policy_utils import (
    artifact_policy_features as _artifact_policy_features,
    constant_softmax_theta as _constant_policy_theta,
    optimization_trace_summary as _trace_payload,
)
from experiments.provenance import file_record as _file_record
from experiments.slurm import submit_to_slurm_if_needed
from objective.base import Objective
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize


ACTION_LOW = -0.1
ACTION_HIGH = 0.2
BIN_WIDTH = 0.01
OPTIMIZED_COLOR = "#86002d"
EXPECTED_SEED = 8
EXPECTED_TRAIN_ROWS = 572_018
EXPECTED_TEST_ROWS = 143_005
EXPECTED_ALL_ROWS = 715_023
DEFAULT_MAX_STEPS = 100
DEFAULT_GRAD_NORM_TOL = 1e-6
DEFAULT_INITIAL_CONSTR_PENALTY = 1.0


def _default_glm_policy() -> Path:
    return (
        results_root()
        / "glm-softmax-80-20-first-order"
        / "glm-softmax-80-20-first-order"
        / "seeds"
        / "seed-8"
        / "policies"
        / "first_order"
        / "policy.json"
    )


def _default_cache() -> Path:
    return (
        results_root()
        / "cache"
        / "monotone-spline-xgb-full-v1"
        / "sweeps"
        / "full-715023-v1-20260812"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-policy", type=Path, default=_default_glm_policy())
    parser.add_argument("--spline-cache", type=Path, default=_default_cache())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=results_root() / "glm-vs-full-monotone-spline-xgb-policy-seed-8",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--launch", choices=("local", "slurm"), default="local")
    parser.add_argument("--no-sbatch", action="store_true")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate standalone spline histograms from an existing saved policy.",
    )
    parser.add_argument(
        "--all-title-parenthetical",
        default="all customers",
        help="Parenthetical title text for the all-population standalone histogram.",
    )
    return parser


class FullCacheSplineProfitObjective(Objective):
    """Exact spline-acceptance pricing objective on one fixed row cohort."""

    def __init__(
        self,
        *,
        cache: ShardedMonotoneSplineCache,
        row_indices: np.ndarray,
        loss: np.ndarray,
        premium: np.ndarray,
        policy: SoftmaxPolicy,
        acceptance_floor: float,
    ) -> None:
        self.cache = cache
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.loss = np.asarray(loss, dtype=float)
        self.premium = np.asarray(premium, dtype=float)
        self.policy = policy
        self.acceptance_floor = float(acceptance_floor)
        n_rows = self.row_indices.size
        if n_rows < 1 or self.loss.shape != (n_rows,) or self.premium.shape != (n_rows,):
            raise ValueError("rows, loss, and premium must be nonempty and aligned.")
        if not np.isfinite(self.loss).all() or not np.isfinite(self.premium).all():
            raise ValueError("loss and premium must be finite.")
        if not 0.0 < self.acceptance_floor < 1.0:
            raise ValueError("acceptance_floor must lie in (0, 1).")
        self._cache_key: bytes | None = None
        self._cache_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _components(
        self, theta: np.ndarray, x_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x_batch, dtype=float)
        if x.ndim != 2 or x.shape[0] != self.row_indices.size:
            raise ValueError("x_batch must stay aligned to the fixed objective rows.")
        theta_array = np.ascontiguousarray(np.asarray(theta, dtype=float))
        key = theta_array.tobytes()
        if self._cache_key != key or self._cache_values is None:
            actions = self.policy.value(theta_array, x)
            acceptance, d_acceptance_du = self.cache.pairwise_acceptance_and_derivative(
                self.row_indices, actions
            )
            self._cache_key = key
            self._cache_values = (actions, acceptance, d_acceptance_du)
        return self._cache_values

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions, acceptance, _ = self._components(theta, x_batch)
        cost = acceptance * (self.loss - self.premium * (1.0 + actions))
        return float(np.mean(cost))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        actions, acceptance, d_acceptance_du = self._components(theta, x_batch)
        margin = self.loss - self.premium * (1.0 + actions)
        d_cost_du = d_acceptance_du * margin - acceptance * self.premium
        return self.policy.weighted_grad(theta, x_batch, d_cost_du) / len(actions)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        _, acceptance, _ = self._components(theta, x_batch)
        return float(np.mean(acceptance))

    def mean_acceptance_grad(
        self, theta: np.ndarray, x_batch: np.ndarray
    ) -> np.ndarray:
        actions, _, d_acceptance_du = self._components(theta, x_batch)
        return self.policy.weighted_grad(theta, x_batch, d_acceptance_du) / len(actions)

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, float]:
        actions, acceptance, _ = self._components(theta, x_batch)
        return {
            "mean_acceptance": float(np.mean(acceptance)),
            "projected_loss": float(np.mean(self.loss)),
            "projected_revenue": float(np.mean(self.premium * (1.0 + actions))),
        }

    def summarize(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, Any]:
        actions, acceptance, _ = self._components(theta, x_batch)
        revenue = self.premium * (1.0 + actions)
        cost = acceptance * (self.loss - revenue)
        quantiles = np.quantile(actions, [0.01, 0.25, 0.50, 0.75, 0.99])
        return {
            "n_customers": int(len(actions)),
            "mean_action": float(np.mean(actions)),
            "population_std_action": float(np.std(actions, ddof=0)),
            "action_quantiles": {
                name: float(value)
                for name, value in zip(
                    ("p01", "p25", "p50", "p75", "p99"), quantiles, strict=True
                )
            },
            "mean_acceptance": float(np.mean(acceptance)),
            "objective_value": float(np.mean(cost)),
            "mean_predicted_profit": float(-np.mean(cost)),
            "projected_loss": float(np.mean(self.loss)),
            "projected_revenue": float(np.mean(revenue)),
        }


def _histogram(values: np.ndarray, *, series: str, population: str) -> pd.DataFrame:
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, 31, dtype=float)
    edges[-1] = np.nextafter(ACTION_HIGH, np.inf)
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    density = counts / (len(values) * np.diff(edges))
    return pd.DataFrame(
        {
            "series": series,
            "population": population,
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "density": density,
        }
    )


def _plot_single(values: np.ndarray, path: Path, *, population: str) -> None:
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, 31, dtype=float)
    edges[-1] = np.nextafter(ACTION_HIGH, np.inf)
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        values,
        bins=edges,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        alpha=0.75,
        linewidth=0.5,
    )
    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    ax.set_title(f"Optimized Price Changes — Spline/XGBoost ({population})", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _plot_comparison(glm: np.ndarray, spline: np.ndarray, path: Path) -> None:
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, 31, dtype=float)
    edges[-1] = np.nextafter(ACTION_HIGH, np.inf)
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(glm, bins=edges, density=True, alpha=0.45, linewidth=0.5, label="GLM")
    ax.hist(
        spline,
        bins=edges,
        density=True,
        alpha=0.45,
        linewidth=0.5,
        label="Monotone-spline XGBoost",
    )
    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    ax.set_title("Optimized Price Changes: GLM vs. Monotone-Spline XGBoost", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def replot_saved_standalone_histograms(
    output_dir: Path,
    *,
    all_title_parenthetical: str = "all customers",
) -> list[Path]:
    """Replay saved actions and refresh only the two standalone PDFs."""
    result_dir = output_dir.resolve()
    policy_path = result_dir / "spline_policy_seed_8.npz"
    summary_path = result_dir / "summary.json"
    with np.load(policy_path, allow_pickle=False) as payload:
        train_actions = np.asarray(payload["train_actions"], dtype=float)
        all_actions = np.asarray(payload["all_actions"], dtype=float)
        action_bounds = np.asarray(payload["action_bounds"], dtype=float)
    if not np.allclose(action_bounds, [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Saved policy bounds do not match [-0.1, 0.2].")
    if not np.isfinite(train_actions).all() or not np.isfinite(all_actions).all():
        raise ValueError("Saved policy actions must be finite.")

    train_pdf = result_dir / "optimized_price_changes_spline_train.pdf"
    all_pdf = result_dir / "optimized_price_changes_spline_all_customers.pdf"
    _plot_single(train_actions, train_pdf, population="training population")
    _plot_single(all_actions, all_pdf, population=all_title_parenthetical)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for path in (train_pdf, all_pdf):
        summary["outputs"][path.name] = _file_record(path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (train_pdf, all_pdf, summary_path):
        print(path, flush=True)
    return [train_pdf, all_pdf, summary_path]


def run_analysis(args: argparse.Namespace) -> list[Path]:
    if int(args.max_steps) <= 0 or int(args.n_jobs) == 0:
        raise ValueError("max_steps must be positive and n_jobs cannot be zero.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    glm_policy_path = args.glm_policy.resolve()
    glm_policy = load_policy_artifact(glm_policy_path)
    if glm_policy.estimator != "first_order":
        raise ValueError("The comparison GLM policy must use first_order.")
    if glm_policy.objective.model_type != "linear":
        raise ValueError("The comparison policy must use the saved linear/GLM objective.")
    if glm_policy.policy_head.type != "SoftmaxPolicy" or not np.allclose(
        [glm_policy.policy_head.action_low, glm_policy.policy_head.action_high],
        [ACTION_LOW, ACTION_HIGH],
    ):
        raise ValueError("The comparison GLM policy must be softmax-bounded to [-0.1, 0.2].")
    train_rows = np.asarray(glm_policy.row_indices("train"), dtype=np.int64)
    test_rows = np.asarray(glm_policy.row_indices("test"), dtype=np.int64)
    all_rows = np.asarray(glm_policy.row_indices("all"), dtype=np.int64)
    if (len(train_rows), len(test_rows), len(all_rows)) != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_TEST_ROWS,
        EXPECTED_ALL_ROWS,
    ):
        raise ValueError("Saved GLM split sizes do not match the seed-8 comparison contract.")
    if not np.array_equal(np.sort(np.concatenate([train_rows, test_rows])), all_rows):
        raise ValueError("Saved train/test rows do not partition the selected full cohort.")

    cache = ShardedMonotoneSplineCache(args.spline_cache.resolve(), verify_checksums=False)
    if not np.array_equal(np.asarray(cache.eligible_row_indices), all_rows):
        raise ValueError("Full spline cache rows do not match the saved GLM cohort.")
    if cache.manifest.get("validation", {}).get("status") != "passed":
        raise ValueError("Full spline cache must have passed collection validation.")

    print(f"Loading {len(all_rows):,} exact-spline customer rows...", flush=True)
    frame = load_x_frame("xgb", row_indices=all_rows)
    xgb_acceptance, xgb_loss = load_model_artifact_pair("xgb", "xgb")
    for artifact in (xgb_acceptance, xgb_loss):
        if hasattr(artifact.model, "set_params"):
            available = artifact.model.get_params(deep=False)
            updates: dict[str, Any] = {}
            if "n_jobs" in available:
                updates["n_jobs"] = int(args.n_jobs)
            if "device" in available:
                updates["device"] = "cpu"
            if updates:
                artifact.model.set_params(**updates)
    raw_policy_features = _artifact_policy_features(xgb_acceptance, frame)
    loss = np.asarray(xgb_loss.model.predict(xgb_loss.model_frame(frame)), dtype=float)
    premium = frame["X_policy_premium"].to_numpy(dtype=float)

    positions = np.searchsorted(all_rows, train_rows)
    test_positions = np.searchsorted(all_rows, test_rows)
    if not np.array_equal(all_rows[positions], train_rows) or not np.array_equal(
        all_rows[test_positions], test_rows
    ):
        raise ValueError("Saved split rows are not aligned to the full cohort.")
    preprocessor = PolicyFeaturePreprocessor(standardize=True, sphere=True)
    train_features = preprocessor.fit_transform(raw_policy_features[positions])
    test_features = preprocessor.transform(raw_policy_features[test_positions])
    all_features = preprocessor.transform(raw_policy_features)
    del raw_policy_features, frame

    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(), action_low=ACTION_LOW, action_high=ACTION_HIGH
    )
    acceptance_floor = float(glm_policy.objective.acceptance_floor)
    train_objective = FullCacheSplineProfitObjective(
        cache=cache,
        row_indices=train_rows,
        loss=loss[positions],
        premium=premium[positions],
        policy=policy,
        acceptance_floor=acceptance_floor,
    )
    theta0 = _constant_policy_theta(policy, train_features.shape[1], 0.0)
    initial = train_objective.summarize(theta0, train_features)
    if initial["mean_acceptance"] < acceptance_floor:
        raise ValueError("The matched u=0 initialization is not spline-acceptance feasible.")

    print("Running repository first-order trust-constr optimizer...", flush=True)
    started = time.perf_counter()
    theta, trace = run_first_order_minimize(
        theta0,
        train_features,
        train_objective,
        t_steps=int(args.max_steps),
        n_grad_samples=8,
        sigma=0.05,
        perturbation_space="u",
        algorithm="trust-constr",
        step_size=0.01,
        batch_size=None,
        grad_norm_tol=DEFAULT_GRAD_NORM_TOL,
        initial_constr_penalty=DEFAULT_INITIAL_CONSTR_PENALTY,
    )
    runtime = time.perf_counter() - started
    train_summary = train_objective.summarize(theta, train_features)
    test_objective = FullCacheSplineProfitObjective(
        cache=cache,
        row_indices=test_rows,
        loss=loss[test_positions],
        premium=premium[test_positions],
        policy=policy,
        acceptance_floor=acceptance_floor,
    )
    test_summary = test_objective.summarize(theta, test_features)
    all_objective = FullCacheSplineProfitObjective(
        cache=cache,
        row_indices=all_rows,
        loss=loss,
        premium=premium,
        policy=policy,
        acceptance_floor=acceptance_floor,
    )
    all_summary = all_objective.summarize(theta, all_features)
    train_actions = policy.value(theta, train_features)
    test_actions = policy.value(theta, test_features)
    all_actions = policy.value(theta, all_features)
    glm_train_actions = np.asarray(glm_policy.predict_u(split="train"), dtype=float)

    state = preprocessor.to_state()
    policy_path = output_dir / "spline_policy_seed_8.npz"
    np.savez_compressed(
        policy_path,
        theta=theta,
        train_row_indices=train_rows,
        test_row_indices=test_rows,
        selected_row_indices=all_rows,
        train_actions=train_actions,
        test_actions=test_actions,
        all_actions=all_actions,
        action_bounds=np.asarray([ACTION_LOW, ACTION_HIGH]),
        acceptance_floor=np.asarray(acceptance_floor),
        **{
            f"policy_preprocessor__{name}": value
            for name, value in state["arrays"].items()
        },
    )
    histogram_path = output_dir / "optimized_price_histograms.csv"
    pd.concat(
        [
            _histogram(glm_train_actions, series="GLM", population="train"),
            _histogram(train_actions, series="Monotone-spline XGBoost", population="train"),
            _histogram(all_actions, series="Monotone-spline XGBoost", population="all"),
        ],
        ignore_index=True,
    ).to_csv(histogram_path, index=False)
    spline_train_pdf = output_dir / "optimized_price_changes_spline_train.pdf"
    spline_all_pdf = output_dir / "optimized_price_changes_spline_all_customers.pdf"
    comparison_pdf = output_dir / "optimized_price_changes_glm_vs_spline_train.pdf"
    _plot_single(train_actions, spline_train_pdf, population="training population")
    _plot_single(all_actions, spline_all_pdf, population="all customers")
    _plot_comparison(glm_train_actions, train_actions, comparison_pdf)

    summary = {
        "analysis": "full-customer-glm-vs-monotone-spline-xgboost-policy-comparison",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison_contract": {
            "seed": EXPECTED_SEED,
            "train_fraction": 0.8,
            "test_fraction": 0.2,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "all_rows": len(all_rows),
            "same_saved_glm_row_split": True,
            "policy": "SoftmaxPolicy(IdentityFeatureMap), standardized and sphered features",
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "acceptance_floor": acceptance_floor,
            "initial_action": 0.0,
        },
        "models": {
            "acceptance": "exact full-cache monotone-spline XGBoost",
            "loss": "XGBoost financial loss",
            "raw_xgboost_acceptance_fallback": False,
            "acceptance_artifact": _file_record(Path(xgb_acceptance.artifact_path)),
            "loss_artifact": _file_record(Path(xgb_loss.artifact_path)),
            "spline_cache_manifest": _file_record(args.spline_cache.resolve() / "manifest.json"),
        },
        "optimizer": {
            "entry_point": "optimization.solvers.run_first_order_minimize",
            "step_rule": "trust-constr",
            "gradient": "first_order",
            "max_steps": int(args.max_steps),
            "n_grad_samples": 8,
            "sigma": 0.05,
            "grad_norm_tol": DEFAULT_GRAD_NORM_TOL,
            "initial_constr_penalty": DEFAULT_INITIAL_CONSTR_PENALTY,
            "runtime_seconds": runtime,
            **_trace_payload(trace),
        },
        "spline": {
            "initial_train": initial,
            "optimized_train": train_summary,
            "optimized_test": test_summary,
            "optimized_all": all_summary,
        },
        "glm_reference": {
            "policy_artifact": _file_record(glm_policy_path),
            "train_metrics": glm_policy.train_metrics.to_dict()
            if hasattr(glm_policy.train_metrics, "to_dict")
            else vars(glm_policy.train_metrics),
            "test_metrics": glm_policy.test_metrics.to_dict()
            if hasattr(glm_policy.test_metrics, "to_dict")
            else vars(glm_policy.test_metrics),
        },
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    outputs = [policy_path, histogram_path, spline_train_pdf, spline_all_pdf, comparison_pdf]
    summary["outputs"] = {path.name: _file_record(path) for path in outputs}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in [*outputs, summary_path]:
        print(path, flush=True)
    return [*outputs, summary_path]


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.replot_only:
        replot_saved_standalone_histograms(
            args.output_dir,
            all_title_parenthetical=str(args.all_title_parenthetical),
        )
        return
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    if args.launch == "slurm":
        submission = submit_to_slurm_if_needed(
            requires_jax=False,
            no_sbatch=bool(args.no_sbatch),
            argv=original_argv,
            cwd=ROOT,
        )
        if submission is not None:
            print(
                f"Submitted {submission.profile.name} Slurm job {submission.job_id}; "
                f"logs: {submission.profile.output}",
                flush=True,
            )
            return
    run_analysis(args)


if __name__ == "__main__":
    main()
