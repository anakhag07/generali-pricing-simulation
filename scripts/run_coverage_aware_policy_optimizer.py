#!/usr/bin/env python3
"""Refit the 20k XGBoost pricing policy with customer-specific coverage penalties."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.dataset_metadata import (  # noqa: E402
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
)
from data.loader import (  # noqa: E402
    load_model_artifacts,
    load_observed_u_array,
    load_x_frame,
)
from experiments.config import make_model_based_objective  # noqa: E402
from objective.policy import SoftmaxPolicy  # noqa: E402
from scripts.plot_customer_coverage_envelope_slides import (  # noqa: E402
    ACTION_BANDWIDTH,
    U_GRID,
    _local_support_matrix,
    _mixed_customer_embedding,
    _predict_acceptance_matrix,
    _predict_loss,
)


RESULTS_ROOT = REPO_ROOT.parent / "results"
DEFAULT_DIAGNOSTICS = (
    RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_BASELINE_POLICY = (
    RESULTS_ROOT
    / "xgboost-full-dataset-historical-support"
    / "optimized_policy_cropped.npz"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "coverage-aware-policy-20k"
DEFAULT_WIDTH_SCALE = 10.0
DEFAULT_N_NEIGHBORS = 500
DEFAULT_MAXITER = 300
DEFAULT_FD_EPS = 1e-4
DEFAULT_ACCEPTANCE_BUFFER = 2e-4


def normalize_coverage_widths(
    support: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Return per-customer widths in ``[0, scale]`` from local support."""
    support_arr = np.asarray(support, dtype=float)
    if support_arr.ndim != 2 or support_arr.shape[1] < 2:
        raise ValueError("support must be a 2D customer-by-action array.")
    if not np.isfinite(support_arr).all() or np.any(support_arr < 0.0):
        raise ValueError("support must contain finite nonnegative values.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")
    row_max = np.max(support_arr, axis=1, keepdims=True)
    if np.any(row_max <= 0.0):
        raise ValueError("Every customer must have positive support at some action.")
    return (float(scale) * (1.0 - support_arr / row_max)).astype(np.float32)


def interpolate_rows(
    values: np.ndarray,
    grid: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate one action-grid curve per customer and its slope."""
    values_arr = np.asarray(values, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)
    actions_arr = np.asarray(actions, dtype=float).reshape(-1)
    if values_arr.shape != (len(actions_arr), len(grid_arr)):
        raise ValueError("values must have one grid curve per action.")
    if len(grid_arr) < 2 or not np.all(np.diff(grid_arr) > 0.0):
        raise ValueError("grid must be strictly increasing with at least two points.")

    clipped = np.clip(actions_arr, grid_arr[0], grid_arr[-1])
    right = np.searchsorted(grid_arr, clipped, side="right")
    left = np.clip(right - 1, 0, len(grid_arr) - 2)
    right = left + 1
    row = np.arange(len(actions_arr))
    span = grid_arr[right] - grid_arr[left]
    fraction = (clipped - grid_arr[left]) / span
    low = values_arr[row, left]
    high = values_arr[row, right]
    interpolated = low + fraction * (high - low)
    slope = (high - low) / span
    return interpolated, slope


class CoveragePolicyEvaluator:
    """Cache exact objective, gradient, and constraint values at one theta."""

    def __init__(
        self,
        *,
        objective: Any,
        frame: pd.DataFrame,
        widths: np.ndarray,
        profit_grid: np.ndarray,
        acceptance_grid: np.ndarray,
        u_grid: np.ndarray,
    ) -> None:
        self.objective = objective
        self.frame = frame
        self.widths = np.asarray(widths, dtype=float)
        self.profit_grid = np.asarray(profit_grid, dtype=float)
        self.acceptance_grid = np.asarray(acceptance_grid, dtype=float)
        self.u_grid = np.asarray(u_grid, dtype=float)
        self._theta: np.ndarray | None = None
        self._metrics: dict[str, Any] | None = None

    def evaluate(self, theta: np.ndarray) -> dict[str, Any]:
        theta_arr = np.asarray(theta, dtype=float)
        if self._theta is not None and np.array_equal(theta_arr, self._theta):
            assert self._metrics is not None
            return self._metrics

        actions = np.asarray(
            self.objective.policy_value(theta_arr, self.frame), dtype=float
        )
        profit, d_profit_du = interpolate_rows(
            self.profit_grid,
            self.u_grid,
            actions,
        )
        acceptance, d_acceptance_du = interpolate_rows(
            self.acceptance_grid,
            self.u_grid,
            actions,
        )
        width, d_width_du = interpolate_rows(self.widths, self.u_grid, actions)
        grad_u = -d_profit_du + d_width_du
        objective_grad = (
            self.objective.policy_weighted_grad(theta_arr, self.frame, grad_u)
            / len(actions)
        )
        acceptance_grad = (
            self.objective.policy_weighted_grad(
                theta_arr,
                self.frame,
                d_acceptance_du,
            )
            / len(actions)
        )

        metrics = {
            "actions": actions,
            "acceptance": acceptance,
            "base_cost": -profit,
            "coverage_width": width,
            "objective": float(np.mean(-profit + width)),
            "raw_profit": float(np.mean(profit)),
            "mean_width": float(np.mean(width)),
            "mean_acceptance": float(np.mean(acceptance)),
            "objective_grad": np.asarray(objective_grad, dtype=float),
            "acceptance_grad": np.asarray(acceptance_grad, dtype=float),
        }
        self._theta = theta_arr.copy()
        self._metrics = metrics
        return metrics

    def fun(self, theta: np.ndarray) -> float:
        return float(self.evaluate(theta)["objective"])

    def jac(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self.evaluate(theta)["objective_grad"], dtype=float)

    def mean_acceptance(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray([self.evaluate(theta)["mean_acceptance"]], dtype=float)

    def mean_acceptance_jac(self, theta: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self.evaluate(theta)["acceptance_grad"])


def _load_sample_rows(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as diagnostics:
        rows = diagnostics["row_indices"].astype(int)
    if len(rows) != 20_000 or len(np.unique(rows)) != len(rows):
        raise ValueError("Expected the deterministic 20,000-row diagnostic sample.")
    return rows


def _load_baseline(path: Path, sample_rows: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        policy_rows = saved["row_indices"].astype(int)
        positions = np.searchsorted(policy_rows, sample_rows)
        if np.any(positions >= len(policy_rows)) or not np.array_equal(
            policy_rows[positions], sample_rows
        ):
            raise ValueError("Diagnostic rows are missing from the baseline policy.")
        return {
            "theta": saved["theta"].astype(float),
            "actions": saved["actions"][positions].astype(float),
            "acceptance_floor": float(saved["acceptance_floor"]),
        }


def _load_or_build_widths(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    observed_u: np.ndarray,
    acceptance_artifact: Any,
    n_neighbors: int,
    n_jobs: int,
    width_scale: float,
    force: bool,
) -> np.ndarray:
    cache_path = output_dir / "customer_coverage_widths.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_rows = saved["row_indices"]
            widths = saved["widths"]
            cached_scale = float(saved["width_scale"])
            cached_neighbors = int(saved["n_neighbors"])
        if (
            np.array_equal(cached_rows, row_indices)
            and np.isclose(cached_scale, width_scale)
            and cached_neighbors == n_neighbors
            and widths.shape == (len(row_indices), len(U_GRID))
        ):
            print(f"Reusing {cache_path}", flush=True)
            return widths.astype(np.float32)

    print("Computing customer embeddings and local coverage support...", flush=True)
    embedding = _mixed_customer_embedding(acceptance_artifact, frame)
    support = _local_support_matrix(
        embedding,
        observed_u,
        U_GRID,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    widths = normalize_coverage_widths(support, scale=width_scale)
    np.savez_compressed(
        cache_path,
        row_indices=row_indices,
        u_grid=U_GRID,
        widths=widths,
        width_scale=float(width_scale),
        n_neighbors=int(n_neighbors),
        action_bandwidth=float(ACTION_BANDWIDTH),
    )
    print(f"Wrote {cache_path}", flush=True)
    return widths


def _load_or_build_response_grid(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    acceptance_artifact: Any,
    loss_artifact: Any,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = output_dir / "customer_response_grid.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_rows = saved["row_indices"]
            acceptance = saved["acceptance"]
            profit = saved["profit"]
        if (
            np.array_equal(cached_rows, row_indices)
            and acceptance.shape == (len(row_indices), len(U_GRID))
            and profit.shape == acceptance.shape
        ):
            print(f"Reusing {cache_path}", flush=True)
            return acceptance.astype(np.float32), profit.astype(np.float32)

    print("Computing the customer-by-action XGBoost response grid...", flush=True)
    acceptance = _predict_acceptance_matrix(acceptance_artifact, frame, U_GRID)
    loss = _predict_loss(loss_artifact, frame)
    premium = frame[PREMIUM_COL].to_numpy(dtype=float)
    revenue = premium[:, None] * (1.0 + U_GRID[None, :])
    profit = acceptance * (revenue - loss[:, None])
    np.savez_compressed(
        cache_path,
        row_indices=row_indices,
        u_grid=U_GRID,
        acceptance=acceptance.astype(np.float32),
        profit=profit.astype(np.float32),
    )
    print(f"Wrote {cache_path}", flush=True)
    return acceptance.astype(np.float32), profit.astype(np.float32)


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    actions = np.asarray(metrics["actions"], dtype=float)
    return {
        "mean_u": float(np.mean(actions)),
        "q05_u": float(np.quantile(actions, 0.05)),
        "median_u": float(np.median(actions)),
        "q95_u": float(np.quantile(actions, 0.95)),
        "mean_acceptance": float(metrics["mean_acceptance"]),
        "raw_mean_profit": float(metrics["raw_profit"]),
        "mean_coverage_width": float(metrics["mean_width"]),
        "coverage_adjusted_profit": float(metrics["raw_profit"] - metrics["mean_width"]),
    }


def _save_histograms(
    baseline_actions: np.ndarray,
    coverage_actions: np.ndarray,
    output_dir: Path,
) -> None:
    bins = np.linspace(0.0, 0.16, 33)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes[0].hist(baseline_actions, bins=bins)
    axes[0].set_title("Original Policy", fontsize=14)
    axes[1].hist(coverage_actions, bins=bins)
    axes[1].set_title("Coverage-Aware Policy", fontsize=14)
    for axis in axes:
        axis.set_xlabel("Optimizer Price Change", fontsize=12)
        axis.set_ylabel("Number of Customers", fontsize=12)
        axis.tick_params(labelsize=10)
    fig.suptitle(
        "Optimizer Actions Before and After Customer-Specific Coverage Adjustment",
        fontsize=16,
    )
    fig.savefig(
        output_dir / "optimizer_histograms_baseline_vs_coverage_aware.pdf",
        format="pdf",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--baseline-policy", type=Path, default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--width-scale", type=float, default=DEFAULT_WIDTH_SCALE)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--fd-eps", type=float, default=DEFAULT_FD_EPS)
    parser.add_argument(
        "--acceptance-buffer",
        type=float,
        default=DEFAULT_ACCEPTANCE_BUFFER,
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--force-coverage", action="store_true")
    args = parser.parse_args()
    if args.acceptance_buffer < 0.0:
        raise ValueError("acceptance-buffer must be nonnegative.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_indices = _load_sample_rows(args.diagnostics)
    baseline = _load_baseline(args.baseline_policy, row_indices)
    frame = load_x_frame("xgb", row_indices=row_indices)
    observed_u = load_observed_u_array("xgb", row_indices=row_indices)
    acceptance_artifact, loss_artifact = load_model_artifacts("xgb")
    acceptance_artifact.model.set_params(n_jobs=int(args.xgb_n_jobs))
    loss_artifact.model.set_params(n_jobs=int(args.xgb_n_jobs))

    policy = SoftmaxPolicy(action_low=0.0, action_high=0.16)
    objective = make_model_based_objective(
        policy=policy,
        acceptance_model=acceptance_artifact,
        loss_model=loss_artifact,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        u_bounds=(0.0, 0.16),
        acceptance_floor=baseline["acceptance_floor"],
    )
    object.__setattr__(objective, "_fd_eps", float(args.fd_eps))
    widths = _load_or_build_widths(
        output_dir=args.output_dir,
        frame=frame,
        row_indices=row_indices,
        observed_u=observed_u,
        acceptance_artifact=acceptance_artifact,
        n_neighbors=int(args.n_neighbors),
        n_jobs=int(args.xgb_n_jobs),
        width_scale=float(args.width_scale),
        force=bool(args.force_coverage),
    )
    acceptance_grid, profit_grid = _load_or_build_response_grid(
        output_dir=args.output_dir,
        frame=frame,
        row_indices=row_indices,
        acceptance_artifact=acceptance_artifact,
        loss_artifact=loss_artifact,
    )
    evaluator = CoveragePolicyEvaluator(
        objective=objective,
        frame=frame,
        widths=widths,
        profit_grid=profit_grid,
        acceptance_grid=acceptance_grid,
        u_grid=U_GRID,
    )
    theta0 = np.asarray(baseline["theta"], dtype=float)
    reproduced = objective.policy_value(theta0, frame)
    if not np.allclose(reproduced, baseline["actions"], rtol=0.0, atol=1e-12):
        raise ValueError("The saved baseline theta does not reproduce its sample actions.")

    initial_metrics = evaluator.evaluate(theta0).copy()
    constraint_target = float(baseline["acceptance_floor"] + args.acceptance_buffer)
    trace: list[dict[str, float | int]] = []

    def callback(theta: np.ndarray, state: Any | None = None) -> bool:
        metrics = evaluator.evaluate(theta)
        trace.append(
            {
                "iteration": len(trace),
                "coverage_adjusted_cost": float(metrics["objective"]),
                "raw_mean_profit": float(metrics["raw_profit"]),
                "mean_coverage_width": float(metrics["mean_width"]),
                "mean_acceptance": float(metrics["mean_acceptance"]),
                "mean_u": float(np.mean(metrics["actions"])),
                "optimality": float(getattr(state, "optimality", np.nan)),
                "constraint_violation": float(
                    getattr(state, "constr_violation", np.nan)
                ),
            }
        )
        print(
            f"iteration={len(trace):03d} adjusted_cost={metrics['objective']:.6f} "
            f"profit={metrics['raw_profit']:.6f} width={metrics['mean_width']:.6f} "
            f"acceptance={metrics['mean_acceptance']:.6f} "
            f"mean_u={np.mean(metrics['actions']):.6f}",
            flush=True,
        )
        return False

    constraint = NonlinearConstraint(
        evaluator.mean_acceptance,
        lb=np.asarray([constraint_target], dtype=float),
        ub=np.asarray([np.inf], dtype=float),
        jac=evaluator.mean_acceptance_jac,
    )
    print("Starting coverage-aware trust-constr optimization...", flush=True)
    started = time.perf_counter()
    result = minimize(
        evaluator.fun,
        theta0,
        method="trust-constr",
        jac=evaluator.jac,
        constraints=[constraint],
        callback=callback,
        options={
            "maxiter": int(args.maxiter),
            "gtol": 1e-6,
            "initial_constr_penalty": 1.0,
        },
    )
    runtime = time.perf_counter() - started
    final_metrics = evaluator.evaluate(result.x).copy()
    baseline_direct_acceptance = float(
        np.mean(objective._acceptance_proba(frame, initial_metrics["actions"]))
    )
    coverage_direct_acceptance = float(
        np.mean(objective._acceptance_proba(frame, final_metrics["actions"]))
    )

    np.savez_compressed(
        args.output_dir / "coverage_aware_policy_20k.npz",
        row_indices=row_indices,
        baseline_theta=theta0,
        coverage_theta=np.asarray(result.x, dtype=float),
        baseline_actions=np.asarray(initial_metrics["actions"], dtype=float),
        coverage_actions=np.asarray(final_metrics["actions"], dtype=float),
        baseline_acceptance=np.asarray(initial_metrics["acceptance"], dtype=float),
        coverage_acceptance=np.asarray(final_metrics["acceptance"], dtype=float),
        baseline_width=np.asarray(initial_metrics["coverage_width"], dtype=float),
        coverage_width=np.asarray(final_metrics["coverage_width"], dtype=float),
        acceptance_floor=float(baseline["acceptance_floor"]),
    )
    pd.DataFrame(
        {
            "csv_row_index": row_indices,
            "baseline_u": initial_metrics["actions"],
            "coverage_aware_u": final_metrics["actions"],
            "delta_u": final_metrics["actions"] - initial_metrics["actions"],
            "baseline_coverage_width": initial_metrics["coverage_width"],
            "coverage_aware_width": final_metrics["coverage_width"],
        }
    ).to_csv(args.output_dir / "customer_policy_comparison.csv", index=False)
    pd.DataFrame(trace).to_csv(args.output_dir / "optimization_trace.csv", index=False)
    _save_histograms(
        np.asarray(initial_metrics["actions"], dtype=float),
        np.asarray(final_metrics["actions"], dtype=float),
        args.output_dir,
    )

    metadata = {
        "n_customers": int(len(row_indices)),
        "sample_source": str(args.diagnostics.resolve()),
        "sample_seed": 20260831,
        "baseline_policy_source": str(args.baseline_policy.resolve()),
        "coverage_definition": {
            "normalization": "per-customer maximum local support",
            "width_scale": float(args.width_scale),
            "n_neighbors": int(args.n_neighbors),
            "action_bandwidth": float(ACTION_BANDWIDTH),
            "interpolation": (
                "piecewise linear customer profit, acceptance, and coverage width "
                "on u=0.000,...,0.160"
            ),
        },
        "acceptance_floor": float(baseline["acceptance_floor"]),
        "interpolated_acceptance_target": constraint_target,
        "baseline": _summary(initial_metrics),
        "coverage_aware": _summary(final_metrics),
        "optimizer": {
            "method": "trust-constr",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", len(trace))),
            "function_evaluations": int(getattr(result, "nfev", -1)),
            "gradient_evaluations": int(getattr(result, "njev", -1)),
            "optimality": float(getattr(result, "optimality", np.nan)),
            "constraint_violation": max(
                0.0,
                constraint_target
                - float(final_metrics["mean_acceptance"]),
            ),
            "runtime_sec": float(runtime),
        },
    }
    metadata["baseline"]["direct_xgboost_mean_acceptance"] = baseline_direct_acceptance
    metadata["coverage_aware"][
        "direct_xgboost_mean_acceptance"
    ] = coverage_direct_acceptance
    metadata["optimizer"]["direct_xgboost_constraint_violation"] = max(
        0.0,
        float(baseline["acceptance_floor"]) - coverage_direct_acceptance,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
