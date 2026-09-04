#!/usr/bin/env python3
"""Compare full-cohort softmax pricing policies with and without support risk.

Both policies use the repository ``SoftmaxPolicy`` with identity customer
features,

    u_i(theta) = 0.16 * sigmoid(theta_0 + theta_x^T x_i),

and are fitted by the repository action-space finite-difference optimizer. The
support-adjusted objective adds the same marginal-support half-width plotted in
``01_full_population_profit_with_support_weighted_band.pdf`` to each customer's
XGBoost minimization objective at that customer's policy action.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from data.dataset_metadata import ACCEPTANCE_STATE_COLS, LOSS_FEATURE_COLS, PREMIUM_COL
from data.loader import eligible_csv_row_indices, load_model_artifacts, load_x_frame
from experiments.config import make_model_based_objective
from objective.modifications.bias import ActionBias, BiasedObjective
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from optimization.solvers import run_finite_difference_minimize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPORT_CSV = (
    REPOSITORY_ROOT
    / "results"
    / "customer-coverage-envelope-slides"
    / "01_full_population_profit_with_support_weighted_band.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "results" / "full-population-support-softmax-policy"
)
DEFAULT_CONSTRAINED_REFERENCE = (
    REPOSITORY_ROOT
    / "results"
    / "xgboost-full-dataset-historical-support"
    / "optimized_policy_cropped.npz"
)
DEFAULT_DIAGNOSTIC_SAMPLE = (
    REPOSITORY_ROOT
    / "results"
    / "customer-coverage-envelope-slides"
    / "coverage_diagnostics.npz"
)
ACTION_LOW = 0.0
ACTION_HIGH = 0.16
OPTIMIZER_SIGMA_U = 0.001
OPTIMIZER_T_STEPS = 200
OPTIMIZER_GRAD_NORM_TOL = 1e-6
OPTIMIZER_FTOL = 1e-10


@dataclass(frozen=True)
class _SupportBandActionBias(ActionBias):
    """Natural-cubic interpolation of the displayed support half-width."""

    u_grid: np.ndarray
    half_width: np.ndarray
    lambda_bias: float = 1.0
    _spline: CubicSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        u = np.asarray(self.u_grid, dtype=float)
        width = np.asarray(self.half_width, dtype=float)
        if u.ndim != 1 or width.shape != u.shape or len(u) < 2:
            raise ValueError("u_grid and half_width must be matching 1D arrays.")
        if not np.all(np.diff(u) > 0.0):
            raise ValueError("u_grid must be strictly increasing.")
        if not np.isfinite(width).all() or np.any(width < 0.0):
            raise ValueError("half_width must contain finite nonnegative values.")
        object.__setattr__(self, "u_grid", u)
        object.__setattr__(self, "half_width", width)
        object.__setattr__(
            self,
            "_spline",
            CubicSpline(u, width, bc_type="natural", extrapolate=False),
        )

    def _bounded(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=float), self.u_grid[0], self.u_grid[-1])

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        return np.asarray(self._spline(self._bounded(u)), dtype=float)

    def grad_u(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        return np.asarray(self._spline(self._bounded(u), 1), dtype=float)


def _load_support_bias(path: Path) -> _SupportBandActionBias:
    frame = pd.read_csv(path)
    required = {"u", "illustrative_band_half_width"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Support CSV is missing columns: {sorted(missing)}")
    return _SupportBandActionBias(
        frame["u"].to_numpy(dtype=float),
        frame["illustrative_band_half_width"].to_numpy(dtype=float),
    )


def _load_replayed_profit_policy(
    path: Path,
    *,
    theta_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an exact saved repository-optimizer policy and its provenance."""

    with np.load(path) as artifact:
        if "profit_theta" not in artifact:
            raise ValueError(f"{path} does not contain 'profit_theta'.")
        theta = np.asarray(artifact["profit_theta"], dtype=float)
    if theta.shape != (theta_dim,):
        raise ValueError(
            f"Saved profit_theta has shape {theta.shape}; expected {(theta_dim,)}."
        )
    if not np.isfinite(theta).all():
        raise ValueError("Saved profit_theta must contain only finite values.")

    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        raise ValueError(
            "Exact optimizer provenance is required, but the companion summary "
            f"does not exist: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "profit_policy" not in payload:
        raise ValueError(f"{summary_path} does not contain 'profit_policy'.")
    return theta, dict(payload["profit_policy"])


def _load_constrained_reference_policy(
    path: Path,
    *,
    expected_rows: np.ndarray,
    theta_dim: int,
) -> dict[str, Any]:
    """Load the broad saved policy as a reference, without claiming convergence."""

    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "row_indices",
            "actions",
            "theta",
            "u_bounds",
            "acceptance_floor",
            "optimizer_success",
            "optimizer_message",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"Reference policy is missing keys: {sorted(missing)}")
        rows = np.asarray(artifact["row_indices"], dtype=int)
        actions = np.asarray(artifact["actions"], dtype=float)
        theta = np.asarray(artifact["theta"], dtype=float)
        bounds = np.asarray(artifact["u_bounds"], dtype=float)
        acceptance_floor = float(artifact["acceptance_floor"])
        optimizer_success = bool(artifact["optimizer_success"])
        optimizer_message = str(artifact["optimizer_message"])
    expected = np.asarray(expected_rows, dtype=int)
    if actions.shape != rows.shape or not np.isfinite(actions).all():
        raise ValueError("Reference actions must be finite and aligned to row_indices.")
    positions = np.searchsorted(rows, expected)
    if np.any(positions >= len(rows)) or not np.array_equal(rows[positions], expected):
        raise ValueError("Reference policy does not contain every requested sample row.")
    if theta.shape != (theta_dim,) or not np.isfinite(theta).all():
        raise ValueError(f"Reference theta must have shape {(theta_dim,)} and be finite.")
    if not np.allclose(bounds, [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Reference policy bounds must be [0.0, 0.16].")
    if not 0.0 < acceptance_floor < 1.0:
        raise ValueError("Reference acceptance_floor must lie in (0, 1).")
    return {
        "row_indices": expected,
        "actions": actions[positions],
        "theta": theta,
        "acceptance_floor": acceptance_floor,
        "optimizer_success": optimizer_success,
        "optimizer_message": optimizer_message,
    }


def _run_policy_optimizer(
    objective: Any,
    frame: pd.DataFrame,
    theta0: np.ndarray,
    *,
    t_steps: int,
    algorithm: str = "l-bfgs-b",
    initial_constr_penalty: float | None = None,
    require_success: bool = True,
) -> tuple[np.ndarray, Any, float]:
    started = time.perf_counter()
    theta, trace = run_finite_difference_minimize(
        theta0,
        frame,
        objective,
        t_steps=t_steps,
        n_grad_samples=1,
        sigma=OPTIMIZER_SIGMA_U,
        perturbation_space="u",
        algorithm=algorithm,
        grad_norm_tol=OPTIMIZER_GRAD_NORM_TOL,
        ftol=OPTIMIZER_FTOL if algorithm != "trust-constr" else None,
        initial_constr_penalty=initial_constr_penalty,
    )
    runtime = time.perf_counter() - started
    if require_success and not trace.optimizer_success:
        raise RuntimeError(
            "Repository finite-difference optimization failed: "
            f"{trace.optimizer_message}"
        )
    return np.asarray(theta, dtype=float), trace, runtime


def _policy_summary(
    *,
    theta: np.ndarray,
    trace: Any | None,
    runtime: float | None,
    base_objective: Any,
    adjusted_objective: Any,
    frame: pd.DataFrame,
    support_bias: _SupportBandActionBias,
) -> tuple[dict[str, Any], np.ndarray]:
    actions = np.asarray(base_objective.policy_value(theta, frame), dtype=float)
    raw_cost = float(base_objective.value(theta, frame))
    adjusted_cost = float(adjusted_objective.value(theta, frame))
    support_width = np.asarray(support_bias.values(frame, actions), dtype=float)
    quantile_levels = np.asarray([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantile_values = np.quantile(actions, quantile_levels)
    summary = {
            "theta": theta.tolist(),
            "mean_action": float(np.mean(actions)),
            "standard_deviation_action": float(np.std(actions)),
            "action_quantiles": {
                f"{level:.2f}": float(value)
                for level, value in zip(quantile_levels, quantile_values, strict=True)
            },
            "raw_mean_profit": -raw_cost,
            "mean_support_half_width": float(np.mean(support_width)),
            "support_adjusted_mean_profit": -adjusted_cost,
    }
    mean_acceptance_fn = getattr(base_objective, "mean_acceptance", None)
    if callable(mean_acceptance_fn):
        summary["mean_acceptance"] = float(mean_acceptance_fn(theta, frame))
    if trace is not None:
        summary.update(
            {
                "optimizer_success": bool(trace.optimizer_success),
                "optimizer_status": int(trace.optimizer_status),
                "optimizer_message": str(trace.optimizer_message),
                "optimizer_steps": max(0, len(trace.steps) - 1),
                "runtime_seconds": float(runtime),
            }
        )
    return summary, actions


def _plot_action_histograms(
    profit_actions: np.ndarray,
    support_actions: np.ndarray,
    output_dir: Path,
    *,
    left_title: str = "Mean-Profit Policy",
    right_title: str = "Support-Lower-Bound Policy",
    filename: str = "01_softmax_policy_actions_profit_vs_support_lower.pdf",
) -> None:
    bins = np.linspace(ACTION_LOW, ACTION_HIGH, 33)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 5.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    weights = np.full(len(profit_actions), 100.0 / len(profit_actions))
    axes[0].hist(profit_actions, bins=bins, weights=weights)
    axes[0].set_title(left_title, fontsize=14)
    axes[1].hist(support_actions, bins=bins, weights=weights)
    axes[1].set_title(right_title, fontsize=14)
    for ax in axes:
        ax.set_xlabel("Optimizer Price Change", fontsize=12)
        ax.set_ylabel("Customers (%)", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    fig.suptitle(
        "Customer-Dependent Softmax Policy Actions",
        fontsize=16,
    )
    fig.savefig(
        output_dir / filename,
        format="pdf",
    )
    plt.close(fig)


def _plot_single_action_histogram(
    actions: np.ndarray,
    output_dir: Path,
    *,
    title: str,
    filename: str,
) -> None:
    bins = np.linspace(ACTION_LOW, ACTION_HIGH, 33)
    weights = np.full(len(actions), 100.0 / len(actions))
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.hist(actions, bins=bins, weights=weights)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Optimizer Price Change", fontsize=12)
    ax.set_ylabel("Customers (%)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    fig.savefig(
        output_dir / filename,
        format="pdf",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=OPTIMIZER_T_STEPS)
    parser.add_argument(
        "--row-indices-npz",
        type=Path,
        help="Use the saved 'row_indices' cohort from this NPZ instead of all rows.",
    )
    parser.add_argument(
        "--replay-profit-npz",
        type=Path,
        help=(
            "Replay the exact saved profit_theta from a prior repository-optimizer "
            "run, then warm-start the lower-bound optimizer from that policy."
        ),
    )
    parser.add_argument(
        "--constrained-reference-npz",
        type=Path,
        help=(
            "Keep the saved broad constrained-policy actions as the left reference "
            "and optimize the support-lower-bound policy from its theta with the "
            "same cohort acceptance floor."
        ),
    )
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--initial-constr-penalty", type=float, default=1.0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Write an explicitly labeled intermediate policy when the repository "
            "optimizer reaches its step cap without convergence."
        ),
    )
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--support-only", action="store_true")
    arm_group.add_argument("--profit-only", action="store_true")
    args = parser.parse_args()
    if args.replay_profit_npz is not None and (args.support_only or args.profit_only):
        parser.error("--replay-profit-npz cannot be combined with an arm-only flag.")
    if args.constrained_reference_npz is not None and (
        args.replay_profit_npz is not None or args.support_only or args.profit_only
    ):
        parser.error(
            "--constrained-reference-npz cannot be combined with replay or arm-only flags."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.row_indices_npz is None:
        row_indices = eligible_csv_row_indices("xgb")
    else:
        with np.load(args.row_indices_npz, allow_pickle=False) as sample_artifact:
            if "row_indices" not in sample_artifact:
                raise ValueError(
                    f"{args.row_indices_npz} does not contain 'row_indices'."
                )
            row_indices = np.asarray(sample_artifact["row_indices"], dtype=int)
        if row_indices.ndim != 1 or len(row_indices) == 0:
            raise ValueError("Saved sample row_indices must be a nonempty 1D array.")
    frame = load_x_frame("xgb", row_indices=row_indices)
    acceptance_artifact, loss_artifact = load_model_artifacts("xgb")
    acceptance_artifact.model.set_params(
        n_jobs=int(args.n_jobs), device=args.xgb_device
    )
    loss_artifact.model.set_params(n_jobs=int(args.n_jobs), device=args.xgb_device)

    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
    )
    reference_policy: dict[str, Any] | None = None
    reference_acceptance_floor: float | None = None
    if args.constrained_reference_npz is not None:
        with np.load(args.constrained_reference_npz, allow_pickle=False) as artifact:
            reference_acceptance_floor = float(artifact["acceptance_floor"])
    base_objective = make_model_based_objective(
        policy=policy,
        acceptance_model=acceptance_artifact,
        loss_model=loss_artifact,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        u_bounds=(ACTION_LOW, ACTION_HIGH),
        acceptance_floor=reference_acceptance_floor,
    )
    support_bias = _load_support_bias(args.support_csv)
    adjusted_objective = BiasedObjective(
        base_objective=base_objective,
        bias=support_bias,
    )
    theta0 = np.zeros(base_objective.policy_theta_dim(), dtype=float)
    if args.constrained_reference_npz is not None:
        reference_policy = _load_constrained_reference_policy(
            args.constrained_reference_npz,
            expected_rows=row_indices,
            theta_dim=len(theta0),
        )

    replayed_profit_summary: dict[str, Any] | None = None
    profit_result: tuple[np.ndarray, Any | None, float | None] | None = None
    if reference_policy is not None:
        profit_result = None
    elif args.replay_profit_npz is not None:
        replayed_theta, replayed_profit_summary = _load_replayed_profit_policy(
            args.replay_profit_npz,
            theta_dim=len(theta0),
        )
        profit_result = (replayed_theta, None, None)
    elif not args.support_only:
        profit_result = _run_policy_optimizer(
            base_objective,
            frame,
            theta0,
            t_steps=int(args.max_steps),
        )
    support_result: tuple[np.ndarray, Any, float] | None = None
    if not args.profit_only:
        if reference_policy is not None:
            support_theta0 = np.asarray(reference_policy["theta"], dtype=float)
        else:
            support_theta0 = profit_result[0] if profit_result is not None else theta0
        support_result = _run_policy_optimizer(
            adjusted_objective,
            frame,
            support_theta0,
            t_steps=int(args.max_steps),
            algorithm="trust-constr" if reference_policy is not None else "l-bfgs-b",
            initial_constr_penalty=(
                float(args.initial_constr_penalty)
                if reference_policy is not None
                else None
            ),
            require_success=not args.allow_incomplete,
        )

    if args.profit_only:
        if profit_result is None:
            raise RuntimeError("Profit-policy optimizer result is unavailable.")
        profit_theta, profit_trace, profit_runtime = profit_result
        profit_summary, profit_actions = _policy_summary(
            theta=profit_theta,
            trace=profit_trace,
            runtime=profit_runtime,
            base_objective=base_objective,
            adjusted_objective=adjusted_objective,
            frame=frame,
            support_bias=support_bias,
        )
        np.savez_compressed(
            args.output_dir / "profit_softmax_policy_actions.npz",
            row_indices=row_indices,
            profit_theta=profit_theta,
            profit_actions=profit_actions,
        )
        _plot_single_action_histogram(
            profit_actions,
            args.output_dir,
            title="Mean-Profit Softmax Policy Actions",
            filename="profit_softmax_policy_actions.pdf",
        )
        profit_only_payload = {
            "population_n_customers": int(len(row_indices)),
            "policy_class": "objective.policy.SoftmaxPolicy",
            "feature_map": "objective.policy.IdentityFeatureMap",
            "policy_formula": (
                "u_i(theta) = 0.16 * sigmoid(theta_0 + theta_x^T x_i)"
            ),
            "policy_theta_dim": int(len(theta0)),
            "initial_theta": theta0.tolist(),
            "initial_action_for_every_customer": 0.08,
            "objective": "mean_i[J_i(theta)]",
            "optimizer": {
                "entry_point": (
                    "optimization.solvers.run_finite_difference_minimize"
                ),
                "direction": "minimize",
                "step_rule": "l-bfgs-b",
                "gradient_estimator": "finite_difference",
                "perturbation_space": "u",
                "sigma_u": OPTIMIZER_SIGMA_U,
                "max_steps": int(args.max_steps),
                "grad_norm_tol": OPTIMIZER_GRAD_NORM_TOL,
                "ftol": OPTIMIZER_FTOL,
            },
            "profit_policy": profit_summary,
        }
        (args.output_dir / "profit_summary.json").write_text(
            json.dumps(profit_only_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(profit_only_payload, indent=2), flush=True)
        return

    if support_result is None:
        raise RuntimeError("Support-policy optimizer result is unavailable.")
    support_theta, support_trace, support_runtime = support_result
    support_summary, support_actions = _policy_summary(
        theta=support_theta,
        trace=support_trace,
        runtime=support_runtime,
        base_objective=base_objective,
        adjusted_objective=adjusted_objective,
        frame=frame,
        support_bias=support_bias,
    )

    if reference_policy is not None:
        intermediate = not bool(support_trace.optimizer_success)
        reference_theta = np.asarray(reference_policy["theta"], dtype=float)
        reference_summary, evaluated_reference_actions = _policy_summary(
            theta=reference_theta,
            trace=None,
            runtime=None,
            base_objective=base_objective,
            adjusted_objective=adjusted_objective,
            frame=frame,
            support_bias=support_bias,
        )
        if not np.allclose(evaluated_reference_actions, reference_policy["actions"]):
            raise ValueError("Reference theta no longer reproduces its saved actions.")
        reference_summary.update(
            {
                "artifact_path": str(args.constrained_reference_npz.resolve()),
                "original_optimizer_success": bool(
                    reference_policy["optimizer_success"]
                ),
                "original_optimizer_message": str(
                    reference_policy["optimizer_message"]
                ),
                "role": "reference artifact; not reported as a converged optimum",
            }
        )
        np.savez_compressed(
            args.output_dir / "constrained_reference_vs_support_lower_actions.npz",
            row_indices=row_indices,
            reference_theta=reference_theta,
            support_theta=support_theta,
            reference_actions=evaluated_reference_actions,
            support_actions=support_actions,
        )
        histogram_edges = np.linspace(ACTION_LOW, ACTION_HIGH, 33)
        reference_counts, _ = np.histogram(
            evaluated_reference_actions, bins=histogram_edges
        )
        support_counts, _ = np.histogram(support_actions, bins=histogram_edges)
        pd.DataFrame(
            {
                "bin_left": histogram_edges[:-1],
                "bin_right": histogram_edges[1:],
                "reference_policy_count": reference_counts,
                "support_policy_count": support_counts,
            }
        ).to_csv(
            args.output_dir / "constrained_reference_vs_support_lower_histograms.csv",
            index=False,
        )
        _plot_action_histograms(
            evaluated_reference_actions,
            support_actions,
            args.output_dir,
            left_title="Previous Constrained Policy Artifact",
            right_title=(
                "Support-Lower-Bound Intermediate"
                if intermediate
                else "Support-Lower-Bound Constrained Policy"
            ),
            filename=(
                "01_constrained_reference_vs_support_lower_intermediate.pdf"
                if intermediate
                else "01_constrained_reference_vs_support_lower_actions.pdf"
            ),
        )
        _plot_single_action_histogram(
            support_actions,
            args.output_dir,
            title=(
                "Support-Lower-Bound Intermediate Actions"
                if intermediate
                else "Support-Lower-Bound Optimizer Actions"
            ),
            filename=(
                "02_support_lower_bound_intermediate_actions.pdf"
                if intermediate
                else "02_support_lower_bound_optimizer_actions.pdf"
            ),
        )
        constrained_payload = {
            "population_n_customers": int(len(row_indices)),
            "row_indices_source": (
                str(args.row_indices_npz.resolve())
                if args.row_indices_npz is not None
                else "all eligible XGBoost rows"
            ),
            "policy_class": "objective.policy.SoftmaxPolicy",
            "feature_map": "objective.policy.IdentityFeatureMap",
            "policy_formula": (
                "u_i(theta) = 0.16 * sigmoid(theta_0 + theta_x^T x_i)"
            ),
            "policy_theta_dim": int(len(theta0)),
            "acceptance_constraint": (
                "mean_i[acceptance_i(u_i(theta))] >= acceptance_floor"
            ),
            "acceptance_floor": float(reference_policy["acceptance_floor"]),
            "support_band_source": str(args.support_csv.resolve()),
            "support_objective": "mean_i[J_i(theta) + H(u_i(theta))]",
            "optimizer": {
                "entry_point": "optimization.solvers.run_finite_difference_minimize",
                "direction": "minimize",
                "step_rule": "trust-constr",
                "gradient_estimator": "finite_difference",
                "perturbation_space": "u",
                "sigma_u": OPTIMIZER_SIGMA_U,
                "max_steps": int(args.max_steps),
                "grad_norm_tol": OPTIMIZER_GRAD_NORM_TOL,
                "initial_constr_penalty": float(args.initial_constr_penalty),
                "xgb_device": args.xgb_device,
            },
            "support_initialization": "saved constrained reference theta",
            "result_role": (
                "nonconverged intermediate iterate; not an optimum"
                if intermediate
                else "converged constrained optimizer solution"
            ),
            "reference_policy": reference_summary,
            "support_lower_bound_policy": support_summary,
        }
        (args.output_dir / "constrained_summary.json").write_text(
            json.dumps(constrained_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(constrained_payload, indent=2), flush=True)
        return

    if args.support_only:
        np.savez_compressed(
            args.output_dir / "support_lower_softmax_policy_actions.npz",
            row_indices=row_indices,
            support_theta=support_theta,
            support_actions=support_actions,
        )
        _plot_single_action_histogram(
            support_actions,
            args.output_dir,
            title="Support-Lower-Bound Softmax Policy Actions",
            filename="support_lower_softmax_policy_actions.pdf",
        )
        support_only_payload = {
            "population_n_customers": int(len(row_indices)),
            "policy_class": "objective.policy.SoftmaxPolicy",
            "feature_map": "objective.policy.IdentityFeatureMap",
            "policy_formula": (
                "u_i(theta) = 0.16 * sigmoid(theta_0 + theta_x^T x_i)"
            ),
            "policy_theta_dim": int(len(theta0)),
            "initial_theta": theta0.tolist(),
            "initial_action_for_every_customer": 0.08,
            "support_band_source": str(args.support_csv.resolve()),
            "support_objective": "mean_i[J_i(theta) + H(u_i(theta))]",
            "optimizer": {
                "entry_point": (
                    "optimization.solvers.run_finite_difference_minimize"
                ),
                "direction": "minimize",
                "step_rule": "l-bfgs-b",
                "gradient_estimator": "finite_difference",
                "perturbation_space": "u",
                "sigma_u": OPTIMIZER_SIGMA_U,
                "max_steps": int(args.max_steps),
                "grad_norm_tol": OPTIMIZER_GRAD_NORM_TOL,
                "ftol": OPTIMIZER_FTOL,
            },
            "support_lower_bound_policy": support_summary,
        }
        (args.output_dir / "support_lower_summary.json").write_text(
            json.dumps(support_only_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(support_only_payload, indent=2), flush=True)
        return

    if profit_result is None:
        raise RuntimeError("Profit-policy optimizer result is unavailable.")
    profit_theta, profit_trace, profit_runtime = profit_result
    profit_summary, profit_actions = _policy_summary(
        theta=profit_theta,
        trace=profit_trace,
        runtime=profit_runtime,
        base_objective=base_objective,
        adjusted_objective=adjusted_objective,
        frame=frame,
        support_bias=support_bias,
    )
    if replayed_profit_summary is not None:
        profit_summary = {**replayed_profit_summary, **profit_summary}
        profit_summary["replayed_from"] = str(args.replay_profit_npz.resolve())

    np.savez_compressed(
        args.output_dir / "softmax_policy_actions.npz",
        row_indices=row_indices,
        profit_theta=profit_theta,
        support_theta=support_theta,
        profit_actions=profit_actions,
        support_actions=support_actions,
    )
    histogram_edges = np.linspace(ACTION_LOW, ACTION_HIGH, 33)
    profit_counts, _ = np.histogram(profit_actions, bins=histogram_edges)
    support_counts, _ = np.histogram(support_actions, bins=histogram_edges)
    pd.DataFrame(
        {
            "bin_left": histogram_edges[:-1],
            "bin_right": histogram_edges[1:],
            "profit_policy_count": profit_counts,
            "support_policy_count": support_counts,
        }
    ).to_csv(args.output_dir / "softmax_policy_action_histograms.csv", index=False)
    _plot_action_histograms(profit_actions, support_actions, args.output_dir)

    payload = {
        "population_n_customers": int(len(row_indices)),
        "policy_class": "objective.policy.SoftmaxPolicy",
        "feature_map": "objective.policy.IdentityFeatureMap",
        "policy_formula": "u_i(theta) = 0.16 * sigmoid(theta_0 + theta_x^T x_i)",
        "policy_theta_dim": int(len(theta0)),
        "profit_initial_theta": theta0.tolist(),
        "profit_initial_action_for_every_customer": 0.08,
        "profit_policy_replayed_from": (
            str(args.replay_profit_npz.resolve())
            if args.replay_profit_npz is not None
            else None
        ),
        "support_initialization": "profit_policy_theta",
        "support_initial_theta": profit_theta.tolist(),
        "support_band_source": str(args.support_csv.resolve()),
        "support_objective": "mean_i[J_i(theta) + H(u_i(theta))]",
        "optimizer": {
            "entry_point": "optimization.solvers.run_finite_difference_minimize",
            "direction": "minimize",
            "step_rule": "l-bfgs-b",
            "gradient_estimator": "finite_difference",
            "perturbation_space": "u",
            "sigma_u": OPTIMIZER_SIGMA_U,
            "max_steps": int(args.max_steps),
            "grad_norm_tol": OPTIMIZER_GRAD_NORM_TOL,
            "ftol": OPTIMIZER_FTOL,
        },
        "profit_policy": profit_summary,
        "support_lower_bound_policy": support_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
