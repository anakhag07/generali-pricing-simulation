#!/usr/bin/env python3
"""Fit a 20k spline/XGBoost support-lower-bound policy and replay it broadly.

The policy is a bounded softmax-linear map fitted by the repository first-order
trust-constr optimizer. Its minimized cost is the exact-spline model cost plus
the absolute local-support half-width from the support-cloud figure. The fitted
policy is then evaluated, without refitting, on every eligible customer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import PREMIUM_COL
from data.loader import (
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_observed_u_array,
    load_x_frame,
)
from objective.base import Objective
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize
from reporting.profit_dispersion import (
    exact_spline_acceptance_matrix,
    load_deterministic_sample_rows,
    row_index_sha256,
)
from scripts.plot_glm_spline_profit_dispersion import _spline_weights
from scripts.plot_spline_support_cloud_with_optimized_prices import (
    _plot_overlay,
)


RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_DIAGNOSTICS = (
    RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_SUPPORT_DIR = RESULTS_ROOT / "monotone-spline-xgb-support-cloud"
DEFAULT_REFERENCE_POLICY = (
    RESULTS_ROOT
    / "xgboost-full-dataset-historical-support"
    / "optimized_policy_cropped.npz"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "spline-xgb-support-lower-bound-policy-20k"
ACTION_GRID = np.linspace(-0.1, 0.2, 301)
ACTION_LOW = float(ACTION_GRID[0])
ACTION_HIGH = float(ACTION_GRID[-1])
DEFAULT_INITIAL_U = 0.08
DEFAULT_MAX_STEPS = 200
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLE_SEED = 20260831
HISTOGRAM_EDGES = np.linspace(ACTION_LOW, ACTION_HIGH, 31)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument(
        "--support-csv",
        type=Path,
        default=DEFAULT_SUPPORT_DIR / "monotone_spline_xgb_local_support_cloud.csv",
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=DEFAULT_SUPPORT_DIR / "run_manifest.json",
    )
    parser.add_argument(
        "--acceptance-reference-policy",
        type=Path,
        default=DEFAULT_REFERENCE_POLICY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--initial-u", type=float, default=DEFAULT_INITIAL_U)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _interpolate_rows(
    values: np.ndarray,
    grid: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=float)
    grid_arr = np.asarray(grid, dtype=float)
    actions_arr = np.asarray(actions, dtype=float)
    if values_arr.shape != (actions_arr.size, grid_arr.size):
        raise ValueError("values must contain one action-grid curve per customer.")
    clipped = np.clip(actions_arr, grid_arr[0], grid_arr[-1])
    right = np.searchsorted(grid_arr, clipped, side="right")
    left = np.clip(right - 1, 0, grid_arr.size - 2)
    right = left + 1
    row = np.arange(actions_arr.size)
    span = grid_arr[right] - grid_arr[left]
    fraction = (clipped - grid_arr[left]) / span
    low = values_arr[row, left]
    high = values_arr[row, right]
    return low + fraction * (high - low), (high - low) / span


class SplineSupportLowerBoundObjective(Objective):
    """Piecewise-linear exact-spline cost plus smooth aggregate support width."""

    def __init__(
        self,
        *,
        policy: SoftmaxPolicy,
        cost_grid: np.ndarray,
        acceptance_grid: np.ndarray,
        action_grid: np.ndarray,
        support_width: np.ndarray,
        acceptance_floor: float,
    ) -> None:
        self.policy = policy
        self.cost_grid = np.asarray(cost_grid, dtype=float)
        self.acceptance_grid = np.asarray(acceptance_grid, dtype=float)
        self.action_grid = np.asarray(action_grid, dtype=float)
        self.support_width = np.asarray(support_width, dtype=float)
        self.acceptance_floor = float(acceptance_floor)
        if self.cost_grid.shape != self.acceptance_grid.shape:
            raise ValueError("cost and acceptance grids must have matching shapes.")
        if self.cost_grid.shape[1:] != (self.action_grid.size,):
            raise ValueError("response grids must align to action_grid.")
        if self.support_width.shape != self.action_grid.shape:
            raise ValueError("support_width must align to action_grid.")
        if not np.isfinite(self.cost_grid).all() or not np.isfinite(
            self.acceptance_grid
        ).all():
            raise ValueError("response grids must be finite.")
        if np.any(self.support_width <= 0.0):
            raise ValueError("absolute support width must be positive everywhere.")
        self._support_spline = CubicSpline(
            self.action_grid,
            self.support_width,
            bc_type="natural",
            extrapolate=False,
        )

    def _actions(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        if len(x_batch) != self.cost_grid.shape[0]:
            raise ValueError("This deterministic objective requires the full 20k batch.")
        return np.asarray(self.policy.value(theta, x_batch), dtype=float)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self._actions(theta, x_batch)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions = self._actions(theta, x_batch)
        cost, _ = _interpolate_rows(self.cost_grid, self.action_grid, actions)
        width = np.asarray(self._support_spline(actions), dtype=float)
        return float(np.mean(cost + width))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        actions = self._actions(theta, x_batch)
        _, cost_slope = _interpolate_rows(self.cost_grid, self.action_grid, actions)
        width_slope = np.asarray(self._support_spline(actions, 1), dtype=float)
        return self.policy.weighted_grad(
            theta,
            x_batch,
            cost_slope + width_slope,
        ) / len(actions)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions = self._actions(theta, x_batch)
        acceptance, _ = _interpolate_rows(
            self.acceptance_grid,
            self.action_grid,
            actions,
        )
        return float(np.mean(acceptance))

    def mean_acceptance_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray,
    ) -> np.ndarray:
        actions = self._actions(theta, x_batch)
        _, acceptance_slope = _interpolate_rows(
            self.acceptance_grid,
            self.action_grid,
            actions,
        )
        return self.policy.weighted_grad(
            theta,
            x_batch,
            acceptance_slope,
        ) / len(actions)

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, float]:
        actions = self._actions(theta, x_batch)
        cost, _ = _interpolate_rows(self.cost_grid, self.action_grid, actions)
        width = np.asarray(self._support_spline(actions), dtype=float)
        return {
            "mean_acceptance": self.mean_acceptance(theta, x_batch),
            "projected_loss": float("nan"),
            "projected_revenue": float("nan"),
            "raw_mean_profit": float(-np.mean(cost)),
            "mean_support_width": float(np.mean(width)),
        }

    def summarize(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, Any]:
        actions = self._actions(theta, x_batch)
        cost, _ = _interpolate_rows(self.cost_grid, self.action_grid, actions)
        acceptance, _ = _interpolate_rows(
            self.acceptance_grid,
            self.action_grid,
            actions,
        )
        width = np.asarray(self._support_spline(actions), dtype=float)
        quantiles = np.quantile(actions, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            "mean_action": float(np.mean(actions)),
            "population_std_action": float(np.std(actions, ddof=0)),
            "action_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                    quantiles,
                    strict=True,
                )
            },
            "mean_acceptance": float(np.mean(acceptance)),
            "raw_mean_profit": float(-np.mean(cost)),
            "mean_support_half_width": float(np.mean(width)),
            "support_lower_bound_mean_profit": float(np.mean(-cost - width)),
        }


def _load_support_width(csv_path: Path, manifest_path: Path) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("support", {}).get("baseline_subtracted") is not False:
        raise ValueError("Support cloud must retain absolute baseline risk.")
    frame = pd.read_csv(csv_path).sort_values("u")
    if not np.allclose(frame["u"].to_numpy(dtype=float), ACTION_GRID):
        raise ValueError("Support cloud must cover the full [-0.1, 0.2] grid.")
    width = frame["smoothed_support_half_width"].to_numpy(dtype=float)
    if not np.isfinite(width).all() or np.any(width <= 0.0):
        raise ValueError("Support width must be finite and positive everywhere.")
    return width


def _acceptance_floor(path: Path) -> float:
    with np.load(path, allow_pickle=False) as artifact:
        floor = float(artifact["acceptance_floor"])
    if not 0.0 < floor < 1.0:
        raise ValueError("Acceptance floor must lie strictly between zero and one.")
    return floor


def _artifact_policy_features(acceptance_artifact: Any, frame: pd.DataFrame) -> np.ndarray:
    transformed = acceptance_artifact.preprocessor.transform(
        frame.loc[:, list(acceptance_artifact.x_feature_cols)]
    )
    features = np.asarray(transformed, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Artifact policy features must be a finite matrix.")
    return features


def _load_or_build_response_grid(
    *,
    output_dir: Path,
    frame: pd.DataFrame,
    sample_rows: np.ndarray,
    eligible_rows: np.ndarray,
    acceptance_artifact: Any,
    loss_artifact: Any,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = output_dir / "sample_exact_spline_response_grid.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                np.array_equal(cached["row_indices"], sample_rows)
                and np.allclose(cached["action_grid"], ACTION_GRID)
            ):
                acceptance = cached["acceptance"].astype(float)
                cost = cached["cost"].astype(float)
                if acceptance.shape == cost.shape == (len(sample_rows), len(ACTION_GRID)):
                    print(f"Reusing {cache_path}", flush=True)
                    return acceptance, cost

    print("Building exact monotone-spline response grid for 20,000 customers...", flush=True)
    spline_weights = _spline_weights(eligible_rows)
    acceptance = exact_spline_acceptance_matrix(
        acceptance_artifact,
        frame,
        ACTION_GRID,
        spline_weights,
        n_jobs=n_jobs,
    )
    loss = np.asarray(
        loss_artifact.model.predict(loss_artifact.model_frame(frame)),
        dtype=float,
    )
    premium = frame[PREMIUM_COL].to_numpy(dtype=float)
    cost = acceptance * (
        loss[:, None] - premium[:, None] * (1.0 + ACTION_GRID[None, :])
    )
    np.savez_compressed(
        cache_path,
        row_indices=sample_rows,
        action_grid=ACTION_GRID,
        acceptance=acceptance.astype(np.float32),
        cost=cost.astype(np.float32),
    )
    return acceptance, cost


def _constant_policy_theta(
    policy: SoftmaxPolicy,
    feature_dim: int,
    initial_u: float,
) -> np.ndarray:
    fraction = (float(initial_u) - policy.action_low) / policy.action_span
    if not 0.0 < fraction < 1.0:
        raise ValueError("initial_u must lie strictly inside the policy bounds.")
    theta = np.zeros(policy.theta_dim(feature_dim), dtype=float)
    theta[0] = np.log(fraction / (1.0 - fraction))
    return theta


def _optimized_histogram(actions: np.ndarray) -> pd.DataFrame:
    values = np.clip(np.asarray(actions, dtype=float), ACTION_LOW, ACTION_HIGH)
    counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
    widths = np.diff(HISTOGRAM_EDGES)
    density = counts / (values.size * widths)
    return pd.DataFrame(
        {
            "plot": "optimized",
            "series": "Optimized",
            "bin_left": HISTOGRAM_EDGES[:-1],
            "bin_right": HISTOGRAM_EDGES[1:],
            "count": counts,
            "density": density,
        }
    )


def _apply_policy_to_full_population(
    *,
    theta: np.ndarray,
    policy: SoftmaxPolicy,
    policy_preprocessor: PolicyFeaturePreprocessor,
    acceptance_artifact: Any,
    eligible_rows: np.ndarray,
) -> np.ndarray:
    print(f"Applying saved 20k policy to {len(eligible_rows):,} eligible customers...", flush=True)
    frame = load_x_frame("xgb", row_indices=eligible_rows)
    artifact_features = _artifact_policy_features(acceptance_artifact, frame)
    policy_features = policy_preprocessor.transform(artifact_features)
    return np.clip(
        np.asarray(policy.value(theta, policy_features), dtype=float),
        ACTION_LOW,
        ACTION_HIGH,
    )


def _trace_payload(trace: Any) -> dict[str, Any]:
    return {
        "success": bool(trace.optimizer_success),
        "status": int(trace.optimizer_status),
        "message": str(trace.optimizer_message),
        "steps": max(0, len(trace.steps) - 1),
        "final_gradient_norm": float(trace.theta_grad_norms[-1]),
        "constraint_violation": (
            None
            if trace.constraint_violation is None
            else float(trace.constraint_violation)
        ),
        "optimality": (
            None
            if trace.optimizer_optimality is None
            else float(trace.optimizer_optimality)
        ),
    }


def run_analysis(args: argparse.Namespace) -> list[Path]:
    if int(args.max_steps) <= 0 or int(args.n_jobs) == 0:
        raise ValueError("max_steps must be positive and n_jobs cannot be zero.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_rows = eligible_csv_row_indices("xgb")
    sample_rows = load_deterministic_sample_rows(
        args.diagnostics,
        eligible_rows,
        sample_size=DEFAULT_SAMPLE_SIZE,
        seed=DEFAULT_SAMPLE_SEED,
    )
    frame = load_x_frame("xgb", row_indices=sample_rows)
    acceptance_artifact, loss_artifact = load_model_artifact_pair("xgb", "xgb")
    for artifact in (acceptance_artifact, loss_artifact):
        if hasattr(artifact.model, "set_params"):
            available = artifact.model.get_params(deep=False)
            if "n_jobs" in available:
                artifact.model.set_params(n_jobs=int(args.n_jobs))

    acceptance_grid, cost_grid = _load_or_build_response_grid(
        output_dir=output_dir,
        frame=frame,
        sample_rows=sample_rows,
        eligible_rows=eligible_rows,
        acceptance_artifact=acceptance_artifact,
        loss_artifact=loss_artifact,
        n_jobs=int(args.n_jobs),
    )
    support_width = _load_support_width(args.support_csv, args.support_manifest)
    artifact_features = _artifact_policy_features(acceptance_artifact, frame)
    policy_preprocessor = PolicyFeaturePreprocessor(
        standardize=True,
        sphere=True,
    )
    policy_features = policy_preprocessor.fit_transform(artifact_features)
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
    )
    floor = _acceptance_floor(args.acceptance_reference_policy)
    objective = SplineSupportLowerBoundObjective(
        policy=policy,
        cost_grid=cost_grid,
        acceptance_grid=acceptance_grid,
        action_grid=ACTION_GRID,
        support_width=support_width,
        acceptance_floor=floor,
    )
    theta0 = _constant_policy_theta(
        policy,
        policy_features.shape[1],
        float(args.initial_u),
    )
    initial_summary = objective.summarize(theta0, policy_features)
    if initial_summary["mean_acceptance"] < floor:
        raise ValueError("The requested initial policy is not acceptance-feasible.")

    print("Running repository first-order trust-constr minimization...", flush=True)
    started = time.perf_counter()
    theta, trace = run_first_order_minimize(
        theta0,
        policy_features,
        objective,
        t_steps=int(args.max_steps),
        n_grad_samples=1,
        sigma=0.001,
        algorithm="trust-constr",
        grad_norm_tol=1e-6,
        initial_constr_penalty=1.0,
    )
    runtime = time.perf_counter() - started
    if not trace.optimizer_success:
        raise RuntimeError(
            "Repository first-order optimizer did not converge: "
            f"{trace.optimizer_message}"
        )
    final_summary = objective.summarize(theta, policy_features)
    if final_summary["mean_acceptance"] < floor - 1e-8:
        raise RuntimeError("Converged policy violates the acceptance constraint.")
    sample_actions = np.clip(
        policy.value(theta, policy_features),
        ACTION_LOW,
        ACTION_HIGH,
    )

    state = policy_preprocessor.to_state()
    policy_path = output_dir / "support_lower_bound_policy_20k.npz"
    np.savez_compressed(
        policy_path,
        row_indices=sample_rows,
        theta=theta,
        sample_actions=sample_actions,
        action_bounds=np.asarray([ACTION_LOW, ACTION_HIGH]),
        acceptance_floor=floor,
        preprocessor_mean=state["arrays"]["mean"],
        preprocessor_scale=state["arrays"]["scale"],
        preprocessor_eigenvalues=state["arrays"]["eigenvalues"],
        preprocessor_explained_variance_ratio=state["arrays"][
            "explained_variance_ratio"
        ],
        preprocessor_cumulative_variance_ratio=state["arrays"][
            "cumulative_variance_ratio"
        ],
        preprocessor_transform_matrix=state["arrays"]["transform_matrix"],
    )

    full_actions = _apply_policy_to_full_population(
        theta=theta,
        policy=policy,
        policy_preprocessor=policy_preprocessor,
        acceptance_artifact=acceptance_artifact,
        eligible_rows=eligible_rows,
    )
    full_actions_path = output_dir / "support_lower_bound_policy_full_actions.npz"
    np.savez_compressed(
        full_actions_path,
        row_indices=eligible_rows,
        actions=full_actions,
        theta=theta,
    )
    histogram = _optimized_histogram(full_actions)
    histogram_path = output_dir / "support_lower_bound_policy_histogram.csv"
    histogram.to_csv(histogram_path, index=False)

    support_frame = pd.read_csv(args.support_csv)
    overlay_path = output_dir / "mean_profit_support_cloud_with_lower_bound_policy.pdf"
    _plot_overlay(support_frame, histogram, overlay_path)

    full_quantiles = np.quantile(
        full_actions,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    summary = {
        "analysis": "20k-spline-xgb-support-lower-bound-softmax-policy",
        "sample": {
            "n_customers": int(sample_rows.size),
            "seed": DEFAULT_SAMPLE_SEED,
            "selection": "sorted choice without replacement from eligible rows",
            "row_indices_sha256": row_index_sha256(sample_rows),
        },
        "full_application": {
            "n_customers": int(eligible_rows.size),
            "row_indices_sha256": row_index_sha256(eligible_rows),
            "mean_action": float(np.mean(full_actions)),
            "population_std_action": float(np.std(full_actions, ddof=0)),
            "action_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                    full_quantiles,
                    strict=True,
                )
            },
        },
        "models": {
            "acceptance": "exact monotone-spline XGBoost",
            "loss": "XGBoost financial risk",
            "spline_boundary_rules": "constant left; clipped-linear right",
            "spline_construction": (
                "17 raw-XGBoost anchors, weighted smoothing spline, isotonic "
                "projection, then PCHIP"
            ),
            "acceptance_artifact": _file_record(
                Path(acceptance_artifact.artifact_path)
            ),
            "loss_artifact": _file_record(Path(loss_artifact.artifact_path)),
        },
        "policy": {
            "class": "objective.policy.SoftmaxPolicy",
            "feature_map": "IdentityFeatureMap over fitted standardized/sphered XGBoost features",
            "formula": "u_i(theta) = -0.1 + 0.3 * sigmoid(theta_0 + theta_x^T z_i)",
            "theta": np.asarray(theta, dtype=float).tolist(),
            "preprocessor": state["metadata"],
        },
        "objective": {
            "direction": "minimize",
            "formula": "mean_i[ModelBasedObjective_i(u_i(theta)) + W(u_i(theta))]",
            "maximization_equivalent": "mean_i[profit_i(u_i(theta)) - W(u_i(theta))]",
            "support_width_source": str(args.support_csv.resolve()),
            "support_baseline_subtracted": False,
            "acceptance_floor": floor,
        },
        "initial_policy_on_sample": initial_summary,
        "optimized_policy_on_sample": final_summary,
        "optimizer": {
            "entry_point": "optimization.solvers.run_first_order_minimize",
            "step_rule": "trust-constr",
            "gradient": "first_order",
            "max_steps": int(args.max_steps),
            "runtime_seconds": runtime,
            **_trace_payload(trace),
        },
        "inputs": {
            "diagnostics": _file_record(args.diagnostics),
            "support_csv": _file_record(args.support_csv),
            "support_manifest": _file_record(args.support_manifest),
            "acceptance_reference_policy": _file_record(
                args.acceptance_reference_policy
            ),
        },
        "outputs": {
            path.name: _sha256_file(path)
            for path in (
                output_dir / "sample_exact_spline_response_grid.npz",
                policy_path,
                full_actions_path,
                histogram_path,
                overlay_path,
            )
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [overlay_path, policy_path, full_actions_path, histogram_path, summary_path]
    for output in outputs:
        print(output, flush=True)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
