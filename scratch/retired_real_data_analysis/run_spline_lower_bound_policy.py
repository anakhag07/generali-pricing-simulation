#!/usr/bin/env python3
"""Fit a 20k support-lower-bound policy and replay it broadly.

The policy is a bounded softmax-linear map fitted by the repository first-order
trust-constr optimizer. Its minimized cost is either the exact-spline/XGBoost
or GLM model cost plus the absolute local-support half-width from the supplied
support-cloud figure. The fitted policy is then evaluated, without refitting,
on every eligible customer.
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
import numpy.core.numeric as _numpy_core_numeric
import pandas as pd

# The bundled GLM pickles were written by NumPy 2 but this runtime uses NumPy 1.26.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.numeric", _numpy_core_numeric)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.loader import (
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_observed_u_array,
    load_x_frame,
)
from data.dataset_metadata import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
)
from experiments.policy_utils import (
    artifact_policy_features as _artifact_policy_features,
    constant_softmax_theta as _constant_policy_theta,
    load_acceptance_floor as _acceptance_floor,
    optimization_trace_summary as _trace_payload,
)
from experiments.provenance import file_record as _file_record, file_sha256 as _sha256_file
from experiments.policy_artifacts import load_policy_artifact
from objective.gridded import SplineSupportLowerBoundObjective
from objective.objectives.generali.model_based import ModelBasedObjective
from objective.policy import ConstantPolicy, IdentityFeatureMap, SoftmaxPolicy
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize
from reporting.profit_dispersion import (
    load_deterministic_sample_rows,
    model_based_objective_matrix,
    row_index_sha256,
)
from reporting.exact_spline_cache import load_or_build_exact_spline_response_grid
from reporting.real_data import plot_support_action_overlay as _plot_overlay


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
DEFAULT_GLM_REFERENCE_POLICY = (
    RESULTS_ROOT
    / "glm-softmax-80-20-first-order"
    / "glm-softmax-80-20-first-order"
    / "seeds"
    / "seed-8"
    / "policies"
    / "first_order"
    / "policy.json"
)
DEFAULT_SPLINE_TAIL_REFERENCE_ACTIONS = (
    RESULTS_ROOT
    / "spline-xgb-synthetic-tail-140-lower-bound-policy-20k"
    / "support_lower_bound_policy_full_actions.npz"
)
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
    parser.add_argument(
        "--mean-model",
        choices=("spline-xgb", "glm"),
        default="spline-xgb",
        help="Customer-level mean-profit model optimized beneath the support penalty.",
    )
    parser.add_argument(
        "--glm-reference-policy",
        type=Path,
        default=DEFAULT_GLM_REFERENCE_POLICY,
        help="Saved repository-optimizer GLM policy used only for like-for-like comparison.",
    )
    parser.add_argument(
        "--spline-tail-reference-actions",
        type=Path,
        default=DEFAULT_SPLINE_TAIL_REFERENCE_ACTIONS,
        help="Saved spline/XGBoost-tail optimizer actions used for comparison.",
    )
    parser.add_argument(
        "--response-cache",
        type=Path,
        default=None,
        help="Optional existing exact-spline response cache to reuse.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--initial-u", type=float, default=DEFAULT_INITIAL_U)
    parser.add_argument(
        "--lower-only-cloud",
        action="store_true",
        help="Plot only the region between mean profit and its lower bound.",
    )
    return parser


def _load_or_build_glm_response_grid(
    *,
    cache_path: Path,
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    action_grid: np.ndarray,
    acceptance_artifact: Any,
    loss_artifact: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or build the exact GLM acceptance and objective grid."""
    identity = {
        "row_indices_sha256": row_index_sha256(row_indices),
        "action_grid_sha256": hashlib.sha256(
            np.asarray(action_grid, dtype="<f8").tobytes()
        ).hexdigest(),
        "acceptance_artifact_sha256": _sha256_file(
            Path(acceptance_artifact.artifact_path)
        ),
        "loss_artifact_sha256": _sha256_file(Path(loss_artifact.artifact_path)),
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_identity = {
                key: str(cached[key].item()) if key in cached else ""
                for key in identity
            }
            acceptance = cached["acceptance"].astype(float)
            cost = cached["cost"].astype(float)
        if cached_identity == identity and acceptance.shape == cost.shape == (
            len(row_indices),
            len(action_grid),
        ):
            return acceptance, cost

    objective = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=acceptance_artifact,
        loss_model=loss_artifact,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
    )
    acceptance = np.column_stack(
        [
            objective._acceptance_proba(
                frame,
                np.full(len(frame), float(proposed_u), dtype=float),
            )
            for proposed_u in action_grid
        ]
    )
    cost = model_based_objective_matrix(
        objective,
        frame,
        action_grid,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        acceptance=acceptance.astype(np.float32),
        cost=cost.astype(np.float32),
        **{key: np.asarray(value) for key, value in identity.items()},
    )
    return acceptance, cost


def _reanchor_support_frame(
    source: pd.DataFrame,
    mean_profit: np.ndarray,
    support_width: np.ndarray,
) -> pd.DataFrame:
    """Put the unchanged support/tail penalty underneath a new mean curve."""
    frame = source.sort_values("u").reset_index(drop=True).copy()
    mean = np.asarray(mean_profit, dtype=float)
    penalty = np.asarray(support_width, dtype=float)
    if mean.shape != ACTION_GRID.shape or penalty.shape != ACTION_GRID.shape:
        raise ValueError("Mean profit and support penalty must align to ACTION_GRID.")
    base_width = frame["smoothed_support_half_width"].to_numpy(dtype=float)
    frame["mean_profit"] = mean
    frame["smoothed_mean_profit"] = mean
    frame["original_support_cloud_lower_profit"] = mean - base_width
    frame["support_cloud_lower_profit"] = mean - penalty
    frame["support_cloud_upper_profit"] = mean + base_width
    frame["optimization_support_penalty"] = penalty
    return frame


def _evaluation_row(
    *,
    policy_name: str,
    population: str,
    actions: np.ndarray,
    frame: pd.DataFrame,
    objective: ModelBasedObjective,
    support_width: np.ndarray | None,
) -> dict[str, float | int | str]:
    """Evaluate one optimizer-derived or historical action vector under GLM."""
    u = np.asarray(actions, dtype=float).reshape(-1)
    if u.shape != (len(frame),) or not np.isfinite(u).all():
        raise ValueError(f"Invalid action vector for {policy_name} on {population}.")
    acceptance = np.asarray(objective._acceptance_proba(frame, u), dtype=float)
    loss = np.asarray(objective._loss_prediction(frame), dtype=float)
    premium = np.asarray(objective._premium_values(frame), dtype=float)
    profit = -np.asarray(
        objective._value_batch_from_components(acceptance, loss, premium, u),
        dtype=float,
    )
    quantiles = np.quantile(u, [0.01, 0.25, 0.5, 0.75, 0.99])
    row: dict[str, float | int | str] = {
        "policy": policy_name,
        "population": population,
        "n_customers": int(u.size),
        "mean_action": float(np.mean(u)),
        "population_std_action": float(np.std(u, ddof=0)),
        "action_p01": float(quantiles[0]),
        "action_p25": float(quantiles[1]),
        "action_p50": float(quantiles[2]),
        "action_p75": float(quantiles[3]),
        "action_p99": float(quantiles[4]),
        "glm_mean_acceptance": float(np.mean(acceptance)),
        "glm_mean_profit": float(np.mean(profit)),
        "glm_total_profit": float(np.sum(profit)),
    }
    if support_width is not None:
        from scipy.interpolate import CubicSpline

        if np.any((u < ACTION_LOW - 1e-10) | (u > ACTION_HIGH + 1e-10)):
            raise ValueError(
                f"Support-adjusted evaluation requires bounded actions for {policy_name}."
            )
        bounded_u = np.clip(u, ACTION_LOW, ACTION_HIGH)
        penalty = np.asarray(
            CubicSpline(ACTION_GRID, support_width, bc_type="natural")(bounded_u),
            dtype=float,
        )
        row["mean_support_penalty"] = float(np.mean(penalty))
        row["glm_support_adjusted_mean_profit"] = float(np.mean(profit - penalty))
        row["glm_support_adjusted_total_profit"] = float(np.sum(profit - penalty))
    return row


def _load_support_width(csv_path: Path, manifest_path: Path) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("support", {}).get("baseline_subtracted") is not False:
        raise ValueError("Support cloud must retain absolute baseline risk.")
    frame = pd.read_csv(csv_path).sort_values("u")
    if not np.allclose(frame["u"].to_numpy(dtype=float), ACTION_GRID):
        raise ValueError("Support cloud must cover the full [-0.1, 0.2] grid.")
    width_column = (
        "optimization_support_penalty"
        if "optimization_support_penalty" in frame.columns
        else "smoothed_support_half_width"
    )
    width = frame[width_column].to_numpy(dtype=float)
    if not np.isfinite(width).all() or np.any(width <= 0.0):
        raise ValueError("Support width must be finite and positive everywhere.")
    return width


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
    frame: pd.DataFrame,
) -> np.ndarray:
    print(f"Applying saved 20k policy to {len(frame):,} eligible customers...", flush=True)
    artifact_features = _artifact_policy_features(acceptance_artifact, frame)
    policy_features = policy_preprocessor.transform(artifact_features)
    return np.clip(
        np.asarray(policy.value(theta, policy_features), dtype=float),
        ACTION_LOW,
        ACTION_HIGH,
    )


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
    if args.mean_model == "glm":
        acceptance_artifact, loss_artifact = load_model_artifact_pair(
            "linear", "linear"
        )
    else:
        acceptance_artifact, loss_artifact = load_model_artifact_pair("xgb", "xgb")
    for artifact in (acceptance_artifact, loss_artifact):
        if args.mean_model != "glm" and hasattr(artifact.model, "set_params"):
            available = artifact.model.get_params(deep=False)
            if "n_jobs" in available:
                artifact.model.set_params(n_jobs=int(args.n_jobs))

    default_cache_name = (
        "sample_glm_response_grid.npz"
        if args.mean_model == "glm"
        else "sample_exact_spline_response_grid.npz"
    )
    response_cache = (
        output_dir / default_cache_name
        if args.response_cache is None
        else args.response_cache.resolve()
    )
    if args.mean_model == "glm":
        acceptance_grid, cost_grid = _load_or_build_glm_response_grid(
            cache_path=response_cache,
            frame=frame,
            row_indices=sample_rows,
            action_grid=ACTION_GRID,
            acceptance_artifact=acceptance_artifact,
            loss_artifact=loss_artifact,
        )
    else:
        acceptance_grid, cost_grid = load_or_build_exact_spline_response_grid(
            cache_path=response_cache,
            frame=frame,
            row_indices=sample_rows,
            eligible_rows=eligible_rows,
            action_grid=ACTION_GRID,
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

    full_frame = load_x_frame("xgb", row_indices=eligible_rows)
    full_actions = _apply_policy_to_full_population(
        theta=theta,
        policy=policy,
        policy_preprocessor=policy_preprocessor,
        acceptance_artifact=acceptance_artifact,
        frame=full_frame,
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

    source_support_frame = pd.read_csv(args.support_csv)
    support_frame_path: Path | None = None
    if args.mean_model == "glm":
        support_frame = _reanchor_support_frame(
            source_support_frame,
            -np.mean(cost_grid, axis=0),
            support_width,
        )
        support_frame_path = output_dir / "glm_synthetic_tail_support_cloud.csv"
        support_frame.to_csv(support_frame_path, index=False)
    else:
        support_frame = source_support_frame
    overlay_path = output_dir / "mean_profit_support_cloud_with_lower_bound_policy.pdf"
    _plot_overlay(
        support_frame,
        histogram,
        overlay_path,
        lower_only=bool(args.lower_only_cloud),
    )

    comparison_path: Path | None = None
    comparison_rows: list[dict[str, float | int | str]] = []
    if args.mean_model == "glm":
        glm_objective = ModelBasedObjective(
            policy=ConstantPolicy(),
            acceptance_model=acceptance_artifact,
            loss_model=loss_artifact,
            acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
            loss_cols=tuple(LOSS_FEATURE_COLS),
            premium_col=PREMIUM_COL,
        )
        historical_full_actions = load_observed_u_array(
            "xgb", row_indices=eligible_rows
        )
        glm_reference = load_policy_artifact(args.glm_reference_policy)
        glm_reference_full_actions = np.asarray(
            glm_reference.predict_u(x_batch=full_frame), dtype=float
        )
        glm_reference_sample_actions = np.asarray(
            glm_reference.predict_u(x_batch=frame), dtype=float
        )
        with np.load(args.spline_tail_reference_actions, allow_pickle=False) as saved:
            spline_tail_rows = np.asarray(saved["row_indices"], dtype=int)
            spline_tail_full_actions = np.asarray(saved["actions"], dtype=float)
        if not np.array_equal(spline_tail_rows, eligible_rows):
            raise ValueError("Spline-tail reference actions do not align to eligible rows.")
        sample_positions = np.searchsorted(eligible_rows, sample_rows)
        if not np.array_equal(eligible_rows[sample_positions], sample_rows):
            raise ValueError("Deterministic sample rows are not aligned to eligible rows.")
        spline_tail_sample_actions = spline_tail_full_actions[sample_positions]
        historical_sample_actions = historical_full_actions[sample_positions]

        for population, eval_frame, policies in (
            (
                "deterministic_20k",
                frame,
                (
                    ("historical_actions", historical_sample_actions, None),
                    ("glm_reference_policy", glm_reference_sample_actions, support_width),
                    ("spline_xgb_synthetic_tail_policy", spline_tail_sample_actions, support_width),
                    ("glm_synthetic_tail_policy", sample_actions, support_width),
                ),
            ),
            (
                "full_eligible",
                full_frame,
                (
                    ("historical_actions", historical_full_actions, None),
                    ("glm_reference_policy", glm_reference_full_actions, support_width),
                    ("spline_xgb_synthetic_tail_policy", spline_tail_full_actions, support_width),
                    ("glm_synthetic_tail_policy", full_actions, support_width),
                ),
            ),
        ):
            for policy_name, actions, width in policies:
                comparison_rows.append(
                    _evaluation_row(
                        policy_name=policy_name,
                        population=population,
                        actions=actions,
                        frame=eval_frame,
                        objective=glm_objective,
                        support_width=width,
                    )
                )
        comparison = pd.DataFrame(comparison_rows)
        for population in comparison["population"].unique():
            mask = comparison["population"] == population
            historical_total = float(
                comparison.loc[
                    mask & (comparison["policy"] == "historical_actions"),
                    "glm_total_profit",
                ].iloc[0]
            )
            comparison.loc[mask, "glm_total_profit_uplift_vs_historical"] = (
                comparison.loc[mask, "glm_total_profit"] - historical_total
            )
        comparison_path = output_dir / "glm_policy_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

    full_quantiles = np.quantile(
        full_actions,
        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    )
    output_candidates = [
        policy_path,
        full_actions_path,
        histogram_path,
        overlay_path,
    ]
    if support_frame_path is not None:
        output_candidates.append(support_frame_path)
    if comparison_path is not None:
        output_candidates.append(comparison_path)
    summary = {
        "analysis": f"20k-{args.mean_model}-support-lower-bound-softmax-policy",
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
            "mean_profit_family": args.mean_model,
            "acceptance": (
                "GLM acceptance"
                if args.mean_model == "glm"
                else "exact monotone-spline XGBoost"
            ),
            "loss": (
                "GLM financial risk"
                if args.mean_model == "glm"
                else "XGBoost financial risk"
            ),
            "spline_boundary_rules": (
                None
                if args.mean_model == "glm"
                else "constant left; clipped-linear right"
            ),
            "spline_construction": (
                None
                if args.mean_model == "glm"
                else (
                    "17 raw-XGBoost anchors, weighted smoothing spline, isotonic "
                    "projection, then PCHIP"
                )
            ),
            "acceptance_artifact": _file_record(
                Path(acceptance_artifact.artifact_path)
            ),
            "loss_artifact": _file_record(Path(loss_artifact.artifact_path)),
        },
        "policy": {
            "class": "objective.policy.SoftmaxPolicy",
            "feature_map": (
                "IdentityFeatureMap over fitted standardized/sphered "
                f"{args.mean_model} artifact features"
            ),
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
            "support_penalty_column": (
                "optimization_support_penalty"
                if "optimization_support_penalty"
                in pd.read_csv(args.support_csv, nrows=1).columns
                else "smoothed_support_half_width"
            ),
            "acceptance_floor": floor,
        },
        "plot": {
            "lower_bound_only": bool(args.lower_only_cloud),
            "upper_bound_displayed": not bool(args.lower_only_cloud),
        },
        "initial_policy_on_sample": initial_summary,
        "optimized_policy_on_sample": final_summary,
        "glm_policy_comparison": comparison_rows if comparison_rows else None,
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
            "response_cache": _file_record(response_cache),
            "glm_reference_policy": (
                _file_record(args.glm_reference_policy)
                if args.mean_model == "glm"
                else None
            ),
            "spline_tail_reference_actions": (
                _file_record(args.spline_tail_reference_actions)
                if args.mean_model == "glm"
                else None
            ),
        },
        "outputs": {
            path.name: _sha256_file(path)
            for path in output_candidates
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [*output_candidates, summary_path]
    for output in outputs:
        print(output, flush=True)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
