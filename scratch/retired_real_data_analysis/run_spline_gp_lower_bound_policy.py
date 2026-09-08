#!/usr/bin/env python3
"""Construct a synthetic GP lower bound and fit the 20k spline policy to it.

The GP uses the saved deterministic 20,000-customer exact-spline/XGBoost
mean-profit curve as its mean function. Fixed zero-residual observations inside
``[0, 0.16]`` make posterior uncertainty small there and smoothly increasing
outside. The lower confidence bound is then optimized with the repository
first-order trust-constr setup; no grid search selects or reports a solution.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.loader import (
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_x_frame,
)
from experiments.policy_utils import (
    artifact_policy_features as _artifact_policy_features,
    constant_softmax_theta as _constant_policy_theta,
    load_acceptance_floor as _acceptance_floor,
    optimization_trace_summary as _trace_payload,
)
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from objective.gridded import SplineSupportLowerBoundObjective
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from optimization.solvers import run_first_order_minimize
from reporting.profit_dispersion import (
    load_deterministic_sample_rows,
    row_index_sha256,
)
from reporting.exact_spline_cache import load_or_build_exact_spline_response_grid


def _default_results_root() -> Path:
    """Resolve the shared project results directory from checkout or worktree."""
    if ROOT.parent.name == "worktrees":
        return ROOT.parent.parent / "results"
    return ROOT.parent / "results"


RESULTS_ROOT = _default_results_root()
DEFAULT_CURVE_DIR = (
    RESULTS_ROOT / "glm-spline-objective-dispersion-minus010-plus020"
)
DEFAULT_DIAGNOSTICS = (
    RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_REFERENCE_POLICY = (
    RESULTS_ROOT
    / "xgboost-full-dataset-historical-support"
    / "optimized_policy_cropped.npz"
)
DEFAULT_RESPONSE_CACHE = (
    RESULTS_ROOT
    / "spline-xgb-support-lower-bound-policy-20k"
    / "sample_exact_spline_response_grid.npz"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "spline-xgb-synthetic-gp-lower-bound-policy-20k"

ACTION_GRID = np.linspace(-0.1, 0.2, 301)
ACTION_LOW = float(ACTION_GRID[0])
ACTION_HIGH = float(ACTION_GRID[-1])
DEFAULT_INITIAL_U = 0.08
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLE_SEED = 20260831

GP_SUPPORT_LOW = 0.0
GP_SUPPORT_HIGH = 0.16
GP_DESIGN_COUNT = 33
GP_AMPLITUDE = 35.0
GP_LENGTH_SCALE = 0.01
GP_NOISE_STD = 0.35
GP_LCB_MULTIPLIER = 1.0
DEFAULT_MAX_STEPS = 1_000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=DEFAULT_CURVE_DIR / "profit_dispersion_curves.csv",
    )
    parser.add_argument(
        "--curve-manifest",
        type=Path,
        default=DEFAULT_CURVE_DIR / "run_manifest.json",
    )
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument(
        "--acceptance-reference-policy",
        type=Path,
        default=DEFAULT_REFERENCE_POLICY,
    )
    parser.add_argument("--response-grid-cache", type=Path, default=DEFAULT_RESPONSE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--initial-u", type=float, default=DEFAULT_INITIAL_U)
    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mean_profit_curve(
    curve_csv: Path,
    curve_manifest: Path,
    diagnostics_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the exact 20k spline/XGBoost mean-profit curve with provenance."""
    manifest = json.loads(curve_manifest.read_text(encoding="utf-8"))
    expected_model = {"acceptance": "monotone_spline_xgb", "loss": "xgb"}
    if manifest.get("model_pairs", {}).get("spline") != expected_model:
        raise ValueError("Curve manifest does not identify the spline/XGBoost pair.")
    objective = manifest.get("objective", {})
    if objective.get("class") != "ModelBasedObjective":
        raise ValueError("Curve manifest must identify ModelBasedObjective.")
    if objective.get("plot_transform") != "profit_i(u) = -objective_i(u)":
        raise ValueError("Curve manifest must identify the objective-to-profit sign flip.")

    with np.load(diagnostics_path, allow_pickle=False) as diagnostics:
        if "row_indices" not in diagnostics.files:
            raise ValueError("Coverage diagnostics do not contain row_indices.")
        row_indices = diagnostics["row_indices"].astype(int)
    if row_indices.size != DEFAULT_SAMPLE_SIZE:
        raise ValueError("Expected the deterministic 20,000-customer sample.")
    actual_hash = row_index_sha256(row_indices)
    if manifest.get("sample", {}).get("row_indices_sha256") != actual_hash:
        raise ValueError("Curve and coverage diagnostics use different customer samples.")

    curves = pd.read_csv(curve_csv)
    selected = curves.loc[
        (curves["model_family"] == "spline")
        & (curves["center_statistic"] == "mean")
        & (curves["acceptance_model"] == "monotone_spline_xgb")
        & (curves["loss_model"] == "xgb")
    ].sort_values("u")
    if len(selected) != ACTION_GRID.size or selected["u"].duplicated().any():
        raise ValueError("Expected one spline mean-profit value per action-grid point.")
    if not np.allclose(
        selected["u"].to_numpy(dtype=float),
        ACTION_GRID,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Mean-profit curve must use the 301-point [-0.1, 0.2] grid.")
    mean_profit = selected["center"].to_numpy(dtype=float)
    if not np.isfinite(mean_profit).all():
        raise ValueError("Mean-profit curve must be finite.")
    return mean_profit, {
        "row_indices": row_indices,
        "row_indices_sha256": actual_hash,
        "curve_manifest": manifest,
    }


def _synthetic_gp_posterior_std(
    action_grid: np.ndarray,
    *,
    support_low: float = GP_SUPPORT_LOW,
    support_high: float = GP_SUPPORT_HIGH,
    design_count: int = GP_DESIGN_COUNT,
    amplitude: float = GP_AMPLITUDE,
    length_scale: float = GP_LENGTH_SCALE,
    noise_std: float = GP_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-design RBF-GP posterior SD and its conditioning actions."""
    grid = np.asarray(action_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("action_grid must be a finite one-dimensional array.")
    if not support_low < support_high:
        raise ValueError("support_low must be below support_high.")
    if design_count < 2 or min(amplitude, length_scale, noise_std) <= 0.0:
        raise ValueError("GP design count and hyperparameters must be positive.")

    design = np.linspace(float(support_low), float(support_high), int(design_count))
    kernel = ConstantKernel(
        constant_value=float(amplitude) ** 2,
        constant_value_bounds="fixed",
    ) * RBF(length_scale=float(length_scale), length_scale_bounds="fixed")
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(noise_std) ** 2,
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(design[:, None], np.zeros(design.size, dtype=float))
    posterior_mean, posterior_std = gp.predict(grid[:, None], return_std=True)
    if not np.allclose(posterior_mean, 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Zero-residual GP posterior mean was unexpectedly nonzero.")
    if not np.isfinite(posterior_std).all() or np.any(posterior_std <= 0.0):
        raise RuntimeError("GP posterior standard deviation must be finite and positive.")
    return np.asarray(posterior_std, dtype=float), design


def _gp_lower_bound_frame(mean_profit: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    posterior_std, design = _synthetic_gp_posterior_std(ACTION_GRID)
    width = GP_LCB_MULTIPLIER * posterior_std
    mean = np.asarray(mean_profit, dtype=float)
    if mean.shape != ACTION_GRID.shape:
        raise ValueError("mean_profit must align to ACTION_GRID.")
    frame = pd.DataFrame(
        {
            "u": ACTION_GRID,
            "posterior_mean_profit": mean,
            "posterior_std_profit": posterior_std,
            "gp_lower_bound_half_width": width,
            "gp_lower_confidence_bound_profit": mean - width,
            "gp_upper_confidence_bound_profit": mean + width,
        }
    )
    return frame, design


def _plot_gp_lower_bound(frame: pd.DataFrame, output_path: Path) -> None:
    u = frame["u"].to_numpy(dtype=float)
    mean = frame["posterior_mean_profit"].to_numpy(dtype=float)
    lower = frame["gp_lower_confidence_bound_profit"].to_numpy(dtype=float)
    upper = frame["gp_upper_confidence_bound_profit"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        color="tab:orange",
        alpha=0.25,
    )
    ax.plot(u, mean, color="C0", linewidth=2.0)
    ax.set_title("Mean Predicted Profit Per Customer vs. Price Change", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _load_response_grid_cache(
    path: Path,
    sample_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        required = {"row_indices", "action_grid", "acceptance", "cost"}
        if not required.issubset(cached.files):
            return None
        if not np.array_equal(cached["row_indices"], sample_rows):
            return None
        if not np.allclose(cached["action_grid"], ACTION_GRID):
            return None
        acceptance = cached["acceptance"].astype(float)
        cost = cached["cost"].astype(float)
    expected = (sample_rows.size, ACTION_GRID.size)
    if acceptance.shape != expected or cost.shape != expected:
        return None
    if not np.isfinite(acceptance).all() or not np.isfinite(cost).all():
        return None
    return acceptance, cost


def _summary_with_gp_names(summary: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(summary)
    renamed["mean_gp_lower_bound_half_width"] = renamed.pop(
        "mean_support_half_width"
    )
    renamed["gp_lower_bound_mean_profit"] = renamed.pop(
        "support_lower_bound_mean_profit"
    )
    return renamed


def run_analysis(args: argparse.Namespace) -> list[Path]:
    if int(args.max_steps) <= 0 or int(args.n_jobs) == 0:
        raise ValueError("max_steps must be positive and n_jobs cannot be zero.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_profit, provenance = _load_mean_profit_curve(
        args.curve_csv,
        args.curve_manifest,
        args.diagnostics,
    )
    sample_rows = load_deterministic_sample_rows(
        args.diagnostics,
        eligible_csv_row_indices("xgb"),
        sample_size=DEFAULT_SAMPLE_SIZE,
        seed=DEFAULT_SAMPLE_SEED,
    )
    if not np.array_equal(sample_rows, provenance["row_indices"]):
        raise ValueError("Saved diagnostics do not reproduce the deterministic sample.")

    gp_frame, gp_design = _gp_lower_bound_frame(mean_profit)
    gp_csv_path = output_dir / "synthetic_gp_lower_bound_20k.csv"
    gp_frame.to_csv(gp_csv_path, index=False)
    gp_pdf_path = output_dir / "synthetic_gp_lower_bound_20k.pdf"
    _plot_gp_lower_bound(gp_frame, gp_pdf_path)

    frame = load_x_frame("xgb", row_indices=sample_rows)
    acceptance_artifact, loss_artifact = load_model_artifact_pair("xgb", "xgb")
    for artifact in (acceptance_artifact, loss_artifact):
        if hasattr(artifact.model, "set_params"):
            available = artifact.model.get_params(deep=False)
            if "n_jobs" in available:
                artifact.model.set_params(n_jobs=int(args.n_jobs))

    cached = _load_response_grid_cache(args.response_grid_cache, sample_rows)
    if cached is None:
        response_grid_source = output_dir / "sample_exact_spline_response_grid.npz"
        acceptance_grid, cost_grid = load_or_build_exact_spline_response_grid(
            cache_path=response_grid_source,
            frame=frame,
            row_indices=sample_rows,
            eligible_rows=eligible_csv_row_indices("xgb"),
            action_grid=ACTION_GRID,
            acceptance_artifact=acceptance_artifact,
            loss_artifact=loss_artifact,
            n_jobs=int(args.n_jobs),
        )
    else:
        acceptance_grid, cost_grid = cached
        response_grid_source = args.response_grid_cache.resolve()
        print(f"Reusing {response_grid_source}", flush=True)

    artifact_features = _artifact_policy_features(acceptance_artifact, frame)
    policy_preprocessor = PolicyFeaturePreprocessor(standardize=True, sphere=True)
    policy_features = policy_preprocessor.fit_transform(artifact_features)
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
    )
    acceptance_floor = _acceptance_floor(args.acceptance_reference_policy)
    objective = SplineSupportLowerBoundObjective(
        policy=policy,
        cost_grid=cost_grid,
        acceptance_grid=acceptance_grid,
        action_grid=ACTION_GRID,
        support_width=gp_frame["gp_lower_bound_half_width"].to_numpy(dtype=float),
        acceptance_floor=acceptance_floor,
    )
    theta0 = _constant_policy_theta(
        policy,
        policy_features.shape[1],
        float(args.initial_u),
    )
    initial_summary = _summary_with_gp_names(
        objective.summarize(theta0, policy_features)
    )
    if initial_summary["mean_acceptance"] < acceptance_floor:
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
    optimized_summary = _summary_with_gp_names(
        objective.summarize(theta, policy_features)
    )
    if optimized_summary["mean_acceptance"] < acceptance_floor - 1e-8:
        raise RuntimeError("Converged policy violates the acceptance constraint.")
    actions = np.asarray(policy.value(theta, policy_features), dtype=float)
    outside = (actions < GP_SUPPORT_LOW) | (actions > GP_SUPPORT_HIGH)

    state = policy_preprocessor.to_state()
    policy_path = output_dir / "synthetic_gp_lower_bound_policy_20k.npz"
    np.savez_compressed(
        policy_path,
        row_indices=sample_rows,
        theta=theta,
        actions=actions,
        action_bounds=np.asarray([ACTION_LOW, ACTION_HIGH]),
        gp_support_interval=np.asarray([GP_SUPPORT_LOW, GP_SUPPORT_HIGH]),
        gp_design_actions=gp_design,
        gp_posterior_std=gp_frame["posterior_std_profit"].to_numpy(dtype=float),
        acceptance_floor=acceptance_floor,
        preprocessor_mean=state["arrays"]["mean"],
        preprocessor_scale=state["arrays"]["scale"],
        preprocessor_eigenvalues=state["arrays"]["eigenvalues"],
        preprocessor_transform_matrix=state["arrays"]["transform_matrix"],
    )

    summary = {
        "analysis": "20k-spline-xgb-synthetic-gp-lower-bound-softmax-policy",
        "sample": {
            "n_customers": int(sample_rows.size),
            "seed": DEFAULT_SAMPLE_SEED,
            "row_indices_sha256": provenance["row_indices_sha256"],
            "selection": "sorted choice without replacement from eligible rows",
        },
        "model": {
            "objective": "ModelBasedObjective",
            "acceptance": "monotone_spline_xgb",
            "loss": "xgb",
            "plot_transform": "profit_i(u) = -objective_i(u)",
        },
        "synthetic_gp": {
            "index_domain": [ACTION_LOW, ACTION_HIGH],
            "mean_function": "saved exact 20k spline/XGBoost mean-profit curve",
            "residual_kernel": "amplitude^2 * RBF(length_scale)",
            "amplitude": GP_AMPLITUDE,
            "length_scale": GP_LENGTH_SCALE,
            "conditioning_actions": gp_design.tolist(),
            "conditioning_residuals": "all zero",
            "observation_noise_std": GP_NOISE_STD,
            "lcb_multiplier": GP_LCB_MULTIPLIER,
            "intended_low_uncertainty_interval": [GP_SUPPORT_LOW, GP_SUPPORT_HIGH],
            "randomness": "none; fixed design and fixed hyperparameters",
            "confidence_interpretation": (
                "synthetic design envelope, not an empirically calibrated confidence statement"
            ),
            "off_grid_optimizer_rule": (
                "natural cubic interpolation of posterior SD on the 301-point grid"
            ),
            "optimizer_boundary_behavior": (
                "softmax policy remains strictly inside [-0.1, 0.2]"
            ),
        },
        "policy": {
            "class": "objective.policy.SoftmaxPolicy",
            "formula": "u_i(theta) = -0.1 + 0.3 * sigmoid(theta_0 + theta_x^T z_i)",
            "theta": np.asarray(theta, dtype=float).tolist(),
            "preprocessor": state["metadata"],
        },
        "objective": {
            "direction": "minimize",
            "formula": "mean_i[ModelBasedObjective_i(u_i(theta)) + posterior_std(u_i(theta))]",
            "maximization_equivalent": "mean_i[profit_i(u_i(theta)) - posterior_std(u_i(theta))]",
            "acceptance_floor": acceptance_floor,
            "global_reference": "none; no grid scan or analytical optimum is reported",
        },
        "initial_policy_on_sample": initial_summary,
        "optimized_policy_on_sample": {
            **optimized_summary,
            "count_outside_0_to_0.16": int(np.count_nonzero(outside)),
            "fraction_outside_0_to_0.16": float(np.mean(outside)),
        },
        "optimizer": {
            "entry_point": "optimization.solvers.run_first_order_minimize",
            "step_rule": "trust-constr",
            "gradient": "first_order",
            "max_steps": int(args.max_steps),
            "runtime_seconds": runtime,
            **_trace_payload(trace),
        },
        "inputs": {
            "curve_csv": str(args.curve_csv.resolve()),
            "curve_manifest": str(args.curve_manifest.resolve()),
            "diagnostics": str(args.diagnostics.resolve()),
            "response_grid": str(response_grid_source),
            "acceptance_reference_policy": str(
                args.acceptance_reference_policy.resolve()
            ),
        },
        "representation": {
            "plot_line": "connected evaluations of the saved 301-point mean curve",
            "plot_band": "posterior mean +/- one posterior residual standard deviation",
            "sweep_or_draw_reuse": "not applicable; one deterministic GP construction",
        },
    }
    generated = [gp_pdf_path, gp_csv_path, policy_path]
    summary["outputs"] = {
        path.name: {"sha256": _sha256_file(path)} for path in generated
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generated.append(summary_path)
    for path in generated:
        print(path, flush=True)
    return generated


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
