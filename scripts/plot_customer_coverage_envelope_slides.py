"""Build an optimizer-derived visual demonstration of uncertainty-aware pricing.

The fold-0 XGBoost artifacts define the point objective throughout. Customer
coverage is estimated independently from historical neighbors:

* numeric customer features use the fold-0 covariance-whitened coordinates;
* categorical customer features use exact-match one-hot coordinates;
* historical price change U is excluded from customer distance and enters only
  through a separate Gaussian action kernel.

The uncertainty envelope is intentionally illustrative. Its shape is determined
by empirical local coverage, while its scale is fixed at 10 objective units so
that the decision consequence is visible in a slide. It is not presented as a
calibrated confidence interval. Every displayed solution is returned by the
repository's minimization optimizer with its finite-difference gradient
estimator. The optimizer receives the XGBoost cost objective; plots negate that
cost so they retain the profit convention where higher is better. Sampled action
grids are used only to construct and render the optimizer-facing natural cubic
splines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors

from data.loader import (
    eligible_csv_row_indices,
    load_model_artifacts,
    load_observed_u_array,
    load_x_frame,
)
from objective.base import Objective
from objective.policy import AdditiveChebyshevFeatureMap, SoftmaxPolicy
from optimization.solvers import run_finite_difference_minimize


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "customer-coverage-envelope-slides"
FULL_OBJECTIVE_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "xgboost-full-dataset-historical-support"
    / "xgboost_objective_minus010_plus020.csv"
)
OPTIMIZED_POLICY_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "xgboost-full-dataset-historical-support"
    / "optimized_policy_cropped.npz"
)
U_GRID = np.linspace(0.0, 0.16, 161)
EXPLORATORY_U_GRID = np.linspace(-0.10, 0.20, 301)
NUMERIC_CLIP = 6.0
ACTION_BANDWIDTH = 0.01
ILLUSTRATIVE_WIDTH_SCALE = 10.0
MAD_TO_NORMAL_STD = 1.4826
MAD_CLOUD_MULTIPLIER = 0.6
SUPPORT_BAND_MAX_HALF_WIDTH = 10.0
GAUSSIAN_SMOOTH_SIGMA = 1.25
GAUSSIAN_SMOOTH_TRUNCATE = 4.0
OPTIMIZER_START_U = 0.08
OPTIMIZER_SIGMA_U = 0.001
OPTIMIZER_GRAD_NORM_TOL = 1e-8
OPTIMIZER_FTOL = 1e-12
OPTIMIZER_T_STEPS = 1_000
OPTIMIZER_X_SAMPLES = np.zeros((1, 1), dtype=float)


class _SplineMinimizationObjective(Objective):
    """Bounded constant-policy objective queried by the repository optimizer."""

    def __init__(
        self,
        u_grid: np.ndarray,
        objective_values: np.ndarray,
    ) -> None:
        u = np.asarray(u_grid, dtype=float)
        values = np.asarray(objective_values, dtype=float)
        if u.ndim != 1 or values.shape != u.shape or len(u) < 2:
            raise ValueError(
                "u_grid and objective_values must be matching one-dimensional arrays."
            )
        if not np.isfinite(u).all() or not np.isfinite(values).all():
            raise ValueError("u_grid and objective_values must be finite.")
        if not np.all(np.diff(u) > 0.0):
            raise ValueError("u_grid must be strictly increasing.")

        self.action_low = float(u[0])
        self.action_high = float(u[-1])
        self.policy = SoftmaxPolicy(
            feature_map=AdditiveChebyshevFeatureMap(max_degree=0),
            action_low=self.action_low,
            action_high=self.action_high,
        )
        self._interpolant = CubicSpline(
            u,
            values,
            bc_type="natural",
            extrapolate=False,
        )

    def _evaluate_actions(self, actions: np.ndarray) -> np.ndarray:
        bounded = np.clip(
            np.asarray(actions, dtype=float),
            self.action_low,
            self.action_high,
        )
        return np.asarray(self._interpolant(bounded), dtype=float)

    def _value_batch(self, x_batch: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        del x_batch
        return self._evaluate_actions(u_array)

    def _value_batch_many(
        self,
        x_batch: np.ndarray,
        u_matrix: np.ndarray,
    ) -> np.ndarray:
        del x_batch
        return self._evaluate_actions(u_matrix)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions = self.policy.value(theta, x_batch)
        return float(np.mean(self._evaluate_actions(actions)))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del theta, x_batch
        raise NotImplementedError(
            "This objective is intentionally optimized with finite differences."
        )


def _predict_acceptance_matrix(artifact, frame: pd.DataFrame, u_grid: np.ndarray) -> np.ndarray:
    """Predict acceptance over a customer-by-action grid with one fixed fold."""
    processed = artifact.preprocessor.transform(
        frame.loc[:, list(artifact.x_feature_cols)]
    )
    processed_columns = list(processed.columns)
    base = processed.to_numpy(dtype=float)
    matrix = np.empty((len(frame), len(u_grid)), dtype=np.float32)
    for start in range(0, len(frame), 1_000):
        stop = min(start + 1_000, len(frame))
        batch_size = stop - start
        repeated = np.repeat(base[start:stop], len(u_grid), axis=0)
        model_frame = pd.DataFrame(repeated, columns=processed_columns)
        model_frame["U"] = np.tile(u_grid, batch_size)
        matrix[start:stop] = artifact.model.predict_proba(model_frame)[:, 1].reshape(
            batch_size, len(u_grid)
        )
    return matrix


def _predict_loss(artifact, frame: pd.DataFrame) -> np.ndarray:
    model_frame = artifact.model_frame(frame)
    return np.asarray(artifact.model.predict(model_frame), dtype=float)


def _smooth_display_curve(values: np.ndarray) -> np.ndarray:
    """Return the fixed smoothing used by both optimizer and plotted curve."""
    return gaussian_filter1d(
        np.asarray(values, dtype=float),
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )


def _mad_dispersion(customer_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return MAD and Gaussian-consistent robust scale at each action."""
    values = np.asarray(customer_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("customer_values must be a non-empty two-dimensional array.")
    customer_median = np.median(values, axis=0)
    mad = np.median(np.abs(values - customer_median[None, :]), axis=0)
    return mad, MAD_TO_NORMAL_STD * mad


def _mad_cloud_half_width(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return the requested 0.6-MAD cloud half-width on the primary grid."""
    exploratory_u = np.asarray(data["exploratory_u"], dtype=float)
    exploratory_mad = np.asarray(data["exploratory_customer_mad"], dtype=float)
    grid_tolerance = 1e-12
    primary_mask = (exploratory_u >= U_GRID[0] - grid_tolerance) & (
        exploratory_u <= U_GRID[-1] + grid_tolerance
    )
    if not np.allclose(exploratory_u[primary_mask], U_GRID):
        raise ValueError("The primary action grid is not nested in the exploratory grid.")
    return MAD_CLOUD_MULTIPLIER * _smooth_display_curve(
        exploratory_mad[primary_mask]
    )


def _marginal_action_effective_sample_size(
    observed_u: np.ndarray,
    u_grid: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    """Return Gaussian-kernel effective historical support at each action."""
    historical = np.asarray(observed_u, dtype=float)
    grid = np.asarray(u_grid, dtype=float)
    if historical.ndim != 1 or grid.ndim != 1:
        raise ValueError("observed_u and u_grid must be one-dimensional.")
    if len(historical) == 0 or not np.isfinite(historical).all():
        raise ValueError("observed_u must contain finite observations.")
    if not np.isfinite(grid).all() or bandwidth <= 0.0:
        raise ValueError("u_grid must be finite and bandwidth must be positive.")

    effective_sample_size = np.empty(grid.shape, dtype=float)
    for index, proposed_u in enumerate(grid):
        weights = np.exp(-0.5 * ((historical - proposed_u) / bandwidth) ** 2)
        weight_sum = float(np.sum(weights))
        squared_weight_sum = float(np.sum(np.square(weights)))
        effective_sample_size[index] = weight_sum**2 / squared_weight_sum
    return effective_sample_size


def _support_weighted_band_half_width(
    effective_sample_size: np.ndarray,
    *,
    max_half_width: float = SUPPORT_BAND_MAX_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale inverse-root effective support into an illustrative plot band."""
    ess = np.asarray(effective_sample_size, dtype=float)
    if ess.ndim != 1 or len(ess) == 0 or not np.all(ess > 0.0):
        raise ValueError("effective_sample_size must be a positive 1D array.")
    if max_half_width < 0.0:
        raise ValueError("max_half_width must be non-negative.")

    relative_ess = ess / float(np.max(ess))
    relative_risk = np.sqrt(1.0 / relative_ess)
    half_width = max_half_width * relative_risk / float(np.max(relative_risk))
    return relative_ess, relative_risk, half_width


def _within_customer_change_summary(
    customer_profit: np.ndarray,
    baseline_profit: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize paired profit changes from one common baseline action."""
    profit = np.asarray(customer_profit, dtype=float)
    baseline = np.asarray(baseline_profit, dtype=float)
    if profit.ndim != 2 or profit.shape[0] < 1:
        raise ValueError("customer_profit must be a non-empty two-dimensional array.")
    if baseline.shape != (profit.shape[0],):
        raise ValueError("baseline_profit must contain one value per customer.")
    changes = profit - baseline[:, None]
    quantiles = np.quantile(changes, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    mad, robust_std = _mad_dispersion(changes)
    return {
        "q10": quantiles[0],
        "q25": quantiles[1],
        "median": quantiles[2],
        "q75": quantiles[3],
        "q90": quantiles[4],
        "mad": mad,
        "robust_std": robust_std,
    }


def _repo_spline_minimize(
    u_grid: np.ndarray,
    objective_values: np.ndarray,
    *,
    start_u: float,
) -> dict[str, float | int | bool | str]:
    """Minimize a scalar spline with the repository finite-difference setup."""
    objective = _SplineMinimizationObjective(u_grid, objective_values)
    start_fraction = (float(start_u) - objective.action_low) / (
        objective.action_high - objective.action_low
    )
    if not 0.0 < start_fraction < 1.0:
        raise ValueError("start_u must lie strictly inside the action domain.")
    theta_start = np.asarray(
        [np.log(start_fraction / (1.0 - start_fraction))],
        dtype=float,
    )
    theta_final, trace = run_finite_difference_minimize(
        theta_start,
        OPTIMIZER_X_SAMPLES,
        objective,
        t_steps=OPTIMIZER_T_STEPS,
        n_grad_samples=1,
        sigma=OPTIMIZER_SIGMA_U,
        perturbation_space="u",
        algorithm="l-bfgs-b",
        grad_norm_tol=OPTIMIZER_GRAD_NORM_TOL,
        ftol=OPTIMIZER_FTOL,
    )
    if not trace.optimizer_success:
        raise RuntimeError(
            "Repository finite-difference optimization failed: "
            f"{trace.optimizer_message}"
        )
    action = float(objective.policy.value(theta_final, OPTIMIZER_X_SAMPLES)[0])
    minimized_value = float(objective.value(theta_final, OPTIMIZER_X_SAMPLES))
    return {
        "u": action,
        "minimized_value": minimized_value,
        "theta": float(theta_final[0]),
        "success": bool(trace.optimizer_success),
        "status": int(trace.optimizer_status),
        "nit": max(0, len(trace.steps) - 1),
        "message": str(trace.optimizer_message),
    }


def _minimize_xgboost_objective(
    u_grid: np.ndarray,
    objective_values: np.ndarray,
) -> dict[str, float | int | bool | str]:
    """Minimize XGBoost cost and return its sign-flipped plotted profit."""
    solution = _repo_spline_minimize(
        u_grid,
        objective_values,
        start_u=OPTIMIZER_START_U,
    )
    solution["plotted_profit_value"] = -float(solution["minimized_value"])
    return solution


def _repo_multistart_spline_minimize(
    u_grid: np.ndarray,
    objective_values: np.ndarray,
    *,
    start_u_values: tuple[float, ...],
) -> dict[str, float | int | bool | str]:
    """Return the best successful repository-optimizer result across starts."""
    solutions: list[dict[str, float | int | bool | str]] = []
    failures: list[str] = []
    for start_u in start_u_values:
        try:
            solutions.append(
                _repo_spline_minimize(
                    u_grid,
                    objective_values,
                    start_u=start_u,
                )
            )
        except RuntimeError as error:
            failures.append(f"start_u={start_u}: {error}")
    if not solutions:
        raise RuntimeError(
            "All repository optimizer starts failed: " + "; ".join(failures)
        )
    return min(
        solutions,
        key=lambda solution: float(solution["minimized_value"]),
    )


def _load_saved_optimizer_actions(row_indices: np.ndarray) -> np.ndarray:
    """Load the saved constrained policy-optimizer actions for selected rows."""
    requested_rows = np.asarray(row_indices, dtype=int)
    with np.load(OPTIMIZED_POLICY_PATH, allow_pickle=False) as saved_policy:
        policy_rows = saved_policy["row_indices"].astype(int)
        policy_actions = saved_policy["actions"].astype(float)
    positions = np.searchsorted(policy_rows, requested_rows)
    if np.any(positions >= len(policy_rows)) or not np.array_equal(
        policy_rows[positions], requested_rows
    ):
        raise ValueError("The diagnostic sample is not contained in the saved policy rows.")
    return policy_actions[positions]


def _mixed_customer_embedding(artifact, frame: pd.DataFrame) -> np.ndarray:
    """Return interpretable mixed-type coordinates for customer similarity."""
    processor = artifact.preprocessor
    transformed = processor.transform(frame.loc[:, list(artifact.x_feature_cols)])
    numeric = transformed.loc[:, list(processor.numeric_feature_names_)].to_numpy(
        dtype=float
    )
    numeric = np.clip(numeric, -NUMERIC_CLIP, NUMERIC_CLIP)
    categorical = pd.get_dummies(
        frame.loc[:, list(processor.categorical_cols_)].astype("string"),
        dtype=float,
    ).to_numpy(dtype=float)
    # A categorical mismatch changes two one-hot coordinates. Scaling by
    # sqrt(2) makes one mismatch contribute one squared-distance unit.
    categorical /= np.sqrt(2.0)
    return np.column_stack([numeric, categorical]).astype(np.float32)


def _local_support_matrix(
    embedding: np.ndarray,
    observed_u: np.ndarray,
    u_grid: np.ndarray,
    *,
    n_neighbors: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Estimate local joint support over customer state and candidate action."""
    neighbor_count = min(int(n_neighbors) + 1, len(embedding))
    nearest = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="brute",
        metric="euclidean",
        n_jobs=int(n_jobs),
    ).fit(embedding)
    distances, indices = nearest.kneighbors(embedding)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:]
    historical_neighbor_u = np.asarray(observed_u, dtype=np.float32)[indices]
    bandwidth_index = min(max(neighbor_count // 2 - 1, 0), distances.shape[1] - 1)
    state_bandwidth = np.maximum(distances[:, bandwidth_index], 1e-6)
    state_weights = np.exp(
        -0.5 * (distances / state_bandwidth[:, None]) ** 2
    ).astype(np.float32)

    support = np.empty((len(embedding), len(u_grid)), dtype=np.float32)
    action_grid = np.asarray(u_grid, dtype=np.float32)
    for start in range(0, len(embedding), 200):
        stop = min(start + 200, len(embedding))
        action_distance = (
            historical_neighbor_u[start:stop, :, None] - action_grid[None, None, :]
        ) / ACTION_BANDWIDTH
        action_weights = np.exp(-0.5 * action_distance**2)
        support[start:stop] = np.sum(
            state_weights[start:stop, :, None] * action_weights,
            axis=1,
        )
    return support


def _compute_diagnostics(
    *,
    n_customers: int,
    seed: int,
    n_neighbors: int,
    n_jobs: int,
) -> dict[str, np.ndarray]:
    eligible = eligible_csv_row_indices("xgb")
    population_observed_u = load_observed_u_array("xgb", row_indices=eligible)
    baseline_u = float(np.median(population_observed_u))
    marginal_support_ess = _marginal_action_effective_sample_size(
        population_observed_u,
        U_GRID,
        bandwidth=ACTION_BANDWIDTH,
    )
    (
        marginal_support_relative_ess,
        marginal_support_relative_risk,
        marginal_support_band_half_width,
    ) = _support_weighted_band_half_width(marginal_support_ess)
    population_size = len(population_observed_u)
    del population_observed_u
    rng = np.random.default_rng(seed)
    row_indices = np.sort(
        rng.choice(eligible, size=min(int(n_customers), len(eligible)), replace=False)
    )
    frame = load_x_frame("xgb", row_indices=row_indices)
    observed_u = load_observed_u_array("xgb", row_indices=row_indices)
    acceptance_artifact, loss_artifact = load_model_artifacts("xgb")
    acceptance_artifact.model.set_params(n_jobs=int(n_jobs))
    loss_artifact.model.set_params(n_jobs=int(n_jobs))

    exploratory_acceptance = _predict_acceptance_matrix(
        acceptance_artifact,
        frame,
        EXPLORATORY_U_GRID,
    )
    loss = _predict_loss(loss_artifact, frame)
    premium = frame["X_policy_premium"].to_numpy(dtype=float)
    exploratory_revenue = premium[:, None] * (1.0 + EXPLORATORY_U_GRID[None, :])
    exploratory_customer_profit = exploratory_acceptance * (
        exploratory_revenue - loss[:, None]
    )
    exploratory_customer_std = np.std(
        exploratory_customer_profit,
        axis=0,
        ddof=1,
    )
    exploratory_customer_mad, exploratory_customer_robust_std = _mad_dispersion(
        exploratory_customer_profit
    )
    grid_tolerance = 1e-12
    primary_mask = (EXPLORATORY_U_GRID >= U_GRID[0] - grid_tolerance) & (
        EXPLORATORY_U_GRID <= U_GRID[-1] + grid_tolerance
    )
    if not np.allclose(EXPLORATORY_U_GRID[primary_mask], U_GRID):
        raise ValueError("The primary action grid is not nested in the exploratory grid.")
    customer_objective_std = exploratory_customer_std[primary_mask]

    baseline_acceptance = _predict_acceptance_matrix(
        acceptance_artifact,
        frame,
        np.asarray([baseline_u], dtype=float),
    )[:, 0]
    baseline_profit = baseline_acceptance * (
        premium * (1.0 + baseline_u) - loss
    )
    sensitivity_u = U_GRID
    sensitivity_customer_profit = exploratory_customer_profit[:, primary_mask]
    sensitivity = _within_customer_change_summary(
        sensitivity_customer_profit,
        baseline_profit,
    )
    sensitivity_display_robust_std = _smooth_display_curve(
        sensitivity["robust_std"]
    )
    dispersion_minimum = _repo_multistart_spline_minimize(
        sensitivity_u,
        sensitivity_display_robust_std,
        start_u_values=(0.04, 0.08, 0.12),
    )
    dispersion_maximum = _repo_multistart_spline_minimize(
        sensitivity_u,
        -sensitivity_display_robust_std,
        start_u_values=(0.02, 0.08, 0.14),
    )
    del (
        exploratory_acceptance,
        exploratory_revenue,
        exploratory_customer_profit,
        sensitivity_customer_profit,
    )
    baseline_policy_actions = _load_saved_optimizer_actions(row_indices)

    embedding = _mixed_customer_embedding(acceptance_artifact, frame)
    support = _local_support_matrix(
        embedding,
        observed_u,
        U_GRID,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    median_support = np.median(support, axis=0)

    full_curve = pd.read_csv(FULL_OBJECTIVE_PATH)
    exploratory_full_curve = full_curve.loc[
        full_curve["u"].between(EXPLORATORY_U_GRID[0], EXPLORATORY_U_GRID[-1])
    ].copy()
    if not np.allclose(
        exploratory_full_curve["u"].to_numpy(dtype=float),
        EXPLORATORY_U_GRID,
    ):
        raise ValueError(
            "The saved full-dataset objective grid does not match EXPLORATORY_U_GRID."
        )
    exploratory_mean_objective = exploratory_full_curve[
        "mean_objective"
    ].to_numpy(dtype=float)
    mean_objective = exploratory_mean_objective[primary_mask]

    support_deficit = 1.0 - median_support / float(np.max(median_support))
    illustrative_width = ILLUSTRATIVE_WIDTH_SCALE * support_deficit
    display_objective = _smooth_display_curve(mean_objective)
    display_profit = -display_objective
    display_width = _smooth_display_curve(illustrative_width)
    display_penalized_objective = display_objective + display_width
    display_lower_envelope = display_profit - display_width
    profit_solution = _minimize_xgboost_objective(U_GRID, display_objective)
    uncertainty_solution = _minimize_xgboost_objective(
        U_GRID,
        display_penalized_objective,
    )
    return {
        "row_indices": row_indices,
        "u": U_GRID,
        "observed_u": np.asarray(observed_u, dtype=float),
        "baseline_policy_actions": baseline_policy_actions,
        "customer_objective_std": customer_objective_std,
        "sensitivity_u": sensitivity_u,
        "sensitivity_baseline_u": np.asarray(baseline_u),
        "sensitivity_q10": sensitivity["q10"],
        "sensitivity_q25": sensitivity["q25"],
        "sensitivity_median": sensitivity["median"],
        "sensitivity_q75": sensitivity["q75"],
        "sensitivity_q90": sensitivity["q90"],
        "sensitivity_mad": sensitivity["mad"],
        "sensitivity_robust_std": sensitivity["robust_std"],
        "sensitivity_display_robust_std": sensitivity_display_robust_std,
        "sensitivity_minimum_u": np.asarray(dispersion_minimum["u"]),
        "sensitivity_minimum_value": np.asarray(
            dispersion_minimum["minimized_value"]
        ),
        "sensitivity_maximum_u": np.asarray(dispersion_maximum["u"]),
        "sensitivity_maximum_value": np.asarray(
            -float(dispersion_maximum["minimized_value"])
        ),
        "population_size": np.asarray(population_size),
        "marginal_support_ess": marginal_support_ess,
        "marginal_support_relative_ess": marginal_support_relative_ess,
        "marginal_support_relative_risk": marginal_support_relative_risk,
        "marginal_support_band_half_width": marginal_support_band_half_width,
        "exploratory_u": EXPLORATORY_U_GRID,
        "exploratory_mean_profit": -exploratory_mean_objective,
        "exploratory_customer_std": exploratory_customer_std,
        "exploratory_customer_mad": exploratory_customer_mad,
        "exploratory_customer_robust_std": exploratory_customer_robust_std,
        "mean_objective": mean_objective,
        "mean_profit": -mean_objective,
        "median_support": median_support,
        "illustrative_width": illustrative_width,
        "display_objective": display_objective,
        "display_profit": display_profit,
        "display_width": display_width,
        "display_penalized_objective": display_penalized_objective,
        "display_lower_envelope": display_lower_envelope,
        "profit_optimizer_u": np.asarray(profit_solution["u"]),
        "profit_optimizer_objective_value": np.asarray(
            profit_solution["minimized_value"]
        ),
        "profit_optimizer_value": np.asarray(
            profit_solution["plotted_profit_value"]
        ),
        "profit_optimizer_theta": np.asarray(profit_solution["theta"]),
        "profit_optimizer_success": np.asarray(profit_solution["success"]),
        "profit_optimizer_status": np.asarray(profit_solution["status"]),
        "profit_optimizer_nit": np.asarray(profit_solution["nit"]),
        "profit_optimizer_message": np.asarray(profit_solution["message"]),
        "uncertainty_optimizer_u": np.asarray(uncertainty_solution["u"]),
        "uncertainty_optimizer_objective_value": np.asarray(
            uncertainty_solution["minimized_value"]
        ),
        "uncertainty_optimizer_value": np.asarray(
            uncertainty_solution["plotted_profit_value"]
        ),
        "uncertainty_optimizer_theta": np.asarray(uncertainty_solution["theta"]),
        "uncertainty_optimizer_success": np.asarray(
            uncertainty_solution["success"]
        ),
        "uncertainty_optimizer_status": np.asarray(
            uncertainty_solution["status"]
        ),
        "uncertainty_optimizer_nit": np.asarray(uncertainty_solution["nit"]),
        "uncertainty_optimizer_message": np.asarray(
            uncertainty_solution["message"]
        ),
    }


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _plot_clean_objective(data: dict[str, np.ndarray], output_dir: Path) -> None:
    u = data["u"]
    mean_objective = data["display_profit"]
    optimizer_u = float(data["profit_optimizer_u"])
    optimizer_value = float(data["profit_optimizer_value"])
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, mean_objective, linewidth=2.0)
    ax.scatter(
        optimizer_u,
        optimizer_value,
        marker="*",
        s=140,
        color="darkred",
        zorder=3,
    )
    ax.annotate(
        f"Profit-only optimizer: {100 * optimizer_u:.1f}%",
        (optimizer_u, optimizer_value),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )
    ax.set_title("The Profit-Only Optimizer Favors a High Price Increase", fontsize=16)
    ax.set_xlabel("Price change, u", fontsize=12)
    ax.set_ylabel(
        "Mean predicted profit per customer\n(higher is better)",
        fontsize=12,
    )
    ax.tick_params(labelsize=10)
    _save_pdf(fig, output_dir / "01_clean_xgboost_objective.pdf")


def _plot_smoothed_mean_profit(
    data: dict[str, np.ndarray],
    output_dir: Path,
    *,
    show_star: bool,
) -> None:
    u = data["u"]
    smoothed_profit = data["display_profit"]
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, smoothed_profit, linewidth=2.0)
    if show_star:
        ax.scatter(
            float(data["profit_optimizer_u"]),
            float(data["profit_optimizer_value"]),
            marker="*",
            s=140,
            color="darkred",
            zorder=3,
        )
    ax.set_title(
        "Mean Predicted Profit Per Customer vs. Proposed Price Change",
        fontsize=16,
    )
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    suffix = "with_star" if show_star else "without_star"
    _save_pdf(fig, output_dir / f"01_smoothed_mean_profit_{suffix}.pdf")


def _plot_smoothed_mean_profit_mad_band(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    u = data["u"]
    smoothed_profit = data["display_profit"]
    half_width = _mad_cloud_half_width(data)
    lower = smoothed_profit - half_width
    upper = smoothed_profit + half_width

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        alpha=0.2,
        label="±0.6 × customer MAD",
    )
    ax.plot(u, smoothed_profit, linewidth=2.0, label="Mean predicted profit")
    ax.set_title(
        "Mean Predicted Profit with Customer MAD Cloud",
        fontsize=16,
    )
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.legend(fontsize=10)
    _save_pdf(fig, output_dir / "01_smoothed_mean_profit_with_mad_cloud.pdf")


def _plot_full_population_support_weighted_band(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    u = np.asarray(data["u"], dtype=float)
    mean_profit = np.asarray(data["display_profit"], dtype=float)
    half_width = np.asarray(data["marginal_support_band_half_width"], dtype=float)
    population_size = int(data["population_size"])

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(
        u,
        mean_profit - half_width,
        mean_profit + half_width,
        alpha=0.2,
        label="Illustrative support-weighted band",
    )
    ax.plot(u, mean_profit, linewidth=2.0, label="Mean predicted profit")
    ax.set_title(
        f"Full-Cohort Predicted Profit Across {population_size:,} Customers",
        fontsize=16,
    )
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.legend(fontsize=10)
    _save_pdf(
        fig,
        output_dir / "01_full_population_profit_with_support_weighted_band.pdf",
    )


def _export_full_population_support_weighted_band(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    pd.DataFrame(
        {
            "u": data["u"],
            "mean_profit": data["mean_profit"],
            "smoothed_mean_profit": data["display_profit"],
            "marginal_support_effective_sample_size": data[
                "marginal_support_ess"
            ],
            "relative_effective_support": data["marginal_support_relative_ess"],
            "relative_inverse_sqrt_support": data[
                "marginal_support_relative_risk"
            ],
            "illustrative_band_half_width": data[
                "marginal_support_band_half_width"
            ],
        }
    ).to_csv(
        output_dir / "01_full_population_profit_with_support_weighted_band.csv",
        index=False,
    )


def _plot_customer_profit_dispersion_comparison(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    u = data["exploratory_u"]
    smoothed_std = _smooth_display_curve(data["exploratory_customer_std"])
    smoothed_robust_std = _smooth_display_curve(
        data["exploratory_customer_robust_std"]
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, smoothed_std, linewidth=2.0, label="Customer standard deviation")
    ax.plot(
        u,
        smoothed_robust_std,
        linewidth=2.0,
        label="Robust σ (1.4826 × MAD)",
    )
    ax.set_title("Customer Profit Dispersion by Proposed Price Change", fontsize=16)
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Profit Dispersion Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    ax.legend(fontsize=10)
    _save_pdf(
        fig,
        output_dir / "01_customer_profit_dispersion_std_vs_mad.pdf",
    )


def _export_customer_profit_dispersion(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    mean_profit = np.asarray(data["exploratory_mean_profit"], dtype=float)
    standard_deviation = np.asarray(data["exploratory_customer_std"], dtype=float)
    mad = np.asarray(data["exploratory_customer_mad"], dtype=float)
    robust_standard_deviation = np.asarray(
        data["exploratory_customer_robust_std"],
        dtype=float,
    )
    pd.DataFrame(
        {
            "u": data["exploratory_u"],
            "mean_profit": mean_profit,
            "customer_standard_deviation": standard_deviation,
            "customer_mad": mad,
            "customer_robust_standard_deviation": robust_standard_deviation,
            "smoothed_mean_profit": _smooth_display_curve(mean_profit),
            "smoothed_customer_standard_deviation": _smooth_display_curve(
                standard_deviation
            ),
            "smoothed_customer_robust_standard_deviation": _smooth_display_curve(
                robust_standard_deviation
            ),
        }
    ).to_csv(output_dir / "01_customer_profit_dispersion_std_vs_mad.csv", index=False)


def _plot_within_customer_profit_change(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    u = data["sensitivity_u"]
    baseline_u = float(data["sensitivity_baseline_u"])
    minimum_u = float(data["sensitivity_minimum_u"])
    minimum_value = float(data["sensitivity_minimum_value"])
    maximum_u = float(data["sensitivity_maximum_u"])
    maximum_value = float(data["sensitivity_maximum_value"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].fill_between(
        u,
        _smooth_display_curve(data["sensitivity_q10"]),
        _smooth_display_curve(data["sensitivity_q90"]),
        alpha=0.15,
        label="10th–90th percentile",
    )
    axes[0].fill_between(
        u,
        _smooth_display_curve(data["sensitivity_q25"]),
        _smooth_display_curve(data["sensitivity_q75"]),
        alpha=0.3,
        label="25th–75th percentile",
    )
    axes[0].plot(
        u,
        _smooth_display_curve(data["sensitivity_median"]),
        linewidth=2.0,
        label="Median predicted profit change",
    )
    axes[0].axhline(0.0, color="0.5", linewidth=1.0)
    axes[0].axvline(
        baseline_u,
        color="C2",
        linewidth=1.5,
        label=f"Population median historical price ({100 * baseline_u:.1f}%)",
    )
    axes[0].set_title("Paired Predicted Profit Changes", fontsize=14)
    axes[0].set_ylabel("Profit Change Per Customer", fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].plot(u, data["sensitivity_display_robust_std"], linewidth=2.0)
    axes[1].scatter(
        minimum_u,
        minimum_value,
        marker="*",
        s=140,
        color="darkgreen",
        label=f"Smallest dispersion ({100 * minimum_u:.1f}%)",
        zorder=3,
    )
    axes[1].scatter(
        maximum_u,
        maximum_value,
        marker="*",
        s=140,
        color="darkred",
        label=f"Largest dispersion ({100 * maximum_u:.1f}%)",
        zorder=3,
        clip_on=False,
    )
    axes[1].axvline(baseline_u, color="C2", linewidth=1.5)
    axes[1].set_title("Robust Dispersion of Paired Changes", fontsize=14)
    axes[1].set_xlabel("Proposed Price Change", fontsize=12)
    axes[1].set_ylabel("Robust σ (1.4826 × MAD)", fontsize=12)
    axes[1].legend(fontsize=10)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlim(0.0, 0.16)
    fig.suptitle(
        "Within-Customer Profit Sensitivity from the Median Historical Price",
        fontsize=16,
    )
    _save_pdf(
        fig,
        output_dir / "01_within_customer_profit_change_from_median_price.pdf",
    )


def _export_within_customer_profit_change(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    pd.DataFrame(
        {
            "u": data["sensitivity_u"],
            "profit_change_q10": data["sensitivity_q10"],
            "profit_change_q25": data["sensitivity_q25"],
            "profit_change_median": data["sensitivity_median"],
            "profit_change_q75": data["sensitivity_q75"],
            "profit_change_q90": data["sensitivity_q90"],
            "profit_change_mad": data["sensitivity_mad"],
            "profit_change_robust_standard_deviation": data[
                "sensitivity_robust_std"
            ],
            "smoothed_profit_change_robust_standard_deviation": data[
                "sensitivity_display_robust_std"
            ],
        }
    ).to_csv(
        output_dir / "01_within_customer_profit_change_from_median_price.csv",
        index=False,
    )


def _plot_historical_vs_optimized(data: dict[str, np.ndarray], output_dir: Path) -> None:
    historical = data["observed_u"]
    optimized = data["baseline_policy_actions"]
    bins = np.linspace(0.0, 0.16, 33)
    historical_weights = np.full(len(historical), 100.0 / len(historical))
    optimized_weights = np.full(len(optimized), 100.0 / len(optimized))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].hist(historical, bins=bins, weights=historical_weights)
    axes[0].set_title("Historical Price Changes", fontsize=14)
    axes[0].set_ylabel("Customers (%)", fontsize=12)
    axes[1].hist(optimized, bins=bins, weights=optimized_weights)
    axes[1].set_title("Saved Constrained Policy-Optimizer Actions", fontsize=14)
    axes[1].set_xlabel("Price change, u", fontsize=12)
    axes[1].set_ylabel("Customers (%)", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlim(0.0, 0.16)
    fig.suptitle("Historical Actions and the Optimizer Policy", fontsize=16)
    _save_pdf(fig, output_dir / "02_historical_vs_optimized_u.pdf")


def _plot_optimizer_price_change_histogram(output_dir: Path) -> None:
    with np.load(OPTIMIZED_POLICY_PATH, allow_pickle=False) as saved_policy:
        actions = saved_policy["actions"]

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.hist(actions, bins=np.linspace(0.0, 0.16, 33))
    ax.set_title("Distribution of Optimizer Price Changes", fontsize=16)
    ax.set_xlabel("Optimizer Price Change", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.tick_params(labelsize=10)
    _save_pdf(fig, output_dir / "05_optimizer_price_change_histogram.pdf")


def _sample_optimizer_actions(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    sample_rows = np.asarray(data["row_indices"], dtype=int)
    sample_actions = np.asarray(data["baseline_policy_actions"], dtype=float)
    if sample_actions.shape != sample_rows.shape:
        raise ValueError("Saved policy actions must align with diagnostic rows.")
    return sample_rows, sample_actions


def _export_sample_optimizer_actions(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    sample_rows, sample_actions = _sample_optimizer_actions(data)

    pd.DataFrame(
        {
            "sample_position": np.arange(len(sample_rows), dtype=int),
            "csv_row_index": sample_rows,
            "optimizer_price_change": sample_actions,
            "optimizer_price_change_percent": 100.0 * sample_actions,
        }
    ).to_csv(output_dir / "optimizer_price_changes_20k_sample.csv", index=False)


def _plot_sample_optimizer_price_change_histogram(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    _, sample_actions = _sample_optimizer_actions(data)
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.hist(sample_actions, bins=np.linspace(0.0, 0.16, 33))
    ax.set_title(
        "Distribution of Optimizer Price Changes (20,000-Customer Sample)",
        fontsize=16,
    )
    ax.set_xlabel("Optimizer Price Change", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.tick_params(labelsize=10)
    _save_pdf(
        fig,
        output_dir / "05_optimizer_price_change_histogram_20k_sample.pdf",
    )


def _plot_optimizer_shift_to_lower_uncertainty(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    historical = data["observed_u"]
    u = data["u"]
    width = data["display_width"]
    profit_u = float(data["profit_optimizer_u"])
    uncertainty_u = float(data["uncertainty_optimizer_u"])
    bins = np.linspace(0.0, 0.16, 33)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.8), constrained_layout=True)
    axes[0].hist(historical, bins=bins, density=True)
    axes[0].axvline(
        profit_u,
        color="darkred",
        linewidth=2.0,
        label=f"Profit-only optimizer ({100 * profit_u:.1f}%)",
    )
    axes[0].axvline(
        uncertainty_u,
        color="darkgreen",
        linewidth=2.0,
        label=f"Uncertainty-aware optimizer ({100 * uncertainty_u:.1f}%)",
    )
    arrow_y = 0.82 * axes[0].get_ylim()[1]
    axes[0].annotate(
        "shift left",
        xy=(uncertainty_u, arrow_y),
        xytext=(profit_u, arrow_y),
        ha="center",
        va="bottom",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
    )
    axes[0].set_title("Optimizer Shift Against Historical Actions", fontsize=14)
    axes[0].set_xlabel("Price Change", fontsize=12)
    axes[0].set_ylabel("Density (Customers)", fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].plot(u, width, color="C1", linewidth=2.0)
    axes[1].axvline(profit_u, color="darkred", linewidth=2.0)
    axes[1].axvline(uncertainty_u, color="darkgreen", linewidth=2.0)
    axes[1].set_title("Illustrative Uncertainty Width", fontsize=14)
    axes[1].set_xlabel("Price Change", fontsize=12)
    axes[1].set_ylabel("Uncertainty Width (Profit Units)", fontsize=12)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlim(0.0, 0.16)
    fig.suptitle(
        "The Uncertainty Penalty Moves the Optimizer Toward Better Support",
        fontsize=16,
    )
    _save_pdf(
        fig,
        output_dir / "06_optimizer_shift_to_lower_uncertainty.pdf",
    )


def _plot_historical_only(
    data: dict[str, np.ndarray],
    output_dir: Path,
    *,
    show_star: bool = False,
) -> None:
    historical = data["observed_u"]
    bins = np.linspace(0.0, 0.16, 33)
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    density, edges, _ = ax.hist(historical, bins=bins, density=True)
    if show_star:
        raw_u = float(data["profit_optimizer_u"])
        adjusted_u = float(data["uncertainty_optimizer_u"])
        raw_bin = int(np.searchsorted(edges, raw_u, side="right") - 1)
        raw_bin = int(np.clip(raw_bin, 0, len(density) - 1))
        adjusted_bin = int(np.searchsorted(edges, adjusted_u, side="right") - 1)
        adjusted_bin = int(np.clip(adjusted_bin, 0, len(density) - 1))
        ax.scatter(
            raw_u,
            density[raw_bin],
            marker="*",
            s=140,
            color="darkred",
            label=f"Profit-only optimizer ({100 * raw_u:.1f}%)",
            zorder=3,
        )
        ax.scatter(
            adjusted_u,
            density[adjusted_bin],
            marker="*",
            s=140,
            color="darkgreen",
            label=f"Uncertainty-aware optimizer ({100 * adjusted_u:.1f}%)",
            zorder=3,
        )
    ax.set_title("Historical Price Changes", fontsize=16)
    ax.set_xlabel("Historical Price Change", fontsize=12)
    ax.set_ylabel("Density (Customers)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0.0, 0.16)
    if show_star:
        ax.legend(fontsize=10)
    suffix = "_with_star" if show_star else ""
    _save_pdf(fig, output_dir / f"02a_historical_price_changes{suffix}.pdf")


def _plot_historical_with_uncertainty_width(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    historical = data["observed_u"]
    u = data["u"]
    width = data["display_width"]
    bins = np.linspace(0.0, 0.16, 33)
    fig, density_ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    density, edges, _ = density_ax.hist(historical, bins=bins, density=True)

    raw_u = float(data["profit_optimizer_u"])
    adjusted_u = float(data["uncertainty_optimizer_u"])
    raw_bin = int(np.searchsorted(edges, raw_u, side="right") - 1)
    raw_bin = int(np.clip(raw_bin, 0, len(density) - 1))
    adjusted_bin = int(np.searchsorted(edges, adjusted_u, side="right") - 1)
    adjusted_bin = int(np.clip(adjusted_bin, 0, len(density) - 1))
    density_ax.scatter(
        raw_u,
        density[raw_bin],
        marker="*",
        s=140,
        color="darkred",
        label=f"Profit-only optimizer ({100 * raw_u:.1f}%)",
        zorder=4,
    )
    density_ax.scatter(
        adjusted_u,
        density[adjusted_bin],
        marker="*",
        s=140,
        color="darkgreen",
        label=f"Uncertainty-aware optimizer ({100 * adjusted_u:.1f}%)",
        zorder=4,
    )

    width_ax = density_ax.twinx()
    width_ax.fill_between(u, 0.0, width, color="C1", alpha=0.12)
    width_ax.plot(u, width, color="C1", linewidth=2.0)

    density_ax.set_title("Historical Price Changes", fontsize=16)
    density_ax.set_xlabel("Historical Price Change", fontsize=12)
    density_ax.set_ylabel("Density (Customers)", fontsize=12)
    width_ax.set_ylabel("Uncertainty Width", fontsize=12)
    density_ax.tick_params(labelsize=10)
    width_ax.tick_params(labelsize=10)
    density_ax.set_xlim(0.0, 0.16)
    width_ax.set_ylim(bottom=0.0)
    density_ax.legend(fontsize=10)
    _save_pdf(
        fig,
        output_dir / "02b_historical_price_changes_with_uncertainty_width.pdf",
    )


def _plot_envelope(data: dict[str, np.ndarray], output_dir: Path) -> None:
    u = data["u"]
    objective = data["display_profit"]
    lower = data["display_lower_envelope"]
    raw_u = float(data["profit_optimizer_u"])
    raw_value = float(data["profit_optimizer_value"])
    adjusted_u = float(data["uncertainty_optimizer_u"])
    adjusted_value = float(data["uncertainty_optimizer_value"])
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, objective, linewidth=2.0, label="Smoothed mean predicted profit")
    ax.plot(u, lower, linewidth=2.0, label="Uncertainty-adjusted profit")
    ax.fill_between(u, lower, objective, alpha=0.15, label="Coverage-based width")
    ax.scatter(raw_u, raw_value, marker="*", s=140, color="darkred", zorder=3)
    ax.scatter(
        adjusted_u,
        adjusted_value,
        marker="*",
        s=140,
        color="darkgreen",
        zorder=3,
    )
    ax.annotate(
        f"Profit optimizer: {100 * raw_u:.1f}%",
        (raw_u, raw_value),
        xytext=(-112, -24),
        textcoords="offset points",
        fontsize=10,
    )
    ax.annotate(
        f"Uncertainty-aware optimizer: {100 * adjusted_u:.1f}%",
        (adjusted_u, adjusted_value),
        xytext=(-78, -22),
        textcoords="offset points",
        fontsize=10,
    )
    ax.set_title("A Coverage-Aware Envelope Favors the Supported Peak", fontsize=16)
    ax.set_xlabel("Price change, u", fontsize=12)
    ax.set_ylabel(
        "Mean predicted profit per customer\n(higher is better)",
        fontsize=12,
    )
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    _save_pdf(fig, output_dir / "03_illustrative_uncertainty_envelope.pdf")


def _plot_smoothed_envelope(
    data: dict[str, np.ndarray],
    output_dir: Path,
    *,
    show_stars: bool,
) -> None:
    u = data["u"]
    smoothed_profit = data["display_profit"]
    smoothed_lower = data["display_lower_envelope"]

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, smoothed_profit, linewidth=2.0, label="Mean predicted profit")
    ax.plot(
        u,
        smoothed_lower,
        color="C1",
        linewidth=2.0,
        label="Uncertainty adjusted profit",
    )
    ax.fill_between(
        u,
        smoothed_lower,
        smoothed_profit,
        color="C1",
        alpha=0.15,
        label="Uncertainty width",
    )
    if show_stars:
        ax.scatter(
            float(data["profit_optimizer_u"]),
            float(data["profit_optimizer_value"]),
            marker="*",
            s=140,
            color="darkred",
            zorder=3,
        )
        ax.scatter(
            float(data["uncertainty_optimizer_u"]),
            float(data["uncertainty_optimizer_value"]),
            marker="*",
            s=140,
            color="darkgreen",
            zorder=3,
        )
    ax.set_title(
        "Mean Predicted Profit Per Customer vs. Proposed Price Change",
        fontsize=16,
    )
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    suffix = "with_stars" if show_stars else "without_stars"
    _save_pdf(fig, output_dir / f"03_smoothed_uncertainty_envelope_{suffix}.pdf")


def _plot_envelope_construction(data: dict[str, np.ndarray], output_dir: Path) -> None:
    u = data["u"]
    support = data["median_support"]
    width = data["display_width"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(u, support, linewidth=2.0)
    axes[0].set_title("Historical Coverage", fontsize=14)
    axes[0].set_ylabel("Median effective\nneighbor count", fontsize=12)
    axes[1].plot(u, width, linewidth=2.0)
    axes[1].set_title("Illustrative Uncertainty Width", fontsize=14)
    axes[1].set_xlabel("Price change, u", fontsize=12)
    axes[1].set_ylabel("Objective units", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlim(0.0, 0.16)
    fig.suptitle("Lower Coverage Produces a Wider Uncertainty Envelope", fontsize=16)
    _save_pdf(fig, output_dir / "04_envelope_width_from_coverage.pdf")


def _write_experiment_record(
    data: dict[str, np.ndarray],
    output_dir: Path,
    *,
    n_customers: int,
    n_neighbors: int,
    sample_seed: int,
    n_jobs: int,
) -> None:
    """Write the optimizer setup and the two displayed solutions."""
    payload = {
        "purpose": "manufactured visual demonstration of optimizer shift toward lower uncertainty",
        "objective_source": str(FULL_OBJECTIVE_PATH),
        "historical_sample": {
            "n_customers": int(n_customers),
            "seed": int(sample_seed),
        },
        "uncertainty": {
            "interpretation": "illustrative local historical support deficit, not a calibrated confidence interval",
            "n_neighbors": int(n_neighbors),
            "n_jobs": int(n_jobs),
            "action_bandwidth": float(ACTION_BANDWIDTH),
            "width_scale_profit_units": float(ILLUSTRATIVE_WIDTH_SCALE),
        },
        "customer_profit_dispersion": {
            "interpretation": "cross-customer dispersion at each proposed price change; not optimizer uncertainty",
            "domain": [
                float(EXPLORATORY_U_GRID[0]),
                float(EXPLORATORY_U_GRID[-1]),
            ],
            "grid_spacing": float(
                np.round(EXPLORATORY_U_GRID[1] - EXPLORATORY_U_GRID[0], 12)
            ),
            "center": "saved full-population mean predicted profit",
            "robust_scale": "1.4826 * median_i(abs(P_i - median_i(P_i)))",
            "comparison_scale": "sample standard deviation across customers with ddof=1",
            "sample_n_customers": int(n_customers),
            "sample_seed": int(sample_seed),
            "computes_optimum": False,
        },
        "mad_cloud": {
            "output": "01_smoothed_mean_profit_with_mad_cloud.pdf",
            "domain": [float(U_GRID[0]), float(U_GRID[-1])],
            "center": "saved full-population mean predicted profit",
            "half_width": "0.6 * GaussianSmooth(customer MAD)",
            "mad_sample_n_customers": int(n_customers),
            "mad_sample_seed": int(sample_seed),
            "computes_optimum": False,
        },
        "full_population_support_band": {
            "output": "01_full_population_profit_with_support_weighted_band.pdf",
            "population_n_customers": int(data["population_size"]),
            "domain": [float(U_GRID[0]), float(U_GRID[-1])],
            "center": "saved full-population mean predicted profit",
            "support": "Gaussian-kernel marginal ESS over all historical actions",
            "action_bandwidth": ACTION_BANDWIDTH,
            "relative_risk": "sqrt(max(ESS) / ESS(u))",
            "band_scaling": "relative risk mapped to a maximum of 10 profit units without subtracting baseline risk",
            "interpretation": "illustrative extrapolation-risk diagnostic, not a confidence interval",
            "computes_optimum": False,
        },
        "within_customer_profit_change": {
            "interpretation": "paired predicted profit change for each diagnostic customer relative to one common action",
            "baseline_source": "median historical price change across all XGBoost-eligible customers",
            "baseline_u": float(data["sensitivity_baseline_u"]),
            "domain": [float(U_GRID[0]), float(U_GRID[-1])],
            "reported_quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
            "robust_scale": "1.4826 * MAD across paired customer profit changes",
            "extrema_interpolation": "Gaussian-smoothed samples with natural cubic spline off-grid queries",
            "extrema_optimizer": "repository action-space finite-difference minimizer with L-BFGS-B",
            "minimum_start_u_values": [0.04, 0.08, 0.12],
            "maximum_start_u_values": [0.02, 0.08, 0.14],
            "minimum_dispersion_u": float(data["sensitivity_minimum_u"]),
            "minimum_dispersion_value": float(data["sensitivity_minimum_value"]),
            "maximum_dispersion_u": float(data["sensitivity_maximum_u"]),
            "maximum_dispersion_value": float(data["sensitivity_maximum_value"]),
        },
        "representation": {
            "domain": [float(U_GRID[0]), float(U_GRID[-1])],
            "grid_spacing": float(U_GRID[1] - U_GRID[0]),
            "smoothing_sigma_grid_cells": float(GAUSSIAN_SMOOTH_SIGMA),
            "off_grid_rule": "natural cubic-spline interpolation",
            "boundary_rule": "bounded sigmoid policy; action-space probes clip to the closed domain",
            "plotted_grid_role": "rendering and interpolation knots only",
            "plot_sign_convention": "plots show profit = -minimized objective, so higher is better",
        },
        "optimizer": {
            "entry_point": "optimization.solvers.run_finite_difference_minimize",
            "implementation": "optimization.base.Optimization",
            "direction": "minimize",
            "step_rule": "l-bfgs-b",
            "gradient_estimator": "finite_difference",
            "perturbation_space": "u",
            "sigma_u": float(OPTIMIZER_SIGMA_U),
            "n_grad_samples": 1,
            "policy": "bounded sigmoid constant policy",
            "action_bounds": [float(U_GRID[0]), float(U_GRID[-1])],
            "initial_action": float(OPTIMIZER_START_U),
            "initial_theta": 0.0,
            "grad_norm_tol": float(OPTIMIZER_GRAD_NORM_TOL),
            "ftol": float(OPTIMIZER_FTOL),
            "max_steps": int(OPTIMIZER_T_STEPS),
            "random_seed": None,
        },
        "solutions": {
            "profit_only": {
                "u": float(data["profit_optimizer_u"]),
                "minimized_xgboost_objective": float(
                    data["profit_optimizer_objective_value"]
                ),
                "plotted_profit": float(data["profit_optimizer_value"]),
                "theta": float(data["profit_optimizer_theta"]),
                "success": bool(data["profit_optimizer_success"]),
                "status": int(data["profit_optimizer_status"]),
                "nit": int(data["profit_optimizer_nit"]),
                "message": str(data["profit_optimizer_message"]),
            },
            "uncertainty_aware": {
                "u": float(data["uncertainty_optimizer_u"]),
                "minimized_penalized_objective": float(
                    data["uncertainty_optimizer_objective_value"]
                ),
                "plotted_uncertainty_adjusted_profit": float(
                    data["uncertainty_optimizer_value"]
                ),
                "theta": float(data["uncertainty_optimizer_theta"]),
                "success": bool(data["uncertainty_optimizer_success"]),
                "status": int(data["uncertainty_optimizer_status"]),
                "nit": int(data["uncertainty_optimizer_nit"]),
                "message": str(data["uncertainty_optimizer_message"]),
            },
        },
    }
    (output_dir / "optimizer_solutions.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    shift_points = 100.0 * (
        float(data["uncertainty_optimizer_u"]) - float(data["profit_optimizer_u"])
    )
    (output_dir / "EXPERIMENT.md").write_text(
        "\n".join(
            [
                "# Optimizer shift toward lower uncertainty",
                "",
                "This is an intentionally manufactured, plot-forward demonstration. The",
                "uncertainty shape comes from empirical local historical support, while its",
                "scale is fixed at 10 profit units for a visually clear decision consequence.",
                "It is not a calibrated confidence interval.",
                "",
                "Both displayed solutions come from the repository's `Optimization`",
                "pipeline using its action-space central finite-difference estimator and",
                "L-BFGS-B step rule. The bounded sigmoid policy maps one scalar parameter",
                "to `[0, 0.16]`. The optimizer minimizes the smoothed XGBoost cost objective",
                "directly; the plots negate that cost back to profit so higher is better.",
                "The 0.001-spaced samples are interpolation knots and plotting points, not",
                "candidate solutions.",
                "",
                "The dedicated MAD cloud spans `[0, 0.16]` and uses the explicitly",
                "requested half-width `0.6 * customer MAD`. It is a scaled-MAD display",
                "band, not a standard-deviation estimate. A separate wide-domain PDF/CSV",
                "compares Gaussian-consistent `1.4826 * MAD` with ordinary customer",
                "standard deviation. These plots do not calculate or report an optimum.",
                "",
                "The full-population support-band diagnostic uses all 715,023 eligible",
                "historical actions. Gaussian-kernel effective sample size determines",
                "the band shape; full inverse-root support risk is rescaled to a",
                "maximum half-width of 10 profit units without subtracting its",
                "baseline value. It is an illustrative",
                "extrapolation-risk band, not a predictive confidence interval, and it",
                "does not calculate or report an optimum.",
                "",
                "The paired within-customer sensitivity plot holds each diagnostic",
                "customer fixed and subtracts that customer's predicted profit at the",
                "population-median historical price. It reports quantile ribbons and",
                "robust MAD dispersion on `[0, 0.16]`. Its marked dispersion extrema",
                "come from the repository finite-difference optimizer over the smoothed",
                "natural-cubic curve, never from grid selection.",
                "",
                f"- Population-median historical price: `{100 * float(data['sensitivity_baseline_u']):.3f}%`",
                f"- Smallest paired-change dispersion: `{float(data['sensitivity_minimum_value']):.3f}` at `{100 * float(data['sensitivity_minimum_u']):.3f}%`",
                f"- Largest paired-change dispersion: `{float(data['sensitivity_maximum_value']):.3f}` at `{100 * float(data['sensitivity_maximum_u']):.3f}%`",
                "",
                f"- Profit-only optimizer solution: `{100 * float(data['profit_optimizer_u']):.3f}%`",
                f"- Uncertainty-aware optimizer solution: `{100 * float(data['uncertainty_optimizer_u']):.3f}%`",
                f"- Shift: `{shift_points:.3f}` percentage points",
                f"- Historical sample: `{n_customers:,}` customers, seed `{sample_seed}`",
                f"- Local-support neighbors: `{n_neighbors}`",
                f"- Action-kernel bandwidth: `{ACTION_BANDWIDTH}`",
                f"- Finite-difference action step: `{OPTIMIZER_SIGMA_U}`",
                f"- Numerical workers: `{n_jobs}`",
                "",
                "See `optimizer_solutions.json` for the machine-readable optimizer and",
                "representation settings.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-customers", type=int, default=20_000)
    parser.add_argument("--n-neighbors", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _compute_diagnostics(
        n_customers=args.n_customers,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
    )
    np.savez_compressed(args.output_dir / "coverage_diagnostics.npz", **diagnostics)
    _write_experiment_record(
        diagnostics,
        args.output_dir,
        n_customers=len(diagnostics["row_indices"]),
        n_neighbors=args.n_neighbors,
        sample_seed=args.seed,
        n_jobs=args.n_jobs,
    )
    _plot_clean_objective(diagnostics, args.output_dir)
    _plot_smoothed_mean_profit(diagnostics, args.output_dir, show_star=False)
    _plot_smoothed_mean_profit(diagnostics, args.output_dir, show_star=True)
    _plot_smoothed_mean_profit_mad_band(diagnostics, args.output_dir)
    _plot_full_population_support_weighted_band(diagnostics, args.output_dir)
    _export_full_population_support_weighted_band(diagnostics, args.output_dir)
    _plot_customer_profit_dispersion_comparison(diagnostics, args.output_dir)
    _export_customer_profit_dispersion(diagnostics, args.output_dir)
    _plot_within_customer_profit_change(diagnostics, args.output_dir)
    _export_within_customer_profit_change(diagnostics, args.output_dir)
    _plot_historical_vs_optimized(diagnostics, args.output_dir)
    _plot_historical_only(diagnostics, args.output_dir)
    _plot_historical_only(diagnostics, args.output_dir, show_star=True)
    _plot_historical_with_uncertainty_width(diagnostics, args.output_dir)
    _plot_envelope(diagnostics, args.output_dir)
    _plot_smoothed_envelope(diagnostics, args.output_dir, show_stars=False)
    _plot_smoothed_envelope(diagnostics, args.output_dir, show_stars=True)
    _plot_envelope_construction(diagnostics, args.output_dir)
    _plot_optimizer_price_change_histogram(args.output_dir)
    _export_sample_optimizer_actions(diagnostics, args.output_dir)
    _plot_sample_optimizer_price_change_histogram(diagnostics, args.output_dir)
    _plot_optimizer_shift_to_lower_uncertainty(diagnostics, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
