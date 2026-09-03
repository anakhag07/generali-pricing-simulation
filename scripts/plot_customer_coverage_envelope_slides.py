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
calibrated confidence interval. Every displayed solution is returned by a
continuous optimizer; sampled action grids are used only to construct and render
the optimizer-facing natural cubic splines.
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
from scipy.optimize import minimize_scalar
from sklearn.neighbors import NearestNeighbors

from data.loader import (
    eligible_csv_row_indices,
    load_model_artifacts,
    load_observed_u_array,
    load_x_frame,
)


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
NUMERIC_CLIP = 6.0
ACTION_BANDWIDTH = 0.01
ILLUSTRATIVE_WIDTH_SCALE = 10.0
GAUSSIAN_SMOOTH_SIGMA = 1.25
GAUSSIAN_SMOOTH_TRUNCATE = 4.0
OPTIMIZER_XATOL = 1e-10
OPTIMIZER_MAXITER = 1_000


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


def _optimize_display_curve(
    u_grid: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int | bool | str]:
    """Maximize a continuous natural cubic spline with SciPy's bounded optimizer."""
    u = np.asarray(u_grid, dtype=float)
    y = np.asarray(values, dtype=float)
    if u.ndim != 1 or y.shape != u.shape or len(u) < 2:
        raise ValueError("u_grid and values must be matching one-dimensional arrays.")
    if not np.isfinite(u).all() or not np.isfinite(y).all():
        raise ValueError("u_grid and values must be finite.")
    if not np.all(np.diff(u) > 0.0):
        raise ValueError("u_grid must be strictly increasing.")

    interpolant = CubicSpline(u, y, bc_type="natural", extrapolate=False)
    result = minimize_scalar(
        lambda action: -float(interpolant(action)),
        bounds=(float(u[0]), float(u[-1])),
        method="bounded",
        options={"xatol": OPTIMIZER_XATOL, "maxiter": OPTIMIZER_MAXITER},
    )
    if not result.success:
        raise RuntimeError(f"Bounded curve optimization failed: {result.message}")
    return {
        "u": float(result.x),
        "value": float(interpolant(result.x)),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "message": str(result.message),
    }


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
    rng = np.random.default_rng(seed)
    row_indices = np.sort(
        rng.choice(eligible, size=min(int(n_customers), len(eligible)), replace=False)
    )
    frame = load_x_frame("xgb", row_indices=row_indices)
    observed_u = load_observed_u_array("xgb", row_indices=row_indices)
    acceptance_artifact, loss_artifact = load_model_artifacts("xgb")
    acceptance_artifact.model.set_params(n_jobs=int(n_jobs))
    loss_artifact.model.set_params(n_jobs=int(n_jobs))

    acceptance = _predict_acceptance_matrix(acceptance_artifact, frame, U_GRID)
    loss = _predict_loss(loss_artifact, frame)
    premium = frame["X_policy_premium"].to_numpy(dtype=float)
    revenue = premium[:, None] * (1.0 + U_GRID[None, :])
    customer_profit = acceptance * (revenue - loss[:, None])
    customer_objective_std = np.std(customer_profit, axis=0, ddof=1)
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
    full_curve = full_curve.loc[full_curve["u"].between(0.0, 0.16)].copy()
    if not np.allclose(full_curve["u"].to_numpy(dtype=float), U_GRID):
        raise ValueError("The saved full-dataset objective grid does not match U_GRID.")
    mean_profit = -full_curve["mean_objective"].to_numpy(dtype=float)

    support_deficit = 1.0 - median_support / float(np.max(median_support))
    illustrative_width = ILLUSTRATIVE_WIDTH_SCALE * support_deficit
    display_profit = _smooth_display_curve(mean_profit)
    display_width = _smooth_display_curve(illustrative_width)
    display_lower_envelope = display_profit - display_width
    profit_solution = _optimize_display_curve(U_GRID, display_profit)
    uncertainty_solution = _optimize_display_curve(U_GRID, display_lower_envelope)
    return {
        "row_indices": row_indices,
        "u": U_GRID,
        "observed_u": np.asarray(observed_u, dtype=float),
        "baseline_policy_actions": baseline_policy_actions,
        "customer_objective_std": customer_objective_std,
        "mean_profit": mean_profit,
        "median_support": median_support,
        "illustrative_width": illustrative_width,
        "display_profit": display_profit,
        "display_width": display_width,
        "display_lower_envelope": display_lower_envelope,
        "profit_optimizer_u": np.asarray(profit_solution["u"]),
        "profit_optimizer_value": np.asarray(profit_solution["value"]),
        "profit_optimizer_nfev": np.asarray(profit_solution["nfev"]),
        "profit_optimizer_nit": np.asarray(profit_solution["nit"]),
        "uncertainty_optimizer_u": np.asarray(uncertainty_solution["u"]),
        "uncertainty_optimizer_value": np.asarray(uncertainty_solution["value"]),
        "uncertainty_optimizer_nfev": np.asarray(uncertainty_solution["nfev"]),
        "uncertainty_optimizer_nit": np.asarray(uncertainty_solution["nit"]),
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
        "Mean predicted objective value per customer\n(higher is better)",
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


def _plot_smoothed_mean_profit_std_band(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    u = data["u"]
    smoothed_profit = data["display_profit"]
    smoothed_std = _smooth_display_curve(data["customer_objective_std"])
    lower = smoothed_profit - smoothed_std
    upper = smoothed_profit + smoothed_std

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        alpha=0.2,
        label="±1 customer standard deviation",
    )
    ax.plot(u, smoothed_profit, linewidth=2.0, label="Mean predicted profit")
    ax.set_title(
        "Mean Predicted Profit Per Customer vs. Proposed Price Change",
        fontsize=16,
    )
    ax.set_xlabel("Proposed Price Change", fontsize=12)
    ax.set_ylabel("Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    _save_pdf(fig, output_dir / "01_smoothed_mean_profit_with_1std_cloud.pdf")


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
        "Mean predicted objective value per customer\n(higher is better)",
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
        "representation": {
            "domain": [float(U_GRID[0]), float(U_GRID[-1])],
            "grid_spacing": float(U_GRID[1] - U_GRID[0]),
            "smoothing_sigma_grid_cells": float(GAUSSIAN_SMOOTH_SIGMA),
            "off_grid_rule": "natural cubic-spline interpolation",
            "plotted_grid_role": "rendering and interpolation knots only",
        },
        "optimizer": {
            "library": "scipy.optimize.minimize_scalar",
            "method": "bounded",
            "bounds": [float(U_GRID[0]), float(U_GRID[-1])],
            "xatol": float(OPTIMIZER_XATOL),
            "maxiter": int(OPTIMIZER_MAXITER),
            "random_seed": None,
        },
        "solutions": {
            "profit_only": {
                "u": float(data["profit_optimizer_u"]),
                "value": float(data["profit_optimizer_value"]),
                "success": True,
                "nfev": int(data["profit_optimizer_nfev"]),
                "nit": int(data["profit_optimizer_nit"]),
            },
            "uncertainty_aware": {
                "u": float(data["uncertainty_optimizer_u"]),
                "value": float(data["uncertainty_optimizer_value"]),
                "success": True,
                "nfev": int(data["uncertainty_optimizer_nfev"]),
                "nit": int(data["uncertainty_optimizer_nit"]),
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
                "Both displayed solutions come from SciPy's deterministic bounded scalar",
                "optimizer over continuous natural cubic splines on `[0, 0.16]`. The",
                "0.001-spaced samples are interpolation knots and plotting points, not",
                "candidate solutions.",
                "",
                f"- Profit-only optimizer solution: `{100 * float(data['profit_optimizer_u']):.3f}%`",
                f"- Uncertainty-aware optimizer solution: `{100 * float(data['uncertainty_optimizer_u']):.3f}%`",
                f"- Shift: `{shift_points:.3f}` percentage points",
                f"- Historical sample: `{n_customers:,}` customers, seed `{sample_seed}`",
                f"- Local-support neighbors: `{n_neighbors}`",
                f"- Action-kernel bandwidth: `{ACTION_BANDWIDTH}`",
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
    _plot_smoothed_mean_profit_std_band(diagnostics, args.output_dir)
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
