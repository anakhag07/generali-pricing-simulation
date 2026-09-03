"""Build slide-ready diagnostics for a fixed-fold XGBoost pricing objective.

The fold-0 XGBoost artifacts define the point objective throughout. Customer
coverage is estimated independently from historical neighbors:

* numeric customer features use the fold-0 covariance-whitened coordinates;
* categorical customer features use exact-match one-hot coordinates;
* historical price change U is excluded from customer distance and enters only
  through a separate Gaussian action kernel.

The uncertainty envelope is intentionally illustrative. Its shape is determined
by empirical local coverage, while its scale is fixed at 10 objective units so
that the decision consequence is visible in a slide. It is not presented as a
calibrated confidence interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
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
) -> dict[str, np.ndarray]:
    eligible = eligible_csv_row_indices("xgb")
    rng = np.random.default_rng(seed)
    row_indices = np.sort(
        rng.choice(eligible, size=min(int(n_customers), len(eligible)), replace=False)
    )
    frame = load_x_frame("xgb", row_indices=row_indices)
    observed_u = load_observed_u_array("xgb", row_indices=row_indices)
    acceptance_artifact, loss_artifact = load_model_artifacts("xgb")
    acceptance_artifact.model.set_params(n_jobs=-1)
    loss_artifact.model.set_params(n_jobs=-1)

    acceptance = _predict_acceptance_matrix(acceptance_artifact, frame, U_GRID)
    loss = _predict_loss(loss_artifact, frame)
    premium = frame["X_policy_premium"].to_numpy(dtype=float)
    revenue = premium[:, None] * (1.0 + U_GRID[None, :])
    customer_profit = acceptance * (revenue - loss[:, None])
    optimized_u = U_GRID[np.argmax(customer_profit, axis=1)]
    customer_objective_std = np.std(customer_profit, axis=0, ddof=1)

    embedding = _mixed_customer_embedding(acceptance_artifact, frame)
    support = _local_support_matrix(
        embedding,
        observed_u,
        U_GRID,
        n_neighbors=n_neighbors,
    )
    median_support = np.median(support, axis=0)

    full_curve = pd.read_csv(FULL_OBJECTIVE_PATH)
    full_curve = full_curve.loc[full_curve["u"].between(0.0, 0.16)].copy()
    if not np.allclose(full_curve["u"].to_numpy(dtype=float), U_GRID):
        raise ValueError("The saved full-dataset objective grid does not match U_GRID.")
    mean_profit = -full_curve["mean_objective"].to_numpy(dtype=float)

    support_deficit = 1.0 - median_support / float(np.max(median_support))
    illustrative_width = ILLUSTRATIVE_WIDTH_SCALE * support_deficit
    lower_envelope = mean_profit - illustrative_width
    return {
        "row_indices": row_indices,
        "u": U_GRID,
        "observed_u": np.asarray(observed_u, dtype=float),
        "optimized_u": optimized_u,
        "customer_objective_std": customer_objective_std,
        "mean_profit": mean_profit,
        "median_support": median_support,
        "illustrative_width": illustrative_width,
        "lower_envelope": lower_envelope,
    }


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _plot_clean_objective(data: dict[str, np.ndarray], output_dir: Path) -> None:
    u = data["u"]
    mean_objective = data["mean_profit"]
    optimum = int(np.argmax(mean_objective))
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, mean_objective, linewidth=2.0)
    ax.scatter(u[optimum], mean_objective[optimum], s=45, zorder=3)
    ax.annotate(
        f"Mean optimum: {100 * u[optimum]:.1f}%",
        (u[optimum], mean_objective[optimum]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )
    ax.set_title("The Fixed XGBoost Objective Favors a High Price Increase", fontsize=16)
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
    mean_profit = data["mean_profit"]
    smoothed_profit = gaussian_filter1d(
        mean_profit,
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, smoothed_profit, linewidth=2.0)
    if show_star:
        optimum = int(np.argmax(smoothed_profit))
        ax.scatter(
            u[optimum],
            smoothed_profit[optimum],
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
    smoothed_profit = gaussian_filter1d(
        data["mean_profit"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
    smoothed_std = gaussian_filter1d(
        data["customer_objective_std"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
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
    optimized = data["optimized_u"]
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
    axes[1].set_title("Fold-1 Optimized Price Changes", fontsize=14)
    axes[1].set_xlabel("Price change, u", fontsize=12)
    axes[1].set_ylabel("Customers (%)", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlim(0.0, 0.16)
    fig.suptitle("Historical and Optimized Actions Are Different Distributions", fontsize=16)
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
    with np.load(OPTIMIZED_POLICY_PATH, allow_pickle=False) as saved_policy:
        policy_rows = saved_policy["row_indices"]
        policy_actions = saved_policy["actions"]

    positions = np.searchsorted(policy_rows, sample_rows)
    if np.any(positions >= len(policy_rows)) or not np.array_equal(
        policy_rows[positions], sample_rows
    ):
        raise ValueError("The diagnostic sample is not contained in the saved policy rows.")
    return sample_rows, policy_actions[positions]


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


def _plot_optimizer_vs_coverage_envelope_histograms(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    _, optimizer_actions = _sample_optimizer_actions(data)
    smoothed_lower = gaussian_filter1d(
        data["lower_envelope"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
    coverage_action = float(data["u"][np.argmax(smoothed_lower)])
    coverage_actions = np.full(len(optimizer_actions), coverage_action)
    bins = np.linspace(0.0, 0.16, 33)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.8), constrained_layout=True)
    axes[0].hist(optimizer_actions, bins=bins)
    axes[0].set_title("Customer-Specific Optimizer", fontsize=14)
    axes[0].set_xlabel("Price Change", fontsize=12)
    axes[0].set_ylabel("Number of Customers", fontsize=12)

    axes[1].hist(coverage_actions, bins=bins)
    axes[1].set_title(
        f"Coverage Lower-Envelope Rule ({100 * coverage_action:.1f}%)",
        fontsize=14,
    )
    axes[1].set_xlabel("Price Change", fontsize=12)
    axes[1].set_ylabel("Number of Customers", fontsize=12)

    for ax in axes:
        ax.tick_params(labelsize=10)
    fig.suptitle(
        "Optimizer Actions vs. Coverage-Based Lower-Envelope Rule",
        fontsize=16,
    )
    _save_pdf(
        fig,
        output_dir / "06_optimizer_vs_coverage_lower_envelope_histograms.pdf",
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
        raw_u = 0.13
        adjusted_curve = gaussian_filter1d(
            data["lower_envelope"],
            sigma=GAUSSIAN_SMOOTH_SIGMA,
            mode="nearest",
            truncate=GAUSSIAN_SMOOTH_TRUNCATE,
        )
        adjusted_u = float(data["u"][np.argmax(adjusted_curve)])
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
            zorder=3,
        )
        ax.scatter(
            adjusted_u,
            density[adjusted_bin],
            marker="*",
            s=140,
            color="darkgreen",
            zorder=3,
        )
    ax.set_title("Historical Price Changes", fontsize=16)
    ax.set_xlabel("Historical Price Change", fontsize=12)
    ax.set_ylabel("Density (Customers)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0.0, 0.16)
    suffix = "_with_star" if show_star else ""
    _save_pdf(fig, output_dir / f"02a_historical_price_changes{suffix}.pdf")


def _plot_historical_with_uncertainty_width(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    historical = data["observed_u"]
    u = data["u"]
    width = data["illustrative_width"]
    bins = np.linspace(0.0, 0.16, 33)
    fig, density_ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    density, edges, _ = density_ax.hist(historical, bins=bins, density=True)

    raw_u = 0.13
    adjusted_curve = gaussian_filter1d(
        data["lower_envelope"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
    adjusted_u = float(u[np.argmax(adjusted_curve)])
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
        zorder=4,
    )
    density_ax.scatter(
        adjusted_u,
        density[adjusted_bin],
        marker="*",
        s=140,
        color="darkgreen",
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
    _save_pdf(
        fig,
        output_dir / "02b_historical_price_changes_with_uncertainty_width.pdf",
    )


def _plot_envelope(data: dict[str, np.ndarray], output_dir: Path) -> None:
    u = data["u"]
    objective = data["mean_profit"]
    lower = data["lower_envelope"]
    raw_optimum = int(np.argmax(objective))
    safe_optimum = int(np.argmax(lower))
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.plot(u, objective, linewidth=2.0, label="Fixed XGBoost objective")
    ax.plot(u, lower, linewidth=2.0, label="Illustrative lower envelope")
    ax.fill_between(u, lower, objective, alpha=0.15, label="Coverage-based width")
    ax.scatter(u[raw_optimum], objective[raw_optimum], s=45, zorder=3)
    ax.scatter(u[safe_optimum], lower[safe_optimum], s=45, zorder=3)
    ax.annotate(
        f"Raw: {100 * u[raw_optimum]:.1f}%",
        (u[raw_optimum], objective[raw_optimum]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )
    ax.annotate(
        f"Envelope: {100 * u[safe_optimum]:.1f}%",
        (u[safe_optimum], lower[safe_optimum]),
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
    smoothed_profit = gaussian_filter1d(
        data["mean_profit"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )
    smoothed_lower = gaussian_filter1d(
        data["lower_envelope"],
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )

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
        profit_optimum = int(np.argmax(smoothed_profit))
        lower_optimum = int(np.argmax(smoothed_lower))
        ax.scatter(
            u[profit_optimum],
            smoothed_profit[profit_optimum],
            marker="*",
            s=140,
            color="darkred",
            zorder=3,
        )
        ax.scatter(
            u[lower_optimum],
            smoothed_lower[lower_optimum],
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
    width = data["illustrative_width"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-customers", type=int, default=20_000)
    parser.add_argument("--n-neighbors", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _compute_diagnostics(
        n_customers=args.n_customers,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
    )
    np.savez_compressed(args.output_dir / "coverage_diagnostics.npz", **diagnostics)
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
    _plot_optimizer_vs_coverage_envelope_histograms(diagnostics, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
