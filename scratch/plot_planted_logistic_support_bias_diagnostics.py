"""Plot diagnostics for a completed planted-logistic support-bias sweep.

Reads an existing ``planted_logistic_support_bias_sweep.csv`` plus saved run
``summary.json`` files. It never reruns optimization; the objective slices are
constant-action planted-logistic curves reconstructed from the saved config and
seeded synthetic batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_NAME = "planted-logistic-support-bias-sweep"
SWEEP_CSV = "planted_logistic_support_bias_sweep.csv"
OUTPUT_DIRNAME = "support_bias_diagnostics"
ESTIMATOR = "first_order"
RESULTS_ROOT_ENV = "GENERALI_RESULTS_ROOT"


@dataclass(frozen=True)
class PlantedLogisticSpec:
    alpha: float
    beta: np.ndarray
    bias: float
    u_star: float
    action_low: float
    action_high: float


@dataclass(frozen=True)
class SweepPoint:
    lambda_bias: float
    support_radius: float
    support_upper: float
    true_gap: float
    mean_action_oracle: float
    mean_action_biased_solution: float
    oracle_run_dir: Path
    biased_run_dir: Path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sweep_dir = _resolve_sweep_dir(args.project_dir)
    output_dir = args.output_dir or sweep_dir / "plots" / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    points = _read_sweep_csv(sweep_dir / SWEEP_CSV)
    if not points:
        raise ValueError(f"No rows found in {sweep_dir / SWEEP_CSV}")

    oracle_summary = _load_json(points[0].oracle_run_dir / "summary.json")
    spec = _planted_logistic_spec(oracle_summary)
    x_train = _training_batch(oracle_summary)
    u_grid = np.linspace(spec.action_low, spec.action_high, int(args.grid_size))

    _plot_true_gap_by_bias_fixed_radius(output_dir / "true_gap_by_bias_fixed_radius.png", points)
    _plot_true_gap_by_radius_fixed_bias(output_dir / "true_gap_by_radius_fixed_bias.png", points)
    _plot_objective_slices(output_dir / "objective_slices_by_radius.png", spec, x_train, points, u_grid)

    print(f"Read sweep: {sweep_dir}")
    print(f"Wrote plots under: {output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=_results_root() / PROJECT_NAME,
        help="Project directory or concrete support_bias_sweep_* directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <sweep-dir>/plots/{OUTPUT_DIRNAME}.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=401,
        help="Number of constant-action grid points for objective-slice plots.",
    )
    return parser.parse_args(argv)


def _results_root() -> Path:
    override = os.environ.get(RESULTS_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "projects" / "generali-pricing" / "results").resolve()


def _resolve_sweep_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / SWEEP_CSV).exists():
        return path
    candidates = sorted(
        child for child in path.glob("support_bias_sweep_*") if child.is_dir() and (child / SWEEP_CSV).exists()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find {SWEEP_CSV} under {path}")
    return candidates[-1]


def _read_sweep_csv(path: Path) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append(
                SweepPoint(
                    lambda_bias=float(row["lambda_bias"]),
                    support_radius=float(row["support_radius"]),
                    support_upper=float(row["support_upper"]),
                    true_gap=float(row["true_gap"]),
                    mean_action_oracle=float(row["mean_action_oracle"]),
                    mean_action_biased_solution=float(row["mean_action_biased_solution"]),
                    oracle_run_dir=Path(row["oracle_run_dir"]),
                    biased_run_dir=Path(row["biased_run_dir"]),
                )
            )
    return sorted(points, key=lambda item: (item.support_radius, item.lambda_bias))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _planted_logistic_spec(summary: Mapping[str, Any]) -> PlantedLogisticSpec:
    objective = summary["config"]["objective"]
    if objective.get("type") == "BiasedObjective":
        objective = objective["base_objective"]
    if objective.get("type") != "PlantedLogisticObjective":
        raise ValueError(f"Expected PlantedLogisticObjective, found {objective.get('type')!r}")
    policy = objective["policy"]
    if policy.get("type") != "SoftmaxPolicy":
        raise ValueError(f"Expected SoftmaxPolicy, found {policy.get('type')!r}")
    return PlantedLogisticSpec(
        alpha=float(objective["alpha"]),
        beta=np.asarray(objective["beta"], dtype=float),
        bias=float(objective["bias"]),
        u_star=float(objective["u_star"]),
        action_low=float(policy["action_low"]),
        action_high=float(policy["action_high"]),
    )


def _training_batch(summary: Mapping[str, Any]) -> np.ndarray:
    config = summary["config"]
    if config.get("x_fixed_shape") is not None:
        raise ValueError("This diagnostic supports synthetic planted-logistic batches only.")
    n_samples = int(config["n_samples"])
    state_dim = int(config["state_dim"])
    seeds = config["resolved_seed_setup"]
    rng = np.random.default_rng(int(seeds["data_seed"]))
    x_all = rng.normal(0.0, 1.0, size=(n_samples, state_dim)).astype(float)
    test_fraction = float(config.get("test_fraction", 0.0))
    if test_fraction == 0.0:
        return x_all
    split_rng = np.random.default_rng(int(seeds["split_seed"]))
    shuffled = split_rng.permutation(n_samples).astype(int)
    n_test = int(round(test_fraction * n_samples))
    n_test = min(max(n_test, 1), n_samples - 1)
    return x_all[shuffled[n_test:]]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


def _true_mean_at_u(spec: PlantedLogisticSpec, x_batch: np.ndarray, u: np.ndarray) -> np.ndarray:
    beta_x = x_batch @ spec.beta[: x_batch.shape[1]]
    z_star = spec.alpha * spec.u_star + beta_x + spec.bias
    p_star = _sigmoid(z_star)
    values = []
    for u_val in np.asarray(u, dtype=float):
        z = spec.alpha * float(u_val) + beta_x + spec.bias
        values.append(float(np.mean(np.logaddexp(0.0, z) - p_star * z)))
    return np.asarray(values, dtype=float)


def _surrogate_mean_at_u(
    spec: PlantedLogisticSpec,
    x_batch: np.ndarray,
    point: SweepPoint,
    u_grid: np.ndarray,
) -> np.ndarray:
    true_curve = _true_mean_at_u(spec, x_batch, u_grid)
    support_excess = np.maximum(0.0, np.asarray(u_grid, dtype=float) - point.support_upper)
    return true_curve - point.lambda_bias * support_excess


def _plot_true_gap_by_bias_fixed_radius(path: Path, points: Sequence[SweepPoint]) -> None:
    radii = sorted({point.support_radius for point in points})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharey=True)
    y_max = max(point.true_gap for point in points) * 1.15
    for ax, radius in zip(axes.ravel(), radii):
        subset = [point for point in points if np.isclose(point.support_radius, radius)]
        lambdas = [point.lambda_bias for point in subset]
        gaps = [point.true_gap for point in subset]
        ax.bar([_label(value) for value in lambdas], gaps, color="#4c78a8")
        ax.set_title(rf"Fixed support radius $r={radius:g}$")
        ax.set_xlabel(r"Bias strength $\lambda_{bias}$")
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"True gap $J_\star(\hat\theta_{\lambda,r})-J_\star(\hat\theta_{oracle})$")
    fig.suptitle("True objective degradation as bias increases at fixed support radius")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_true_gap_by_radius_fixed_bias(path: Path, points: Sequence[SweepPoint]) -> None:
    lambdas = sorted({point.lambda_bias for point in points})
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharey=True)
    y_max = max(point.true_gap for point in points) * 1.15
    for ax, lambda_bias in zip(axes.ravel(), lambdas):
        subset = [point for point in points if np.isclose(point.lambda_bias, lambda_bias)]
        radii = [point.support_radius for point in subset]
        gaps = [point.true_gap for point in subset]
        ax.bar([_label(value) for value in radii], gaps, color="#f58518")
        ax.set_title(rf"Fixed bias $\lambda_{{bias}}={lambda_bias:g}$")
        ax.set_xlabel(r"Support radius $r$")
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"True gap $J_\star(\hat\theta_{\lambda,r})-J_\star(\hat\theta_{oracle})$")
    fig.suptitle("True objective degradation as support radius increases at fixed bias")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_objective_slices(
    path: Path,
    spec: PlantedLogisticSpec,
    x_train: np.ndarray,
    points: Sequence[SweepPoint],
    u_grid: np.ndarray,
) -> None:
    radii = sorted({point.support_radius for point in points})
    nonzero_lambdas = sorted({point.lambda_bias for point in points if point.lambda_bias > 0.0})
    colors = dict(zip(nonzero_lambdas, plt.cm.viridis(np.linspace(0.15, 0.9, len(nonzero_lambdas)))))
    true_curve = _true_mean_at_u(spec, x_train, u_grid)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True, sharey=True)
    for ax, radius in zip(axes.ravel(), radii):
        subset = [point for point in points if np.isclose(point.support_radius, radius)]
        ax.plot(u_grid, true_curve, color="black", linewidth=2.2, label=r"oracle true $\bar M_\star(u)$")
        support_upper = subset[0].support_upper
        ax.axvline(
            support_upper,
            color="#d62728",
            linestyle=":",
            linewidth=1.3,
            label=r"support boundary $h=u^\star+r$",
        )
        for point in subset:
            if point.lambda_bias == 0.0:
                continue
            color = colors[point.lambda_bias]
            surrogate_curve = _surrogate_mean_at_u(spec, x_train, point, u_grid)
            ax.plot(
                u_grid,
                surrogate_curve,
                color=color,
                linewidth=1.7,
                label=rf"$\hat{{M}}$, $\lambda={point.lambda_bias:g}$",
            )
            marker_y = np.interp(point.mean_action_biased_solution, u_grid, surrogate_curve)
            ax.scatter(
                [point.mean_action_biased_solution],
                [marker_y],
                color=color,
                s=26,
                zorder=3,
            )
        ax.axvline(spec.u_star, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=rf"$u^\star={spec.u_star:g}$")
        ax.set_title(rf"Support radius $r={radius:g}$")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Constant action $u$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Mean objective value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=8)
    fig.suptitle("Oracle true objective versus upper-support biased surrogates")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _label(value: float) -> str:
    return f"{float(value):g}"


if __name__ == "__main__":
    main()
