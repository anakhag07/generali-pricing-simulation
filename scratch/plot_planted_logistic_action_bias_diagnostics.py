"""Plot diagnostics for a completed planted-logistic action-bias sweep.

This reads an existing ``planted_logistic_action_bias_sweep.csv`` plus per-run
``summary.json`` and ``plots/optimization/steps.csv`` files. It reconstructs the
synthetic training batch from saved seeds and evaluates the planted-logistic
formulas directly, so no optimization is rerun and no JAX modules are imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_NAME = "planted-logistic-action-bias-sweep"
SWEEP_CSV = "planted_logistic_action_bias_sweep.csv"
OUTPUT_DIRNAME = "action_bias_diagnostics"
DIAGNOSTIC_CSV = "action_bias_diagnostics.csv"
ESTIMATOR = "first_order"
RESULTS_ROOT_ENV = "GENERALI_RESULTS_ROOT"
THETA_LABELS = (
    r"$\theta_0$ (intercept)",
    r"$\theta_1$ ($x_1$)",
    r"$\theta_2$ ($x_2$)",
    r"$\theta_3$ ($x_3$)",
)


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
    true_objective_at_oracle: float
    true_objective_at_biased_solution: float
    true_gap: float
    mean_action_oracle: float
    mean_action_biased_solution: float
    surrogate_objective_at_biased_solution: float
    optimism_gap: float
    oracle_run_dir: Path
    biased_run_dir: Path
    theta: np.ndarray
    theta_delta_from_oracle: np.ndarray
    theta_l2_from_oracle: float
    optimizer_success: bool | None
    optimizer_status: int | None
    optimizer_message: str

    def csv_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "lambda_bias": self.lambda_bias,
            "true_objective_at_oracle": self.true_objective_at_oracle,
            "true_objective_at_biased_solution": self.true_objective_at_biased_solution,
            "true_gap": self.true_gap,
            "mean_action_oracle": self.mean_action_oracle,
            "mean_action_biased_solution": self.mean_action_biased_solution,
            "surrogate_objective_at_biased_solution": self.surrogate_objective_at_biased_solution,
            "optimism_gap": self.optimism_gap,
            "theta_l2_from_oracle": self.theta_l2_from_oracle,
            "optimizer_success": "" if self.optimizer_success is None else self.optimizer_success,
            "optimizer_status": "" if self.optimizer_status is None else self.optimizer_status,
            "optimizer_message": self.optimizer_message,
            "oracle_run_dir": str(self.oracle_run_dir),
            "biased_run_dir": str(self.biased_run_dir),
        }
        for idx, value in enumerate(self.theta_delta_from_oracle):
            row[f"theta_{idx}_delta_from_oracle"] = float(value)
        return row


@dataclass(frozen=True)
class StepTrace:
    lambda_bias: float | None
    label: str
    steps: np.ndarray
    mean_u: np.ndarray
    objective_value: np.ndarray


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sweep_dir = _resolve_sweep_dir(args.project_dir)
    output_dir = args.output_dir or sweep_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_sweep_csv(sweep_dir / SWEEP_CSV)
    if not rows:
        raise ValueError(f"No rows found in {sweep_dir / SWEEP_CSV}")
    oracle_summary = _load_json(rows[0].oracle_run_dir / "summary.json")
    spec = _planted_logistic_spec(oracle_summary)
    x_train = _training_batch(oracle_summary)
    oracle_theta = _theta(oracle_summary)
    points = _enrich_points(rows, oracle_theta)
    u_grid = _action_grid(spec, points, int(args.grid_size))

    _write_rows(output_dir / DIAGNOSTIC_CSV, [point.csv_row() for point in points], _diagnostic_fieldnames(points))
    _plot_objective_slices(output_dir / "objective_slices.png", spec, x_train, points, u_grid)
    _plot_surrogate_minus_true(output_dir / "surrogate_minus_true.png", points, u_grid)
    _plot_sweep_metrics(output_dir / "sweep_metrics_by_lambda.png", points)
    _plot_theta_drift(output_dir / "theta_drift_by_lambda.png", points)
    traces = _step_traces(points, oracle_run_dir=rows[0].oracle_run_dir)
    if traces:
        _plot_optimization_traces(output_dir / "optimization_traces.png", traces)

    print(f"Read sweep: {sweep_dir}")
    print(f"Wrote diagnostics CSV: {output_dir / DIAGNOSTIC_CSV}")
    print(f"Wrote plots under: {output_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=_results_root() / PROJECT_NAME,
        help="Project directory or concrete action_bias_sweep_* directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <sweep-dir>/{OUTPUT_DIRNAME}.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=301,
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
        child for child in path.glob("action_bias_sweep_*") if child.is_dir() and (child / SWEEP_CSV).exists()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find {SWEEP_CSV} under {path}")
    return candidates[-1]


def _read_sweep_csv(path: Path) -> list[SweepPoint]:
    rows: list[SweepPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                SweepPoint(
                    lambda_bias=float(row["lambda_bias"]),
                    true_objective_at_oracle=float(row["true_objective_at_oracle"]),
                    true_objective_at_biased_solution=float(row["true_objective_at_biased_solution"]),
                    true_gap=float(row["true_gap"]),
                    mean_action_oracle=float(row["mean_action_oracle"]),
                    mean_action_biased_solution=float(row["mean_action_biased_solution"]),
                    surrogate_objective_at_biased_solution=float(row["surrogate_objective_at_biased_solution"]),
                    optimism_gap=float(row["optimism_gap"]),
                    oracle_run_dir=Path(row["oracle_run_dir"]),
                    biased_run_dir=Path(row["biased_run_dir"]),
                    theta=np.asarray([], dtype=float),
                    theta_delta_from_oracle=np.asarray([], dtype=float),
                    theta_l2_from_oracle=float("nan"),
                    optimizer_success=None,
                    optimizer_status=None,
                    optimizer_message="",
                )
            )
    return sorted(rows, key=lambda item: item.lambda_bias)


def _enrich_points(rows: Sequence[SweepPoint], oracle_theta: np.ndarray) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for row in rows:
        summary = _load_json(row.biased_run_dir / "summary.json")
        theta = _theta(summary)
        theta_delta = theta - oracle_theta
        estimator = summary["estimators"][ESTIMATOR]
        status = estimator.get("optimizer_status")
        success = estimator.get("optimizer_success")
        points.append(
            SweepPoint(
                lambda_bias=row.lambda_bias,
                true_objective_at_oracle=row.true_objective_at_oracle,
                true_objective_at_biased_solution=row.true_objective_at_biased_solution,
                true_gap=row.true_gap,
                mean_action_oracle=row.mean_action_oracle,
                mean_action_biased_solution=row.mean_action_biased_solution,
                surrogate_objective_at_biased_solution=row.surrogate_objective_at_biased_solution,
                optimism_gap=row.optimism_gap,
                oracle_run_dir=row.oracle_run_dir,
                biased_run_dir=row.biased_run_dir,
                theta=theta,
                theta_delta_from_oracle=theta_delta,
                theta_l2_from_oracle=float(np.linalg.norm(theta_delta)),
                optimizer_success=None if success is None else bool(success),
                optimizer_status=None if status is None else int(status),
                optimizer_message=str(estimator.get("optimizer_message", "")),
            )
        )
    return points


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _theta(summary: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(summary["estimators"][ESTIMATOR]["theta"], dtype=float)


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
        raise ValueError("This diagnostic currently supports synthetic planted-logistic batches only.")
    n_samples = int(config["n_samples"])
    state_dim = int(config["state_dim"])
    seeds = config["resolved_seed_setup"]
    rng = np.random.default_rng(int(seeds["data_seed"]))
    x_all = rng.normal(0.0, 1.0, size=(n_samples, state_dim)).astype(float)
    train_fraction = float(config.get("train_fraction", 1.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    if test_fraction == 0.0:
        return x_all
    split_rng = np.random.default_rng(int(seeds["split_seed"]))
    shuffled = split_rng.permutation(n_samples).astype(int)
    n_test = int(round(test_fraction * n_samples))
    n_test = min(max(n_test, 1), n_samples - 1)
    train_indices = shuffled[n_test:]
    expected_train = int(round(train_fraction * n_samples))
    if train_indices.size != expected_train:
        raise ValueError("Reconstructed train split size does not match saved config fractions.")
    return x_all[train_indices]


def _action_grid(spec: PlantedLogisticSpec, points: Sequence[SweepPoint], grid_size: int) -> np.ndarray:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")
    final_actions = np.asarray([point.mean_action_biased_solution for point in points], dtype=float)
    lower = min(spec.action_low, float(np.min(final_actions)) - 0.05)
    upper = max(spec.action_high, float(np.max(final_actions)) + 0.05)
    return np.linspace(lower, upper, grid_size)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


def _true_values_at_u(spec: PlantedLogisticSpec, x_batch: np.ndarray, u: np.ndarray | float) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    beta_x = x_batch @ spec.beta[: x_batch.shape[1]]
    z_star = spec.alpha * spec.u_star + beta_x + spec.bias
    p_star = _sigmoid(z_star)
    if u_arr.ndim == 0:
        z = spec.alpha * float(u_arr) + beta_x + spec.bias
        return np.logaddexp(0.0, z) - p_star * z
    return np.asarray([_true_values_at_u(spec, x_batch, float(u_val)).mean() for u_val in u_arr], dtype=float)


def _true_mean_at_u(spec: PlantedLogisticSpec, x_batch: np.ndarray, u: np.ndarray) -> np.ndarray:
    return _true_values_at_u(spec, x_batch, u)


def _biased_mean_at_u(spec: PlantedLogisticSpec, x_batch: np.ndarray, lambda_bias: float, u: np.ndarray) -> np.ndarray:
    return _true_mean_at_u(spec, x_batch, u) - float(lambda_bias) * np.asarray(u, dtype=float)


def _step_traces(points: Sequence[SweepPoint], *, oracle_run_dir: Path) -> list[StepTrace]:
    traces: list[StepTrace] = []
    oracle_steps = _read_steps(oracle_run_dir / "plots" / "optimization" / "steps.csv")
    if oracle_steps is not None:
        traces.append(StepTrace(None, "oracle: $J_\star(\theta_t)$", *oracle_steps))
    for point in points:
        steps = _read_steps(point.biased_run_dir / "plots" / "optimization" / "steps.csv")
        if steps is None:
            continue
        traces.append(StepTrace(point.lambda_bias, rf"$\lambda_{{bias}}={point.lambda_bias:g}$: $\hat J_\lambda(\theta_t)$", *steps))
    return traces


def _read_steps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((float(row["step"]), float(row["u"]), float(row["value"])))
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _plot_objective_slices(
    path: Path,
    spec: PlantedLogisticSpec,
    x_train: np.ndarray,
    points: Sequence[SweepPoint],
    u_grid: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    true_curve = _true_mean_at_u(spec, x_train, u_grid)
    ax.plot(u_grid, true_curve, color="black", linewidth=2.2, label=r"true $\bar M_\star(u)$")
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, len(points)))
    for point, color in zip(points, colors):
        ax.plot(
            u_grid,
            _biased_mean_at_u(spec, x_train, point.lambda_bias, u_grid),
            color=color,
            linewidth=1.8,
            label=rf"$\bar M_\lambda(u),\ \lambda_{{bias}}={point.lambda_bias:g}$",
        )
        ax.scatter(
            [point.mean_action_biased_solution],
            [point.surrogate_objective_at_biased_solution],
            color=color,
            s=28,
            zorder=3,
        )
    ax.axvline(spec.u_star, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label=rf"$u^\star={spec.u_star:g}$")
    ax.set_xlabel(r"Constant action $u$")
    ax.set_ylabel(r"Mean objective value $n^{-1}\sum_{i=1}^n M(x_i,u)$")
    ax.set_title(r"True and biased planted-logistic constant-action objectives")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_surrogate_minus_true(path: Path, points: Sequence[SweepPoint], u_grid: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, len(points)))
    for point, color in zip(points, colors):
        ax.plot(
            u_grid,
            -point.lambda_bias * u_grid,
            color=color,
            linewidth=2.0,
            label=rf"$\lambda_{{bias}}={point.lambda_bias:g}$",
        )
        ax.scatter(
            [point.mean_action_biased_solution],
            [point.optimism_gap],
            color=color,
            s=28,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel(r"Constant action $u$")
    ax.set_ylabel(r"$\bar M_\lambda(u)-\bar M_\star(u)=-\lambda_{bias}u$")
    ax.set_title(r"Deterministic optimism term in the biased surrogate")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sweep_metrics(path: Path, points: Sequence[SweepPoint]) -> None:
    lambdas = np.asarray([point.lambda_bias for point in points], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    axes[0].plot(lambdas, [point.mean_action_biased_solution for point in points], marker="o")
    axes[0].axhline(points[0].mean_action_oracle, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    axes[0].set_ylabel(r"Mean action $\bar u_\lambda=n^{-1}\sum_i\pi_{\hat\theta_\lambda}(x_i)$")
    axes[1].plot(lambdas, [point.true_gap for point in points], marker="o", color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axes[1].set_ylabel(r"True gap $J_\star(\hat\theta_\lambda)-J_\star(\hat\theta_{oracle})$")
    axes[2].plot(lambdas, [point.optimism_gap for point in points], marker="o", color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axes[2].set_ylabel(r"Signed optimism $\hat J_\lambda(\hat\theta_\lambda)-J_\star(\hat\theta_\lambda)$")
    axes[2].set_xlabel(r"Bias strength $\lambda_{bias}$")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(r"Policy exploitation metrics over deterministic action bias")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_theta_drift(path: Path, points: Sequence[SweepPoint]) -> None:
    lambdas = np.asarray([point.lambda_bias for point in points], dtype=float)
    theta_deltas = np.vstack([point.theta_delta_from_oracle for point in points])
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    axes[0].plot(lambdas, [point.theta_l2_from_oracle for point in points], marker="o", color="tab:blue")
    axes[0].set_ylabel(r"Parameter drift $\|\hat\theta_\lambda-\hat\theta_{oracle}\|_2$")
    for idx in range(theta_deltas.shape[1]):
        label = THETA_LABELS[idx] if idx < len(THETA_LABELS) else rf"$\theta_{idx}$"
        axes[1].plot(lambdas, theta_deltas[:, idx], marker="o", label=label)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    axes[1].set_xlabel(r"Bias strength $\lambda_{bias}$")
    axes[1].set_ylabel(r"Coordinate drift $\hat\theta_{\lambda,j}-\hat\theta_{oracle,j}$")
    axes[1].legend(fontsize=8, ncols=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(r"Theta displacement induced by deterministic action bias")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_optimization_traces(path: Path, traces: Sequence[StepTrace]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=False)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(traces)))
    for trace, color in zip(traces, colors):
        axes[0].plot(trace.steps, trace.mean_u, marker="o", markersize=3, linewidth=1.5, color=color, label=trace.label)
        axes[1].plot(trace.steps, trace.objective_value, marker="o", markersize=3, linewidth=1.5, color=color, label=trace.label)
    axes[0].set_xlabel(r"Optimizer callback step $t$")
    axes[0].set_ylabel(r"Mean policy action $\bar u_t=n^{-1}\sum_i\pi_{\theta_t}(x_i)$")
    axes[1].set_xlabel(r"Optimizer callback step $t$")
    axes[1].set_ylabel(r"Optimizer-facing objective value ($J_\star$ for oracle, $\hat J_\lambda$ for biased runs)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncols=2)
    fig.suptitle(r"Saved optimization traces by bias strength")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _diagnostic_fieldnames(points: Sequence[SweepPoint]) -> tuple[str, ...]:
    max_theta_dim = max((point.theta_delta_from_oracle.size for point in points), default=0)
    return (
        "lambda_bias",
        "true_objective_at_oracle",
        "true_objective_at_biased_solution",
        "true_gap",
        "mean_action_oracle",
        "mean_action_biased_solution",
        "surrogate_objective_at_biased_solution",
        "optimism_gap",
        "theta_l2_from_oracle",
        *(f"theta_{idx}_delta_from_oracle" for idx in range(max_theta_dim)),
        "optimizer_success",
        "optimizer_status",
        "optimizer_message",
        "oracle_run_dir",
        "biased_run_dir",
    )


if __name__ == "__main__":
    main()
