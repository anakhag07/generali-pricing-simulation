"""Post-run diagnostics for the planted-logistic gradient-sample noise sweep.

Reads saved ``summary-seed-*.json`` files, reconstructs each run's synthetic
train batch/noise field, and evaluates final policies against the saved
first-order planted-logistic truth theta. No optimization is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from experiments.paths import results_root  # noqa: E402
from objective.base import sample_states  # noqa: E402
from objective.noise import (  # noqa: E402
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)
from objective.objectives import PlantedLogisticObjective  # noqa: E402
from objective.policy import IdentityFeatureMap, SoftmaxPolicy  # noqa: E402


PROJECT_NAME = "planted-logistic-gradient-sample-noise-sweep"
DEFAULT_TRUTH_SUMMARY = (
    results_root()
    / "planted_logistic_base"
    / "first_order_truth_20260701_174139"
    / "summary.json"
)
FINITE_DIFFERENCE = "finite_difference"
STEIN_DIFFERENCE = "stein_difference"
ESTIMATORS = (FINITE_DIFFERENCE, STEIN_DIFFERENCE)
NOISE_FAMILIES = ("homoskedastic", "heteroskedastic")

DIAGNOSTIC_FIELDNAMES = (
    "variant",
    "estimator",
    "run_seed",
    "noise_family",
    "noise_level",
    "n_grad_samples",
    "theta_distance_to_truth",
    "theta_mse_to_truth",
    "clean_objective_truth",
    "clean_objective_hat",
    "clean_objective_gap",
    "noisy_objective_truth",
    "noisy_objective_hat",
    "noisy_objective_gap",
    "noise_exploitation_gap",
    "final_u",
    "optimizer_success",
    "optimizer_status",
    "summary_path",
)
SUMMARY_METRICS = (
    "theta_distance_to_truth",
    "theta_mse_to_truth",
    "clean_objective_gap",
    "noisy_objective_gap",
    "noise_exploitation_gap",
    "final_u",
)
VARIANCE_FIELDNAMES = (
    "estimator",
    "noise_family",
    "noise_level",
    "n_grad_samples",
    "n_seeds",
    "theta_bias_squared",
    "theta_variance_trace",
    "theta_mse",
    *(f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in ("mean", "std", "var", "min", "max")),
)


@dataclass(frozen=True)
class DiagnosticRow:
    variant: str
    estimator: str
    run_seed: int
    noise_family: str
    noise_level: float
    n_grad_samples: int | None
    theta_hat: np.ndarray
    theta_distance_to_truth: float
    theta_mse_to_truth: float
    clean_objective_truth: float
    clean_objective_hat: float
    clean_objective_gap: float
    noisy_objective_truth: float
    noisy_objective_hat: float
    noisy_objective_gap: float
    noise_exploitation_gap: float
    final_u: float
    optimizer_success: bool | None
    optimizer_status: int | None
    summary_path: Path

    def csv_row(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "estimator": self.estimator,
            "run_seed": self.run_seed,
            "noise_family": self.noise_family,
            "noise_level": self.noise_level,
            "n_grad_samples": "" if self.n_grad_samples is None else self.n_grad_samples,
            "theta_distance_to_truth": self.theta_distance_to_truth,
            "theta_mse_to_truth": self.theta_mse_to_truth,
            "clean_objective_truth": self.clean_objective_truth,
            "clean_objective_hat": self.clean_objective_hat,
            "clean_objective_gap": self.clean_objective_gap,
            "noisy_objective_truth": self.noisy_objective_truth,
            "noisy_objective_hat": self.noisy_objective_hat,
            "noisy_objective_gap": self.noisy_objective_gap,
            "noise_exploitation_gap": self.noise_exploitation_gap,
            "final_u": self.final_u,
            "optimizer_success": "" if self.optimizer_success is None else self.optimizer_success,
            "optimizer_status": "" if self.optimizer_status is None else self.optimizer_status,
            "summary_path": str(self.summary_path),
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=results_root() / PROJECT_NAME,
        help="Completed sweep project directory.",
    )
    parser.add_argument(
        "--truth-summary",
        type=Path,
        default=DEFAULT_TRUTH_SUMMARY,
        help="Noiseless first-order planted-logistic truth summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <project-dir>/diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir or args.project_dir / "diagnostics"
    rows = collect_diagnostic_rows(args.project_dir, args.truth_summary)
    if not rows:
        raise ValueError(f"No diagnostic rows found under {args.project_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "diagnostic_rows.csv", [row.csv_row() for row in rows], DIAGNOSTIC_FIELDNAMES)
    summary_rows = variance_summary_rows(rows, truth_theta=_truth_theta(args.truth_summary))
    _write_rows(output_dir / "variance_summary.csv", summary_rows, VARIANCE_FIELDNAMES)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_diagnostic_plots(rows, plot_dir)
    print(f"Wrote {len(rows)} diagnostic rows to {output_dir / 'diagnostic_rows.csv'}")
    print(f"Wrote variance summary to {output_dir / 'variance_summary.csv'}")
    print(f"Wrote plots under {plot_dir}")


def collect_diagnostic_rows(project_dir: Path, truth_summary: Path) -> list[DiagnosticRow]:
    theta_truth = _truth_theta(truth_summary)
    rows: list[DiagnosticRow] = []
    for summary_path in sorted(project_dir.glob("*/summary-seed-*.json")):
        summary = _load_json(summary_path)
        preset = summary.get("preset", {})
        estimator = _summary_estimator(summary)
        if estimator not in ESTIMATORS:
            continue
        base_objective = _base_objective(summary)
        noisy_objective = _noisy_objective(summary, base_objective)
        x_train = _train_x(summary)
        estimator_payload = summary["estimators"][estimator]
        theta_hat = np.asarray(estimator_payload["theta"], dtype=float)
        clean_truth = float(base_objective.value(theta_truth, x_train))
        clean_hat = float(base_objective.value(theta_hat, x_train))
        noisy_truth = float(noisy_objective.value(theta_truth, x_train))
        noisy_hat = float(noisy_objective.value(theta_hat, x_train))
        clean_gap = clean_hat - clean_truth
        noisy_gap = noisy_hat - noisy_truth
        rows.append(
            DiagnosticRow(
                variant=str(preset.get("variant_name", summary_path.parent.name)),
                estimator=estimator,
                run_seed=_run_seed(summary, summary_path),
                noise_family=str(preset.get("noise_family", _noise_family(summary))),
                noise_level=float(preset.get("noise_level", _noise_level(summary))),
                n_grad_samples=_optional_int(preset.get("n_grad_samples")),
                theta_hat=theta_hat,
                theta_distance_to_truth=float(np.linalg.norm(theta_hat - theta_truth)),
                theta_mse_to_truth=float(np.mean((theta_hat - theta_truth) ** 2)),
                clean_objective_truth=clean_truth,
                clean_objective_hat=clean_hat,
                clean_objective_gap=clean_gap,
                noisy_objective_truth=noisy_truth,
                noisy_objective_hat=noisy_hat,
                noisy_objective_gap=noisy_gap,
                noise_exploitation_gap=noisy_gap - clean_gap,
                final_u=float(estimator_payload["final_u"]),
                optimizer_success=_optional_bool(estimator_payload.get("optimizer_success")),
                optimizer_status=_optional_int(estimator_payload.get("optimizer_status")),
                summary_path=summary_path,
            )
        )
    return sorted(rows, key=_row_sort_key)


def variance_summary_rows(
    rows: Sequence[DiagnosticRow],
    *,
    truth_theta: np.ndarray,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, int | None], list[DiagnosticRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.estimator, row.noise_family, row.noise_level, row.n_grad_samples)].append(row)
    out: list[dict[str, object]] = []
    for key in sorted(grouped, key=_group_key_sort_key):
        estimator, noise_family, noise_level, n_grad_samples = key
        group = sorted(grouped[key], key=lambda row: row.run_seed)
        theta_errors = np.asarray([row.theta_hat - truth_theta for row in group], dtype=float)
        mean_error = np.mean(theta_errors, axis=0)
        theta_sample_var = np.var(theta_errors, axis=0, ddof=1) if len(group) > 1 else np.zeros(theta_errors.shape[1])
        summary: dict[str, object] = {
            "estimator": estimator,
            "noise_family": noise_family,
            "noise_level": noise_level,
            "n_grad_samples": "" if n_grad_samples is None else n_grad_samples,
            "n_seeds": len(group),
            "theta_bias_squared": float(np.dot(mean_error, mean_error)),
            "theta_variance_trace": float(np.sum(theta_sample_var)),
            "theta_mse": float(np.mean(np.sum(theta_errors**2, axis=1))),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([getattr(row, metric) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = _sample_std(values)
            summary[f"{metric}_var"] = _sample_var(values)
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        out.append(summary)
    return out


def write_diagnostic_plots(rows: Sequence[DiagnosticRow], plot_dir: Path) -> None:
    for family in NOISE_FAMILIES:
        finite_rows = [row for row in rows if row.estimator == FINITE_DIFFERENCE and row.noise_family == family]
        if finite_rows:
            _plot_finite_difference_noise_sweep(
                finite_rows,
                plot_dir / f"finite_difference_{family}_noise_sweep.png",
                family,
            )
        stein_rows = [row for row in rows if row.estimator == STEIN_DIFFERENCE and row.noise_family == family]
        if stein_rows:
            _plot_stein_difference_ngrad_grid(
                stein_rows,
                plot_dir / f"stein_difference_{family}_ngrad_grid.png",
                family,
            )


def _plot_finite_difference_noise_sweep(
    rows: Sequence[DiagnosticRow],
    path: Path,
    family: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _plot_metrics()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    levels = sorted({row.noise_level for row in rows})
    for ax, (metric, y_label) in zip(axes, metrics):
        means, stds = _means_stds_by_x(rows, metric, levels, x_attr="noise_level")
        ax.errorbar(
            levels,
            means,
            yerr=_yerr(metric, means, stds),
            marker="o",
            linewidth=1.8,
            markersize=5.5,
            capsize=3.0,
            color="#8c564b",
        )
        _style_axis(ax, levels, metric, y_label)
        ax.set_xlabel(_noise_axis_label(family))
    fig.suptitle(
        f"Finite difference: {family} noise sweep (mean +/- sample std over seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_stein_difference_ngrad_grid(
    rows: Sequence[DiagnosticRow],
    path: Path,
    family: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    import matplotlib.pyplot as plt

    metrics = _plot_metrics()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    levels = sorted({row.noise_level for row in rows})
    n_grads = sorted({int(row.n_grad_samples) for row in rows if row.n_grad_samples is not None})
    cmap = colormaps["viridis"]
    for level_index, level in enumerate(levels):
        level_rows = [row for row in rows if row.noise_level == level]
        color = cmap(0.85 * level_index / max(len(levels) - 1, 1))
        label = f"{_noise_symbol(family)} = {level:g}"
        for ax, (metric, _) in zip(axes, metrics):
            xs = sorted({int(row.n_grad_samples) for row in level_rows if row.n_grad_samples is not None})
            means, stds = _means_stds_by_x(level_rows, metric, xs, x_attr="n_grad_samples")
            ax.errorbar(
                xs,
                means,
                yerr=_yerr(metric, means, stds),
                label=label,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                capsize=3.0,
            )
    for ax, (metric, y_label) in zip(axes, metrics):
        _style_axis(ax, n_grads, metric, y_label)
        ax.set_xlabel("Stein-difference n_grad_samples")
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_grads)
        ax.set_xticklabels([str(value) for value in n_grads])
    axes[0].legend(title=_noise_axis_label(family))
    fig.suptitle(
        f"Stein difference: {family} noise x n_grad_samples grid (mean +/- sample std over seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_metrics() -> tuple[tuple[str, str], ...]:
    return (
        ("theta_distance_to_truth", r"$||\hat{\theta} - \theta^*||_2$"),
        ("clean_objective_gap", r"$J_{clean}(\hat{\theta}) - J_{clean}(\theta^*)$"),
        ("noisy_objective_gap", r"$J_{noisy}(\hat{\theta}) - J_{noisy}(\theta^*)$"),
    )


def _means_stds_by_x(
    rows: Sequence[DiagnosticRow],
    metric: str,
    x_values: Sequence[float | int],
    *,
    x_attr: str,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for value in x_values:
        group = [getattr(row, metric) for row in rows if getattr(row, x_attr) == value]
        arr = np.asarray(group, dtype=float)
        means.append(float(np.mean(arr)))
        stds.append(_sample_std(arr))
    return means, stds


def _yerr(metric: str, means: Sequence[float], stds: Sequence[float]) -> np.ndarray | None:
    if not any(std > 0.0 for std in stds):
        return None
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    if metric == "theta_distance_to_truth":
        return np.vstack([np.minimum(stds_arr, np.maximum(means_arr, 0.0)), stds_arr])
    return np.vstack([stds_arr, stds_arr])


def _style_axis(ax: object, x_values: Sequence[float | int], metric: str, y_label: str) -> None:
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    if x_values:
        ax.set_xticks(list(x_values))
    if metric == "theta_distance_to_truth":
        ax.set_ylim(bottom=0.0)
        return
    values = [line.get_ydata() for line in ax.lines]
    flattened = [float(value) for line in values for value in line if np.isfinite(value)]
    nonzero = [abs(value) for value in flattened if value != 0.0]
    if nonzero:
        ax.set_yscale("symlog", linthresh=max(min(nonzero) * 0.5, 1e-10))


def _train_x(summary: Mapping[str, Any]) -> np.ndarray:
    config = summary["config"]
    if config.get("x_fixed_shape") is not None:
        raise ValueError("This diagnostic only supports synthetic sample_states runs.")
    seeds = config["resolved_seed_setup"]
    x_all = sample_states(
        np.random.default_rng(int(seeds["data_seed"])),
        int(config["n_samples"]),
        int(config["state_dim"]),
    )
    test_fraction = float(config.get("test_fraction", 0.0))
    if test_fraction == 0.0:
        return x_all
    shuffled = np.random.default_rng(int(seeds["split_seed"])).permutation(x_all.shape[0]).astype(int)
    n_test = int(round(test_fraction * x_all.shape[0]))
    n_test = min(max(n_test, 1), x_all.shape[0] - 1)
    return x_all[shuffled[n_test:]]


def _base_objective(summary: Mapping[str, Any]) -> PlantedLogisticObjective:
    objective = summary["config"]["objective"]
    base = objective.get("base_objective", objective)
    if base.get("type") != "PlantedLogisticObjective":
        raise ValueError(f"Expected PlantedLogisticObjective, found {base.get('type')!r}")
    policy_config = base["policy"]
    feature_map = policy_config.get("feature_map", {})
    if policy_config.get("type") != "SoftmaxPolicy" or feature_map.get("kind") != "identity":
        raise ValueError("This diagnostic expects identity-feature SoftmaxPolicy summaries.")
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=float(policy_config["action_low"]),
        action_high=float(policy_config["action_high"]),
    )
    return PlantedLogisticObjective.from_parameters(
        policy=policy,
        alpha=float(base["alpha"]),
        beta=np.asarray(base["beta"], dtype=float),
        bias=float(base["bias"]),
        u_star=float(base["u_star"]),
    )


def _noisy_objective(
    summary: Mapping[str, Any],
    base_objective: PlantedLogisticObjective,
) -> NoisyObjective:
    objective = summary["config"]["objective"]
    noise_payload = objective.get("noise", {})
    noise_type = noise_payload.get("type")
    if noise_type == "HomoskedasticGaussianNoise":
        noise = HomoskedasticGaussianNoise(
            std=float(noise_payload["std"]),
            seed=_optional_int(noise_payload.get("seed")),
        )
    elif noise_type == "HeteroskedasticGaussianNoise":
        noise = HeteroskedasticGaussianNoise(
            base_std=float(noise_payload["base_std"]),
            growth=float(noise_payload["growth"]),
            u_center=float(noise_payload["u_center"]),
            seed=_optional_int(noise_payload.get("seed")),
        )
    else:
        raise ValueError(f"Unsupported noise type {noise_type!r}")
    return NoisyObjective(base_objective, noise)


def _truth_theta(path: Path) -> np.ndarray:
    payload = _load_json(path)
    return np.asarray(payload["estimators"]["first_order"]["theta"], dtype=float)


def _summary_estimator(summary: Mapping[str, Any]) -> str:
    estimators = tuple(summary.get("estimators", {}).keys())
    if len(estimators) != 1:
        raise ValueError(f"Expected exactly one estimator, found {estimators}")
    return str(estimators[0])


def _noise_family(summary: Mapping[str, Any]) -> str:
    noise_type = summary["config"]["objective"].get("noise", {}).get("type")
    if noise_type == "HomoskedasticGaussianNoise":
        return "homoskedastic"
    if noise_type == "HeteroskedasticGaussianNoise":
        return "heteroskedastic"
    raise ValueError(f"Unsupported noise type {noise_type!r}")


def _noise_level(summary: Mapping[str, Any]) -> float:
    noise = summary["config"]["objective"].get("noise", {})
    if "std" in noise:
        return float(noise["std"])
    if "growth" in noise:
        return float(noise["growth"])
    raise ValueError("Could not resolve noise level.")


def _run_seed(summary: Mapping[str, Any], summary_path: Path) -> int:
    seeds = summary.get("config", {}).get("resolved_seed_setup", {})
    if "run_seed" in seeds:
        return int(seeds["run_seed"])
    return int(summary_path.stem.removeprefix("summary-seed-"))


def _row_sort_key(row: DiagnosticRow) -> tuple[object, ...]:
    return (
        row.estimator,
        row.noise_family,
        row.noise_level,
        -1 if row.n_grad_samples is None else row.n_grad_samples,
        row.run_seed,
    )


def _group_key_sort_key(key: tuple[str, str, float, int | None]) -> tuple[object, ...]:
    estimator, family, level, n_grad = key
    return (estimator, family, level, -1 if n_grad is None else n_grad)


def _noise_axis_label(family: str) -> str:
    if family == "homoskedastic":
        return "Homoskedastic noise std"
    return "Heteroskedastic noise growth"


def _noise_symbol(family: str) -> str:
    return "std" if family == "homoskedastic" else "growth"


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sample_var(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _sample_std(values: np.ndarray) -> float:
    return float(np.sqrt(_sample_var(values)))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    return bool(value)


if __name__ == "__main__":
    main()
