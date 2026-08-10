"""Aggregate tables and plots for continuous policy-LCB seed runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.policy_lcb.common import sample_std, shared_gaussian_coverage, wilson_interval
from experiments.policy_lcb.continuous import ContinuousLCBSeedResult, ContinuousPolicyLCBSpec
from experiments.seeds import derive_seed, rng_from_seed


def optimizer_summary_rows(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
) -> list[dict[str, object]]:
    """Aggregate best-of-start optimizer results across problem-noise seeds."""
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        for estimator in spec.optimizer.enabled_estimators:
            group = [
                row
                for seed_result in seed_results
                for row in seed_result.best_results
                if row.delta == delta and row.estimator == estimator
            ]
            policies = np.asarray([row.final_policy for row in group], dtype=float)
            gaps = np.asarray([row.optimization_error for row in group], dtype=float)
            steps = np.asarray([row.n_steps for row in group], dtype=float)
            regrets = np.asarray([row.regret for row in group], dtype=float)
            analytic = np.asarray([row.analytic_policy for row in group], dtype=float)
            ci_low, ci_high = _bootstrap_mean_interval(
                policies,
                seed=derive_seed(
                    spec.reporting_seed,
                    f"continuous-policy-lcb:mean-ci:{delta:g}:{estimator}",
                ),
            )
            rows.append(
                {
                    "delta": delta,
                    "estimator": estimator,
                    "n_seeds": len(group),
                    "converged_count": sum(row.converged for row in group),
                    "endpoint_match_count": sum(
                        np.isclose(row.final_policy, row.analytic_policy) for row in group
                    ),
                    "mean_final_policy": float(np.mean(policies)),
                    "std_final_policy": sample_std(policies),
                    "median_final_policy": float(np.median(policies)),
                    "final_policy_q25": float(np.quantile(policies, 0.25)),
                    "final_policy_q75": float(np.quantile(policies, 0.75)),
                    "mean_ci95_low": ci_low,
                    "mean_ci95_high": ci_high,
                    "analytic_mean_policy": float(np.mean(analytic)),
                    "mean_optimization_error": float(np.mean(gaps)),
                    "median_optimization_error": float(np.median(gaps)),
                    "optimization_error_q25": float(np.quantile(gaps, 0.25)),
                    "optimization_error_q75": float(np.quantile(gaps, 0.75)),
                    "mean_regret": float(np.mean(regrets)),
                    "median_steps": float(np.median(steps)),
                    "steps_q25": float(np.quantile(steps, 0.25)),
                    "steps_q75": float(np.quantile(steps, 0.75)),
                }
            )
    return rows


def coverage_summary_rows(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
) -> list[dict[str, object]]:
    """Summarize exact and empirical continuum-wide confidence coverage."""
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        quantile = next(
            row.quantile
            for result in seed_results
            for row in result.best_results
            if row.delta == delta
        )
        covered = sum(abs(result.z) <= quantile for result in seed_results)
        low, high = wilson_interval(covered, len(seed_results))
        rows.append(
            {
                "delta": delta,
                "quantile": quantile,
                "nominal_coverage": 1.0 - delta,
                "analytic_joint_coverage": shared_gaussian_coverage(delta),
                "n_seeds": len(seed_results),
                "covered_count": covered,
                "empirical_coverage": covered / len(seed_results),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return rows


def oracle_summary_rows(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
) -> list[dict[str, object]]:
    """Summarize continuous-comparator oracle checks by estimator and delta."""
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        for estimator in spec.optimizer.enabled_estimators:
            group = [
                row
                for result in seed_results
                for row in result.best_results
                if row.delta == delta and row.estimator == estimator
            ]
            covered = [row for row in group if row.simultaneous_coverage]
            slacks = np.asarray([row.worst_oracle_slack for row in group], dtype=float)
            rows.append(
                {
                    "delta": delta,
                    "estimator": estimator,
                    "n_seeds": len(group),
                    "covered_seed_count": len(covered),
                    "conditional_violation_count": sum(row.oracle_violation for row in covered),
                    "unconditional_violation_count": sum(row.oracle_violation for row in group),
                    "minimum_worst_slack": float(np.min(slacks)),
                    "mean_worst_slack": float(np.mean(slacks)),
                    "median_worst_slack": float(np.median(slacks)),
                    "worst_slack_q05": float(np.quantile(slacks, 0.05)),
                    "worst_slack_q95": float(np.quantile(slacks, 0.95)),
                }
            )
    return rows


def write_continuous_policy_lcb_plots(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
    optimizer_summary: Sequence[Mapping[str, object]],
    coverage_summary: Sequence[Mapping[str, object]],
    project_dir: Path,
) -> None:
    """Write aggregate and per-seed continuous policy-LCB diagnostics."""
    plots_dir = project_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_final_policy_bands(
        spec,
        seed_results,
        optimizer_summary,
        plots_dir / "final_policy_median_iqr.png",
        center="median",
    )
    _plot_final_policy_bands(
        spec,
        seed_results,
        optimizer_summary,
        plots_dir / "final_policy_mean_ci95.png",
        center="mean",
    )
    _plot_summary_band(
        spec,
        optimizer_summary,
        plots_dir / "optimization_gap.png",
        center_key="median_optimization_error",
        low_key="optimization_error_q25",
        high_key="optimization_error_q75",
        ylabel="LCB optimization error",
        title="Optimization error: median and 25-75% seed spread",
    )
    _plot_summary_band(
        spec,
        optimizer_summary,
        plots_dir / "convergence_steps.png",
        center_key="median_steps",
        low_key="steps_q25",
        high_key="steps_q75",
        ylabel="Projected updates",
        title="Convergence steps: median and 25-75% seed spread",
    )
    _plot_coverage(spec, coverage_summary, plots_dir / "coverage.png")
    _plot_oracle_slack(spec, seed_results, plots_dir / "oracle_slack.png")
    seed_dir = plots_dir / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for seed_result in seed_results:
        _plot_seed_diagnostics(spec, seed_result, seed_dir / f"seed-{seed_result.run_seed}.png")


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    values_arr = np.asarray(values, dtype=float)
    rng = rng_from_seed(seed)
    indices = rng.integers(0, values_arr.size, size=(n_resamples, values_arr.size))
    means = np.mean(values_arr[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _plot_final_policy_bands(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
    summary_rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    center: str,
) -> None:
    estimators = spec.optimizer.enabled_estimators
    fig, axes = plt.subplots(1, len(estimators), figsize=(5.0 * len(estimators), 4.8), sharey=True)
    axes_arr = np.atleast_1d(axes)
    x = np.arange(len(spec.deltas))
    for axis, estimator in zip(axes_arr, estimators):
        for seed_result in seed_results:
            values = [
                next(
                    row.final_policy
                    for row in seed_result.best_results
                    if row.delta == delta and row.estimator == estimator
                )
                for delta in spec.deltas
            ]
            axis.plot(x, values, color="0.55", linewidth=0.7, alpha=0.28)
        rows = [row for row in summary_rows if row["estimator"] == estimator]
        if center == "median":
            center_values = [float(row["median_final_policy"]) for row in rows]
            low = [float(row["final_policy_q25"]) for row in rows]
            high = [float(row["final_policy_q75"]) for row in rows]
            label = "Median; band = seed IQR"
        else:
            center_values = [float(row["mean_final_policy"]) for row in rows]
            low = [float(row["mean_ci95_low"]) for row in rows]
            high = [float(row["mean_ci95_high"]) for row in rows]
            label = "Mean; band = bootstrap 95% CI"
        axis.fill_between(x, low, high, alpha=0.24, color="tab:blue")
        axis.plot(x, center_values, marker="o", linewidth=2.2, color="tab:blue", label=label)
        axis.set_title(estimator.replace("_", " "))
        axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
        axis.set_xlabel(r"Failure probability $\delta$")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes_arr[0].set_ylabel(r"Final policy $\widehat\pi$")
    fig.suptitle(
        "Final continuous policy across Gaussian problem draws\n"
        + ("Empirical median and spread" if center == "median" else "Mean and uncertainty in the mean")
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary_band(
    spec: ContinuousPolicyLCBSpec,
    summary_rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    center_key: str,
    low_key: str,
    high_key: str,
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(spec.deltas))
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    for estimator in spec.optimizer.enabled_estimators:
        rows = [row for row in summary_rows if row["estimator"] == estimator]
        center = np.asarray([float(row[center_key]) for row in rows])
        low = np.asarray([float(row[low_key]) for row in rows])
        high = np.asarray([float(row[high_key]) for row in rows])
        line = axis.plot(x, center, marker="o", label=estimator.replace("_", " "))[0]
        axis.fill_between(x, low, high, color=line.get_color(), alpha=0.18)
    axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_coverage(
    spec: ContinuousPolicyLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    x = np.arange(len(spec.deltas))
    empirical = np.asarray([float(row["empirical_coverage"]) for row in rows])
    low = np.asarray([float(row["wilson_95_low"]) for row in rows])
    high = np.asarray([float(row["wilson_95_high"]) for row in rows])
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    nominal = [1.0 - delta for delta in spec.deltas]
    axis.plot(x, nominal, marker="o", label="Nominal = exact shared-Z coverage")
    axis.errorbar(
        x,
        empirical,
        yerr=np.maximum(0.0, np.vstack([empirical - low, high - empirical])),
        marker="^",
        capsize=4,
        label="Empirical (Wilson 95%)",
    )
    axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel("Simultaneous continuum coverage")
    axis.set_title("Shared-Gaussian confidence coverage over [0,1]")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_oracle_slack(
    spec: ContinuousPolicyLCBSpec,
    seed_results: Sequence[ContinuousLCBSeedResult],
    path: Path,
) -> None:
    estimators = spec.optimizer.enabled_estimators
    fig, axes = plt.subplots(1, len(estimators), figsize=(5.0 * len(estimators), 4.8), sharey=True)
    for axis, estimator in zip(np.atleast_1d(axes), estimators):
        data = [
            [
                row.worst_oracle_slack
                for result in seed_results
                for row in result.best_results
                if row.delta == delta and row.estimator == estimator
            ]
            for delta in spec.deltas
        ]
        axis.boxplot(data, tick_labels=[f"{delta:g}" for delta in spec.deltas], showmeans=True)
        axis.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
        axis.set_title(estimator.replace("_", " "))
        axis.set_xlabel(r"Failure probability $\delta$")
        axis.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Worst continuous-comparator oracle slack")
    fig.suptitle("Continuous policy-LCB oracle inequality")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_seed_diagnostics(
    spec: ContinuousPolicyLCBSpec,
    result: ContinuousLCBSeedResult,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(spec.deltas), figsize=(4.2 * len(spec.deltas), 4.3), sharey=True)
    policy_grid = np.linspace(0.0, 1.0, 101)
    for axis, delta in zip(np.atleast_1d(axes), spec.deltas):
        slope = next(row.quantile - 1.0 - result.z for row in result.best_results if row.delta == delta)
        axis.plot(policy_grid, policy_grid * slope, color="black", linewidth=1.5, label="negative LCB")
        for estimator in spec.optimizer.enabled_estimators:
            for start in spec.optimizer.starts:
                trace = [
                    row
                    for row in result.trajectories
                    if row.delta == delta and row.estimator == estimator and row.start_policy == start
                ]
                axis.plot(
                    [row.policy for row in trace],
                    [row.loss for row in trace],
                    linewidth=0.8,
                    alpha=0.45,
                )
            best = next(
                row for row in result.best_results if row.delta == delta and row.estimator == estimator
            )
            axis.scatter([best.final_policy], [best.final_loss], s=30, label=estimator.replace("_", " "))
        analytic = next(row.analytic_policy for row in result.best_results if row.delta == delta)
        axis.axvline(analytic, color="0.35", linestyle="--", linewidth=1.0)
        axis.set_title(fr"$\delta={delta:g}$")
        axis.set_xlabel(r"Policy $\pi$")
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("Negative LCB")
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Shared Gaussian draw seed {result.run_seed}: Z={result.z:.3f}")
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 1.0))
    fig.savefig(path, dpi=160)
    plt.close(fig)


__all__ = [
    "coverage_summary_rows",
    "optimizer_summary_rows",
    "oracle_summary_rows",
    "write_continuous_policy_lcb_plots",
]
