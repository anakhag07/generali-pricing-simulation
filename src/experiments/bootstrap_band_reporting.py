"""PDF diagnostics from saved quadratic bootstrap band outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_plots(project: Path, payload: dict, rows: list[dict], summaries: list[dict]) -> list[Path]:
    """Render the example and sample-size/noise diagnostics without refitting."""
    first = payload["stage1"]
    source = project / f"stage1-n{first['n']}-sd{first['noise_std']:g}" / "draws.npz"
    with np.load(source) as archive:
        example = {key: archive[key][0] for key in archive.files}
    output = project / "plots"
    output.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, format="pdf")
        plt.close(fig)
        paths.append(path)

    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "figure.titlesize": 16, "legend.fontsize": 10,
             "xtick.labelsize": 10, "ytick.labelsize": 10}
    with plt.rc_context(style):
        x, grid = example["x"], example["grid"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        design_grid = np.linspace(min(x.min(), grid[0]), max(x.max(), grid[-1]), 1001)
        axes[0].scatter(x, example["y"], label="Observed responses", s=15)
        axes[0].plot(design_grid, 5 * design_grid - 5 * design_grid**2, label="True f")
        axes[0].plot(design_grid, np.polynomial.polynomial.polyval(design_grid, example["beta_hat"]), label="OLS fit")
        axes[0].set(title="Training data: x drawn from N(0,1)", xlabel="Training input x", ylabel="Response y")
        axes[0].legend()
        axes[1].plot(grid, example["truth"], label="True f(a)")
        axes[1].plot(grid, example["fitted"], label="OLS estimate")
        axes[1].fill_between(grid, example["lower"], example["upper"], alpha=0.2,
                             label=f"{100*(1-payload['bootstrap']['delta']):g}% nominal bootstrap band")
        axes[1].set(title=f"All grid points covered: {bool(example['simultaneous_covered'])}",
                    xlabel="Evaluation action a", ylabel="Mean response f(a)")
        axes[1].legend()
        fig.suptitle(f"One dataset: n={first['n']}, noise SD={first['noise_std']:g}, B={payload['bootstrap']['draws']}")
        save(fig, "01_single_dataset_band.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].plot(grid, example["standard_error"], label="Prediction standard error")
        axes[0].plot(grid, example["radius"], label="Band half-width")
        axes[0].set(xlabel="Evaluation action a", ylabel="Response units", title="Variable band width from the fitted design")
        axes[0].legend()
        axes[1].hist(example["bootstrap_maxima"], bins=25, label="Bootstrap grid maxima")
        axes[1].axvline(example["critical_value"], color="C1", label=f"Critical value = {example['critical_value']:.3f}")
        axes[1].axvline(example["observed_max_statistic"], color="C2", label=f"Observed statistic = {example['observed_max_statistic']:.3f}")
        axes[1].set(xlabel="Maximum standardized error on the grid", ylabel="Bootstrap draw count", title="Direct simultaneous calibration")
        axes[1].legend()
        save(fig, "02_band_construction.pdf")

        stds = payload["stage2"]["noise_stds"]
        fig, axes = plt.subplots(1, len(stds), figsize=(5 * len(stds), 4.5), squeeze=False, constrained_layout=True)
        for ax, std in zip(axes[0], stds):
            group = [row for row in summaries if row["noise_std"] == std]
            rates = np.array([row["coverage_rate"] for row in group])
            ax.errorbar([row["n"] for row in group], rates,
                        yerr=[rates - [row["wilson_95_low"] for row in group],
                              [row["wilson_95_high"] for row in group] - rates],
                        fmt="o", capsize=3, label="Empirical + 95% Wilson CI")
            ax.axhline(1 - payload["bootstrap"]["delta"], color="C1", linestyle="--", label="Nominal target")
            ax.set(xscale="log", ylim=(0, 1.02), xlabel="Training sample size n", ylabel="Fraction covering every grid point", title=f"Observation noise SD = {std:g}")
            ax.legend()
        fig.suptitle("Approximate bootstrap coverage on the specified grid")
        save(fig, "03_empirical_grid_coverage.pdf")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for j, ax in enumerate(axes):
            for std in stds:
                group = [row for row in summaries if row["noise_std"] == std]
                ax.errorbar([row["n"] for row in group], [row[f"beta_hat_{j}_mean"] for row in group],
                            yerr=[row[f"beta_hat_{j}_std"] for row in group], marker="o", capsize=3,
                            label=f"Noise SD = {std:g}")
            ax.axhline(payload["truth"]["coefficients"][j], color="black", linestyle="--", label="True coefficient")
            ax.set(xscale="log", xlabel="Training sample size n", ylabel=f"Estimated coefficient beta[{j}]", title=f"Coefficient {j}: mean ± dataset SD")
            ax.legend()
        save(fig, "04_ols_coefficients.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for std in stds:
            group = [row for row in summaries if row["noise_std"] == std]
            ns = [row["n"] for row in group]
            axes[0].errorbar(ns, [row["sigma2_hat_mean"] / std**2 for row in group],
                             yerr=[row["sigma2_hat_std"] / std**2 for row in group], marker="o", capsize=3, label=f"Noise SD = {std:g}")
            axes[1].plot(ns, [row["mean_radius_mean"] for row in group], marker="o", label=f"Noise SD = {std:g}")
        axes[0].axhline(1, color="black", linestyle="--", label="True variance ratio")
        axes[0].set(xscale="log", xlabel="Training sample size n", ylabel="Estimated variance / true variance", title="Residual variance: mean ± dataset SD")
        axes[1].set(xscale="log", yscale="log", xlabel="Training sample size n", ylabel="Average band half-width on grid", title="Band width versus sample size and noise")
        for ax in axes:
            ax.legend()
        save(fig, "05_variance_and_band_width.pdf")
    return paths
