"""Mathematical PDF illustrations of analytically certified all-real bands."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bootstrap_band import features


def _read_example(project, row):
    path = project / f"{row['stage']}-n{row['n']}-sd{row['noise_std']:g}" / "draws.npz"
    with np.load(path) as data:
        return {key: data[key][row["dataset"]] for key in data.files}


def _curves(data, a):
    p = features(a)
    se = data["sigma_hat"] * np.sqrt(np.einsum("ij,jk,ik->i", p, data["covariance"], p))
    return p @ data["beta_hat"], p @ data["error_coefficients"], data["critical_value"]*se


def write_plots(project: Path, source: Path, specification: dict, rows: list, summaries: list):
    """Sample analytic formulas only for display; read coverage from certificates."""
    first = rows[0]
    data = _read_example(project, first)
    a = np.linspace(*specification["display_window"], 1001)
    fitted, error, radius = _curves(data, a)
    truth = 5*a - 5*a**2
    bootstrap_curve = features(a) @ data["bootstrap_beta"][0]
    output = project / "plots"
    output.mkdir(exist_ok=True)
    paths = []
    with np.load(source / f"stage1-n{first['n']}-sd{first['noise_std']:g}" / "draws.npz") as saved:
        b = len(saved["bootstrap_beta"][0])
    from experiments.policy_lcb.common import read_json
    delta = read_json(source / "summary.json")["manifest"]["bootstrap"]["delta"]

    def save(fig, filename):
        path = output / filename
        fig.savefig(path, format="pdf")
        plt.close(fig)
        paths.append(path)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
                         "figure.titlesize": 16, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(a, truth, label=r"$f(a)=5a-5a^2$")
        axes[0].plot(a, fitted, label=r"$\widehat f(a)=p(a)^\top\widehat\beta$")
        axes[0].plot(a, bootstrap_curve, linestyle="--", label=r"$\widehat f_1^*(a)=p(a)^\top\widehat\beta_1^*$")
        axes[0].fill_between(a, fitted-radius, fitted+radius, color="C1", alpha=.2,
                             label=r"$\widehat f(a)\pm r_\delta(a)$")
        axes[0].set(xlabel=r"$a$ (finite display window)", ylabel=r"Mean response $f(a)$", title="True, fitted, and one bootstrap curve")
        axes[1].plot(a, error, label=r"$e(a)=p(a)^\top(\widehat\beta-\beta_0)$")
        axes[1].plot(a, radius, color="C1", label=r"$\pm r_\delta(a)=\pm\widehat c\,\widehat s(a)$")
        axes[1].plot(a, -radius, color="C1")
        axes[1].axhline(0, color="C2", linestyle=":")
        axes[1].set(xlabel=r"$a$ (finite display window)", ylabel=r"Estimation error $e(a)$", title=rf"Analytical containment on $\mathbb{{R}}$: $C={int(data['simultaneous_covered'])}$")
        for ax in axes:
            ax.legend()
        fig.suptitle(rf"$n={first['n']},\quad\sigma={first['noise_std']:g},\quad B={b},\quad\delta={delta:g}$")
        save(fig, "01_function_and_error.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].hist(data["bootstrap_supremum_upper"], bins=25, label=r"$T_b^*$: bootstrap suprema")
        axes[0].axvline(data["critical_value"], color="C1", label=rf"$\widehat c={data['critical_value']:.3f}$")
        axes[0].axvline(data["observed_statistic_upper"], color="C2", label=rf"$T_{{\rm obs}}={data['observed_statistic_upper']:.3f}$")
        axes[0].set(xlabel=r"$T_b^*$ (certified all-real supremum)", ylabel="Bootstrap draw count", title="Bootstrap calibration")
        axes[0].legend()
        theta = np.linspace(-np.pi/2, np.pi/2, 1001)
        # Projective coordinates render the entire real line plus its tail limits.
        homogeneous = np.column_stack((np.cos(theta)**2, np.sin(theta)*np.cos(theta), np.sin(theta)**2))
        normalized = np.abs(homogeneous @ data["error_coefficients"])/(data["critical_value"]*data["sigma_hat"]*np.sqrt(np.einsum("ij,jk,ik->i", homogeneous, data["covariance"], homogeneous)))
        axes[1].plot(theta, normalized, label=r"$|e(\tan\theta)|/r_\delta(\tan\theta)$")
        axes[1].axhline(1, color="C1", linestyle="--", label=r"Containment threshold $1$")
        axes[1].set_xticks([-np.pi/2, 0, np.pi/2], [r"$-\pi/2$", "$0$", r"$\pi/2$"])
        axes[1].set(xlabel=r"$\theta=\arctan(a)$; endpoints are $a=\pm\infty$", ylabel=r"Normalized error $|e(a)|/r_\delta(a)$", title="Entire real line and both tail limits")
        axes[1].legend()
        fig.suptitle(r"$e(a)=p(a)^\top\left(\sum_i p_i p_i^\top\right)^{-1}\sum_i p_i\epsilon_i$")
        save(fig, "02_calibration_and_all_real_containment.pdf")

        illustration_n = 100 if any(row["n"] == 100 and row["stage"] == "stage2" for row in rows) else summaries[0]["n"]
        examples = [row for row in rows if row["stage"] == "stage2" and row["n"] == illustration_n and row["dataset"] == 0]
        fig, axes = plt.subplots(1, len(examples), figsize=(5*len(examples), 4.7), sharey=True, squeeze=False, constrained_layout=True)
        for ax, row in zip(axes[0], examples):
            example = _read_example(project, row)
            _, err, rad = _curves(example, a)
            ax.plot(a, err, label=r"$e(a)=\widehat f(a)-f(a)$")
            ax.plot(a, rad, color="C1", label=r"$\pm r_\delta(a)$")
            ax.plot(a, -rad, color="C1")
            ax.axhline(0, color="C2", linestyle=":")
            ax.set(xlabel=r"$a$ (finite display window)", ylabel=r"Estimation error $e(a)$",
                   title=rf"$\sigma={row['noise_std']:g},\quad C={int(row['simultaneous_covered'])}$")
            ax.legend()
        fig.suptitle(rf"Saved independent datasets: $n={illustration_n}$; containment certified on $\mathbb{{R}}$")
        save(fig, "03_error_by_observation_noise.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for std in sorted({row["noise_std"] for row in summaries}):
            group = sorted([row for row in summaries if row["noise_std"] == std], key=lambda row: row["n"])
            n = [row["n"] for row in group]
            rates = np.array([row["coverage_rate"] for row in group])
            axes[0].errorbar(n, rates, yerr=[rates-[row["wilson_95_low"] for row in group],
                                           [row["wilson_95_high"] for row in group]-rates],
                            marker="o", capsize=3, label=rf"$\sigma={std:g}$")
            axes[1].plot(n, [row["mean_radius_display_0_1"] for row in group], marker="o", label=rf"$\sigma={std:g}$")
        axes[0].axhline(1-delta, color="C3", linestyle="--", label=rf"$1-\delta={1-delta:g}$")
        axes[0].set(xscale="log", ylim=(0, 1.02), xlabel=r"Training sample size $n$", ylabel=r"$\widehat{\rm Coverage}=R^{-1}\sum_{r=1}^R C_r$", title="All-real coverage; 95% Wilson intervals")
        axes[1].set(xscale="log", yscale="log", xlabel=r"Training sample size $n$", ylabel=r"Average $r_\delta(a)$ on $[0,1]$", title="Band half-width on a fixed display interval")
        for ax in axes:
            ns = sorted({row["n"] for row in summaries})
            ax.set_xticks(ns, [str(n) for n in ns])
            ax.legend()
        fig.suptitle(r"$C_r=\mathbf{1}[\,|e_r(a)|\leq r_{\delta,r}(a)\ \forall a\in\mathbb{R}\,]$")
        save(fig, "04_coverage_and_width.pdf")
    return paths
