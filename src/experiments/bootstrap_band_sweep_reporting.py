"""Three metric-by-axis PDFs from completed controlled sweep summaries."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sweep(rows, payload, destination):
    """Plot coverage, local half-width, and regret versus requested axes."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
             "figure.titlesize": 16}
    figures = []
    names = ["01_simultaneous_coverage.pdf", "02_envelope_half_width.pdf", "03_lcb_regret.pdf"]
    titles = [r"Simultaneous coverage: $\mathbb{P}(|e(a)|\leq r_\delta(a)\ \forall a\in\mathbb{R})$",
              r"Envelope half-width at the true optimum: $W=r_\delta(a^\star)$",
              r"Unconstrained LCB regret: $R_{\mathrm{LCB}}=f(a^\star)-f(\widehat a_{\mathrm{LCB}})$"]
    labels = {"sigma": r"Observation-noise SD $\sigma$", "N": r"Training sample size $N$",
              "B": r"Bootstrap refit count $B$"}
    axis_names = tuple(payload["axes"])
    with plt.rc_context(style):
        for metric, filename, title in zip(("coverage", "width_at_true_optimum", "regret"), names, titles):
            fig, axes = plt.subplots(1, len(axis_names), figsize=(max(7, 5*len(axis_names)), 4.5),
                                     sharey=True, squeeze=False, constrained_layout=True)
            fig.suptitle(title)
            for ax, axis in zip(axes[0], axis_names):
                group = sorted((r for r in rows if r["axis"] == axis), key=lambda r: r["axis_value"])
                x = np.array([r["axis_value"] for r in group])
                fixed = {k: v for k, v in payload["baseline"].items() if k != axis}
                math_names = {"N": "N", "sigma": r"\sigma", "B": "B"}
                subtitle = ",\\ ".join(f"{math_names[k]}={v:g}" for k, v in fixed.items())
                ax.set_title(rf"${subtitle},\ \delta={payload['delta']:g}$")
                if metric == "coverage":
                    y = np.array([r["coverage"] for r in group])
                    errors = np.array([[r["coverage"]-r["coverage_lower"] for r in group],
                                       [r["coverage_upper"]-r["coverage"] for r in group]])
                    ax.errorbar(x, y, yerr=errors, marker="o", capsize=3,
                                label=f"{payload['datasets']} datasets; 95% Wilson CI")
                    ax.axhline(1-payload["delta"], linestyle="--", color="C1", label=r"Target $1-\delta$")
                    ax.set_ylabel(r"$\widehat{\mathrm{Coverage}}=R^{-1}\sum_{r=1}^R C_r$")
                else:
                    y = np.array([r[f"{metric}_mean"] if r[f"{metric}_mean"] is not None else np.nan for r in group])
                    se = np.array([r[f"{metric}_se"] if r[f"{metric}_se"] is not None else np.nan for r in group])
                    excluded = metric == "regret" and any(r["certified_count"] != r["datasets"] for r in group)
                    ax.errorbar(x, y, yerr=se, marker="o", capsize=3,
                                label="Certified cases only; mean ± SE" if excluded else "Mean ± SE across datasets")
                    if metric == "regret":
                        bound = [r["regret_bound_mean"] for r in group]
                        ax.plot(x, bound, "--", label=r"$\mathbb{E}[2W]$; coverage-event benchmark")
                        ax.set_ylabel(r"Mean regret $\mathbb{E}[R_{\mathrm{LCB}}]$")
                    else:
                        ax.set_ylabel(r"Mean half-width $\mathbb{E}[r_\delta(a^\star)]$")
                    if np.all(y > 0):
                        ax.set_yscale("log")
                if axis != "sigma":
                    ax.set_xscale("log")
                if len(x) <= 7:
                    ax.set_xticks(x, [f"{v:g}" for v in x])
                ax.set_xlabel(labels[axis])
                ax.legend()
            path = destination/filename
            fig.savefig(path, format="pdf")
            plt.close(fig)
            figures.append(path)
    return figures


def plot_joint_coverage(rows, payload, destination):
    """Show empirical coverage at each joint N/B setting, with pointwise CIs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
             "figure.titlesize": 16}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(len(rows))
        y = np.array([r["coverage"] for r in rows])
        errors = np.array([[r["coverage"]-r["coverage_lower"] for r in rows],
                           [r["coverage_upper"]-r["coverage"] for r in rows]])
        ax.errorbar(x, y, yerr=errors, marker="o", capsize=3,
                    label=f"{payload['datasets']} datasets; 95% Wilson CI")
        ax.axhline(1-payload["delta"], color="C1", linestyle="--", label="Nominal coverage")
        ax.set_xticks(x, [f"{r['N']}\n{r['B']}" for r in rows])
        ax.set_xlabel("Joint setting: training size N (top), bootstrap count B (bottom)")
        ax.set_ylabel("Empirical simultaneous coverage")
        ax.set_title(rf"Pairs bootstrap: $\sigma={payload['sigma']:g}$, $\delta={payload['delta']:g}$")
        ax.legend()
        path = destination/"joint_N_B_coverage.pdf"
        fig.savefig(path, format="pdf")
        plt.close(fig)
    return path


def plot_cartesian_coverage(rows, payload, destination):
    """Render all N/B cells without interpolation, plus Monte Carlo uncertainty."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ns, bs = payload["grid"]["N"], payload["grid"]["B"]
    by_cell = {(r["N"], r["B"]): r for r in rows}
    if set(by_cell) != {(n,b) for n in ns for b in bs}:
        raise ValueError("A Cartesian heatmap requires every N/B cell.")
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.titlesize": 16}
    paths = []
    for uncertainty in (False, True):
        values = np.array([[(by_cell[n,b]["coverage_upper"]-by_cell[n,b]["coverage_lower"])/2
                            if uncertainty else by_cell[n,b]["coverage"] for b in bs] for n in ns])*100
        with plt.rc_context(style):
            fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
            mesh = ax.pcolormesh(np.arange(len(bs)+1), np.arange(len(ns)+1), values, cmap="viridis")
            ax.set_xticks(np.arange(len(bs))+.5, [str(b) for b in bs])
            ax.set_yticks(np.arange(len(ns))+.5, [str(n) for n in ns])
            ax.set_xlabel(r"Bootstrap refit count $B$")
            ax.set_ylabel(r"Training sample size $N$")
            title = "95% Wilson interval half-width" if uncertainty else "Empirical simultaneous coverage"
            ax.set_title(f"{title}\n{payload['datasets']} datasets per cell; "
                         rf"$\sigma={payload['sigma']:g}$; nominal coverage {100*(1-payload['delta']):g}%")
            colorbar = fig.colorbar(mesh, ax=ax)
            colorbar.set_label("Percentage points" if uncertainty else "Coverage (%)")
            for i in range(len(ns)):
                for j in range(len(bs)):
                    # Contrast only; the scalar values retain the standard viridis map.
                    color = "white" if mesh.norm(values[i,j]) < .45 else "black"
                    ax.text(j+.5, i+.5, f"{values[i,j]:.2f}", ha="center", va="center", color=color)
            path = destination/("coverage_interval_half_width.pdf" if uncertainty else "cartesian_N_B_coverage.pdf")
            fig.savefig(path, format="pdf")
            plt.close(fig)
            paths.append(path)
    return paths
