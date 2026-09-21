"""Three clean metric-by-axis PDFs from completed controlled sweep summaries."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sweep(rows, payload, destination):
    """Plot coverage, local half-width, and regret versus sigma, N, and B."""
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
    with plt.rc_context(style):
        for metric, filename, title in zip(("coverage", "width_at_true_optimum", "regret"), names, titles):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
            fig.suptitle(title)
            for ax, axis in zip(axes, ("sigma", "N", "B")):
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
                ax.set_xticks(x, [f"{v:g}" for v in x])
                ax.set_xlabel(labels[axis])
                ax.legend()
            path = destination/filename
            fig.savefig(path, format="pdf")
            plt.close(fig)
            figures.append(path)
    return figures
