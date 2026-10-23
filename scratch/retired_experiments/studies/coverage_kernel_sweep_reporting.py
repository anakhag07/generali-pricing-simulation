"""Default-Matplotlib vector reports for the Gaussian-support N sweep."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sweep(rows, payload, destination):
    """Write coverage, half-width, and regret PDFs from the completed summary."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
             "figure.titlesize": 16}
    ordered = sorted(rows, key=lambda row: row["N"])
    x = np.array([row["N"] for row in ordered])
    specs = (
        ("01_simultaneous_coverage.pdf", "Two-sided simultaneous coverage on the real line", "coverage"),
        ("02_envelope_half_width.pdf", "Envelope half-width at the true optimum", "width_at_true_optimum"),
        ("03_lcb_regret.pdf", "Regret of the lower-envelope action", "regret"),
    )
    figures = []
    with plt.rc_context(style):
        for filename, title, metric in specs:
            fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
            fig.suptitle(title)
            if metric == "coverage":
                y = np.array([row["coverage"] if row["coverage"] is not None else np.nan for row in ordered])
                lower = np.array([row["coverage_lower"] if row["coverage_lower"] is not None else np.nan for row in ordered])
                upper = np.array([row["coverage_upper"] if row["coverage_upper"] is not None else np.nan for row in ordered])
                ax.errorbar(x, y, yerr=np.vstack((y-lower, upper-y)), marker="o", capsize=3,
                            label="Resolved datasets; 95% Wilson interval")
                ax.axhline(1-payload["alpha"], linestyle="--", color="C1", label="Nominal 95% pointwise level")
                ax.set_ylabel("Empirical simultaneous coverage")
                ax.set_ylim(-0.05, 1.05)
            else:
                y = np.array([row[f"{metric}_mean"] if row[f"{metric}_mean"] is not None else np.nan for row in ordered])
                se = np.array([row[f"{metric}_se"] if row[f"{metric}_se"] is not None else np.nan for row in ordered])
                label = "Mean ± SE"
                if metric == "regret" and any(row["optimizer_certified_count"] != row["datasets"] for row in ordered):
                    label = "Globally certified cases; mean ± SE"
                ax.errorbar(x, y, yerr=se, marker="o", capsize=3, label=label)
                if metric == "regret":
                    bound = [row["regret_bound_mean"] for row in ordered]
                    ax.plot(x, bound, "--", label="Mean 2W; coverage-event benchmark")
                    ax.set_ylabel("Mean true regret")
                else:
                    ax.set_ylabel("Mean envelope half-width")
                if np.all(y > 0):
                    ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("Training sample size N")
            ax.legend()
            path = destination/filename
            fig.savefig(path, format="pdf")
            plt.close(fig)
            figures.append(path)
    return figures
