"""Default-Matplotlib vector reports for paired Gaussian-support parameter sweeps."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_sweep(rows, payload, destination):
    """Show q or b response at prespecified N values; full cells remain in CSV."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
             "figure.titlesize": 16}
    axis = "q" if payload["kind"] == "coverage_kernel_q_sweep" else "b"
    axis_label = "q" if axis == "q" else "bandwidth b"
    specs = (
        ("01_simultaneous_coverage.pdf", f"Two-sided simultaneous coverage versus {axis_label}", "coverage"),
        ("02_envelope_half_width.pdf", f"Envelope half-width versus {axis_label}", "width_at_true_optimum"),
        ("03_lcb_regret.pdf", f"True regret versus {axis_label}", "regret"),
    )
    baseline = float(norm.ppf(0.975)) if axis == "q" else 0.1
    figures = []
    with plt.rc_context(style):
        for filename, title, metric in specs:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
            fig.suptitle(title)
            for n in payload["report_N_values"]:
                group = sorted((row for row in rows if row["N"] == n), key=lambda row: row["axis_value"])
                x = np.array([row["axis_value"] for row in group])
                if metric == "coverage":
                    y = np.array([row["coverage"] if row["coverage"] is not None else np.nan for row in group])
                    lower = np.array([row["coverage_lower"] if row["coverage_lower"] is not None else np.nan for row in group])
                    upper = np.array([row["coverage_upper"] if row["coverage_upper"] is not None else np.nan for row in group])
                    errors = np.vstack((y-lower, upper-y))
                else:
                    y = np.array([row[f"{metric}_mean"] if row[f"{metric}_mean"] is not None else np.nan for row in group])
                    errors = np.array([row[f"{metric}_se"] if row[f"{metric}_se"] is not None else np.nan for row in group])
                ax.errorbar(x, y, yerr=errors, marker="o", capsize=2, label=f"N={n}")
            ax.axvline(baseline, linestyle="--", color="C4", label=f"Current {axis}={baseline:.2g}")
            if metric == "coverage":
                ax.axhline(0.95, linestyle=":", color="C5", label="95% reference")
                ax.set_ylabel("Empirical simultaneous coverage")
                ax.set_ylim(-0.05, 1.05)
            elif metric == "width_at_true_optimum":
                ax.set_ylabel("Mean half-width at true optimum")
            else:
                ax.set_ylabel("Mean true regret, certified cases")
            ax.set_xlabel("Envelope coefficient q (λ = q σ̂)" if axis == "q" else "Gaussian bandwidth b")
            ax.legend()
            path = destination/filename
            fig.savefig(path, format="pdf")
            plt.close(fig)
            figures.append(path)
    return figures
