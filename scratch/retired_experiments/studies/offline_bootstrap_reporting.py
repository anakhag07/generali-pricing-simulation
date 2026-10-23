"""PDF diagnostics for one saved dataset and a fixed predictor."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


_LABELS = {"N": "Offline sample size N", "B": "Bootstrap refit count B",
           "sigma": r"Observation-noise scale $\sigma$"}
_STYLE = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
          "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
          "figure.titlesize": 16}


def _values(rows, key):
    return np.asarray([float(row[key]) if row.get(key) is not None else np.nan
                       for row in rows])


def _axis(ax, axis, values):
    ax.set_xlabel(_LABELS[axis])
    if axis in {"N", "B"} and np.all(values > 0):
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
    if len(values) <= 7:
        ax.set_xticks(values, [f"{value:g}" for value in values])


def _interval(ax, x, rows, prefix, label):
    lower = _values(rows, f"{prefix}_lower")
    upper = _values(rows, f"{prefix}_upper")
    midpoint = (lower + upper) / 2
    halfwidth = np.maximum(0, (upper - lower) / 2)
    ax.errorbar(x, midpoint, yerr=halfwidth, marker="o", capsize=3, label=label)


def write_plots(rows, payload, destination) -> list[Path]:
    """Plot whole-line containment, radius components, and certified regret."""
    rows = list(rows)
    if not rows:
        raise ValueError("Offline bootstrap plots require at least one result row.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    axis_names = tuple(payload.get("axes", dict.fromkeys(row["axis"] for row in rows)))
    profiles = tuple(dict.fromkeys(row["profile"] for row in rows))
    if not axis_names or any(axis not in _LABELS for axis in axis_names):
        raise ValueError("Offline bootstrap plots support N, B, and sigma axes.")
    outputs = []
    specifications = (
        ("coverage", "01_analytical_containment.pdf", 2,
         "Whole-line containment for the saved dataset"),
        ("radius", "02_radius_components.pdf", 1,
         "Transferred radius at the certified truth reference"),
        ("regret", "03_optimization_regret.pdf", 2,
         "Optimization regret and conditional certificate"),
    )
    with plt.rc_context(_STYLE):
        for metric, filename, panels, title in specifications:
            fig, axes = plt.subplots(
                panels * len(profiles), len(axis_names), squeeze=False,
                figsize=(6 * len(axis_names), 3.8 * panels * len(profiles)),
                constrained_layout=True,
            )
            fig.suptitle(title)
            for profile_index, profile in enumerate(profiles):
                for axis_index, axis in enumerate(axis_names):
                    group = sorted(
                        (row for row in rows if row["profile"] == profile and row["axis"] == axis),
                        key=lambda row: row["axis_value"],
                    )
                    if not group:
                        plt.close(fig)
                        raise ValueError(f"Missing result rows for profile {profile!r}, axis {axis!r}.")
                    x = _values(group, "axis_value")
                    ax = axes[panels * profile_index, axis_index]
                    ax.set_title(str(profile).replace("_", " "))
                    if metric == "coverage":
                        ax.plot(x, _values(group, "auxiliary_coverage_ratio"), marker="o",
                                label="Required / calibrated critical value")
                        ax.axhline(1, linestyle="--", color="C1", label="Auxiliary containment threshold")
                        ax.set_ylabel("Auxiliary radius ratio (upper certificate)")
                        status = axes[panels * profile_index + 1, axis_index]
                        # The separate status panel avoids calling a one-dataset
                        # boolean a sampling coverage probability.
                        covered = [(-1 if row.get("envelope_covered") is None
                                    else int(row["envelope_covered"])) for row in group]
                        status.plot(x, covered, linestyle="none", marker="o")
                        status.set_yticks([-1, 0, 1], ["Unresolved", "Fails", "Contains truth"])
                        status.set_ylim(-1.3, 1.3)
                        status.set_ylabel("Final-envelope containment on R")
                        _axis(status, axis, x)
                    elif metric == "radius":
                        for key, label in (
                            ("width_at_reference", "Bootstrap half-width w"),
                            ("discrepancy_at_reference", "Predictor / auxiliary discrepancy"),
                            ("radius_at_reference", "Total radius r"),
                        ):
                            ax.plot(x, _values(group, key), marker="o", label=label)
                        ax.set_ylabel("Radius component at truth reference")
                    else:
                        _interval(ax, x, group, "plugin_regret", "Fixed plug-in regret interval")
                        _interval(ax, x, group, "lcb_regret", "LCB regret interval")
                        bound = np.asarray([
                            row["lcb_regret_bound"] if row.get("regret_bound_applicable")
                            and row.get("lcb_regret_bound") is not None else np.nan
                            for row in group
                        ], dtype=float)
                        ax.plot(x, bound, linestyle="--", marker="s",
                                label="Applicable coverage-event bound")
                        ax.set_ylabel("True regret / certified upper bound")
                        status = axes[panels * profile_index + 1, axis_index]
                        states = [
                            2 if row.get("regret_bound_applicable") else
                            (1 if row.get("lcb_regret_upper") is not None else 0)
                            for row in group
                        ]
                        status.plot(x, states, linestyle="none", marker="o")
                        status.set_yticks([0, 1, 2], ["Solve uncertified", "Bound unavailable", "Bound applies"])
                        status.set_ylim(-0.3, 2.3)
                        status.set_ylabel("LCB certification status")
                        _axis(status, axis, x)
                    _axis(ax, axis, x)
                    ax.legend()
            path = destination / filename
            fig.savefig(path, format="pdf")
            plt.close(fig)
            outputs.append(path)
    return outputs
