"""Replay clean comparisons of analytical and wild-bootstrap envelope widths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

from .diagnostics import _slice_indices
from .reporting import _STYLE

_METHODS = (("analytical", "analytical-radius", "C0"),
            ("wild_bootstrap", "bootstrap", "C1"))


def _rows(summary, sweep, delta, method):
    baseline = summary["manifest"]["baseline"]
    fixed = "B" if sweep == "n" else "n"
    selected = [row for row in summary["metrics"]
                if row["method"] == method and row["delta"] == delta
                and row[fixed] == baseline[fixed]]
    selected.sort(key=lambda row: row[sweep])
    expected = sorted(summary["manifest"][f"{sweep}_values"])
    if [row[sweep] for row in selected] != expected:
        raise ValueError(f"Incomplete or duplicate {method} {sweep} sweep at delta={delta}.")
    return selected


def _panels(deltas):
    fig, axes = plt.subplots(1, len(deltas), figsize=(4.7*len(deltas), 4.8),
                             sharey=True, squeeze=False, constrained_layout=True)
    for ax, delta in zip(axes[0], deltas):
        ax.set_title(rf"$\delta={delta:g}$")
    return fig, axes[0]


def _sweep_axis(ax, values, sweep):
    ax.set_xscale("log")
    ax.set_xticks(sorted(values))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Calibration size $n$" if sweep == "n" else "Bootstrap draws $B$")


def _coverage_figure(summary, sweep, event):
    p = summary["manifest"]
    fig, axes = _panels(p["deltas"])
    title = (r"$\widehat f$ coverage" if event == "auxiliary"
             else r"Transferred-envelope coverage: $\widehat f\pm r_\delta$")
    fig.suptitle(title)
    lower_limits = []
    for ax, delta in zip(axes, p["deltas"]):
        for method, label, color in _METHODS:
            rows = _rows(summary, sweep, delta, method)
            coverage = np.array([row[f"{event}_coverage"] for row in rows])
            low = np.array([row[f"{event}_lower"] for row in rows])
            high = np.array([row[f"{event}_upper"] for row in rows])
            if (not np.isfinite([coverage, low, high]).all()
                    or np.any(low < 0) or np.any(high > 1)
                    or np.any(low > coverage) or np.any(high < coverage)):
                raise ValueError("Coverage estimates and intervals must be ordered probabilities.")
            lower_limits.extend(low)
            ax.errorbar([row[sweep] for row in rows], coverage,
                        yerr=[coverage-low, high-coverage], marker="o", color=color,
                        capsize=3, elinewidth=.7, capthick=.7, label=label)
        ax.axhline(1-delta, color="0.5", linestyle="--", label=r"Target $1-\delta$")
        _sweep_axis(ax, p[f"{sweep}_values"], sweep)
        ax.set_ylabel("Simultaneous coverage")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    # Shared bounds retain small-sample failures while making nominal coverage readable.
    axes[0].set_ylim(max(0., min(.8, min(lower_limits)-.04)), 1.015)
    axes[0].legend()
    fixed = (rf"$B={p['baseline']['B']}$" if sweep == "n"
             else rf"$n={p['baseline']['n']}$")
    fig.supxlabel(
        f"{p['repetitions']:,} repeated datasets at each frozen design; {fixed}; 95% Wilson intervals."
        "\n" + r"Each success covers all calibration contexts and every $a\in\mathbb{R}$.",
        fontsize=10,
    )
    return fig


def _width_figure(summary, example):
    p = summary["manifest"]
    fig, axes = _panels(p["deltas"])
    fig.suptitle(r"$w_\delta$ (half width)")
    for ax, delta in zip(axes, p["deltas"]):
        for method, label, color in _METHODS:
            rows = _rows(summary, "n", delta, method)
            width = np.array([row["mean_width"] for row in rows])
            if not np.isfinite(width).all() or np.any(width < 0):
                raise ValueError("Mean auxiliary widths must be finite and nonnegative.")
            ax.plot([row["n"] for row in rows], width,
                    marker="o", color=color, label=label)
        _sweep_axis(ax, p["n_values"], "n")
        ax.set_ylabel(r"Mean half-width $\overline{w}_\delta$")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    actions = np.asarray(example["actions_grid"])
    fig.supxlabel(
        rf"$B={p['baseline']['B']}$; means over {p['repetitions']:,} datasets, calibration contexts,"
        rf" and {len(actions)} equally weighted actions in $[{actions[0]:g},{actions[-1]:g}]$."
        "\n" + r"The display-window mean summarizes width; coverage is checked over all real actions.",
        fontsize=10,
    )
    return fig


def _validate_example(snapshot, example):
    X = np.asarray(example["X"])
    actions = np.asarray(example["actions_grid"])
    if (X.ndim != 2 or len(X) == 0 or actions.ndim != 1 or len(actions) == 0
            or not np.isfinite(X).all() or not np.isfinite(actions).all()
            or np.any(np.diff(actions) <= 0)
            or not np.array_equal(X, np.asarray(snapshot["X"])[:len(X)])):
        raise ValueError("Example must use saved calibration contexts and ordered diagnostic actions.")
    for key in ("truth", "predictor", "auxiliary", "analytical_width", "bootstrap_width", "sigma_grid"):
        values = np.asarray(example[key])
        if values.shape != (len(X), len(actions)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved example array: {key}.")
        if key.endswith("width") and np.any(values < 0):
            raise ValueError("Auxiliary widths must be nonnegative.")
    return _slice_indices(X)


def _slice_panels(example, indices):
    fig, axes = plt.subplots(1, len(indices), figsize=(4.7*len(indices), 4.8),
                             sharey=True, squeeze=False, constrained_layout=True)
    ranks = ("Central", "Median norm", "Outer") if len(indices) == 3 else ("Central", "Outer")
    for ax, index, rank in zip(axes[0], indices, ranks):
        norm = np.linalg.norm(example["X"][index])
        ax.set_title(rank + rf": $\|x_i\|_2={norm:.2f}$")
        ax.set_xlabel("Action $a$")
    return fig, axes[0]


def _slices_figure(summary, example, indices, method, label):
    p = summary["manifest"]
    fig, axes = _slice_panels(example, indices)
    fig.suptitle(label + rf" envelope: $\delta={p['baseline']['delta']:g}$")
    a = np.asarray(example["actions_grid"])
    key = "analytical_width" if method == "analytical" else "bootstrap_width"
    for ax, index in zip(axes, indices):
        truth, predictor, auxiliary, width = (np.asarray(example[k])[index]
                                              for k in ("truth", "predictor", "auxiliary", key))
        radius = np.abs(predictor-auxiliary)+width
        ax.plot(a, truth, color="C0", label=r"$f(x_i,a)$")
        ax.plot(a, predictor, "--", color="C1", label=r"$\widehat f(x_i,a)$")
        ax.plot(a, auxiliary, ":", color="C2", label=r"$\widetilde f(x_i,a)$")
        ax.fill_between(a, predictor-radius, predictor+radius, color="C1", alpha=.2,
                        label=r"$\widehat f(x_i,a)\pm r_\delta(x_i,a)$")
        ax.set_ylabel(r"Response $f(x_i,a)$")
    axes[0].legend()
    fig.supxlabel(
        rf"First saved dataset; $n={p['baseline']['n']}$, $B={p['baseline']['B']}$;"
        " contexts selected by smallest, median, and largest norm."
        "\nContext norm measures centrality; these slices illustrate one realization.",
        fontsize=10,
    )
    return fig


def _noise_figure(summary, example, indices):
    p = summary["manifest"]
    fig, axes = _slice_panels(example, indices)
    fig.suptitle("Bounded noise standard deviation")
    a = np.asarray(example["actions_grid"])
    for ax, index in zip(axes, indices):
        ax.plot(a, np.asarray(example["sigma_grid"])[index], label=r"$\sigma(x_i,a)$")
        ax.axhline(p["noise"]["sigma_max"], color="C1", linestyle="--", label=r"Cap $\sigma_{\max}$")
        ax.set_ylabel(r"Noise standard deviation $\sigma(x_i,a)$")
        ax.set_ylim(bottom=0)
    axes[0].legend()
    fig.supxlabel(
        "One fixed GP volatility profile; independent Gaussian observation errors conditional on that profile."
        "\nThe known variance cap holds for all inputs; it is not estimated from the displayed actions.",
        fontsize=10,
    )
    return fig


def write_reports(summary, snapshot, example, destination) -> list[Path]:
    """Render saved joint coverage, auxiliary widths, and example context slices as PDFs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    indices = _validate_example(snapshot, example)
    paths = []
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figures = [
            ("01_auxiliary_coverage_vs_n", _coverage_figure(summary, "n", "auxiliary")),
            ("02_auxiliary_coverage_vs_B", _coverage_figure(summary, "B", "auxiliary")),
            ("03_mean_width_vs_n", _width_figure(summary, example)),
            ("04_transferred_coverage_vs_n", _coverage_figure(summary, "n", "transferred")),
            ("05_analytical_context_slices", _slices_figure(summary, example, indices, "analytical", "analytical-radius")),
            ("06_wild_bootstrap_context_slices", _slices_figure(summary, example, indices, "wild_bootstrap", "bootstrap")),
            ("07_noise_profile", _noise_figure(summary, example, indices)),
        ]
        for name, fig in figures:
            path = destination / f"{name}.pdf"
            try:
                fig.savefig(path, format="pdf")
            finally:
                plt.close(fig)
            paths.append(path)
    return paths
