"""Replay uncertainty-envelope coverage and finite-location diagnostics as PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.provenance import file_record

_STATUSES = ("covered", "violated", "unknown", "fit_failed")
_STYLE = {
    "figure.titlesize": 16,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
}


def _outcomes(summary):
    records = summary["repetitions"]
    counts = dict.fromkeys(_STATUSES, 0)
    for row in records:
        if row["status"] not in counts:
            raise ValueError(f"Unknown containment status: {row['status']}")
        counts[row["status"]] += 1
    empirical = summary["coverage"].get("empirical")
    missing = summary.get("requested_repetitions", len(records)) - len(records)
    if missing < 0:
        raise ValueError("More saved repetitions than requested.")
    if empirical is not None:
        missing = empirical["requested"] - len(records)
        if (missing < 0 or empirical["counts"] != counts
                or empirical["missing"] != missing):
            raise ValueError("Empirical coverage counts disagree with saved repetitions.")
        unresolved = missing + counts["unknown"] + counts["fit_failed"]
        if unresolved and (empirical["estimate"] is not None
                           or empirical["interval"] is not None):
            raise ValueError("Unresolved repetitions cannot produce a coverage estimate.")
    return counts, missing


def _read_points(project, record):
    source = Path(record["path"])
    if not source.is_absolute():
        source = project / source
    actual = file_record(source)
    if any(actual[key] != record[key] for key in ("sha256", "bytes")):
        raise ValueError(f"Diagnostic artifact hash/size mismatch: {source}")
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key], dtype=float)
                  for key in ("radius", "excess", "weights")}
    radius, excess, weights = (arrays[key] for key in ("radius", "excess", "weights"))
    if (radius.ndim != 1 or len(radius) == 0 or excess.shape != radius.shape
            or weights.shape != radius.shape
            or not all(np.isfinite(value).all() for value in arrays.values())
            or np.any(radius < 0) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError(f"Invalid saved pointwise radius/excess/weights: {source}")
    arrays["weights"] = weights / weights.sum()
    return arrays


def _metric_arrays(project, summary):
    cases = []
    for row in summary["repetitions"]:
        if row["status"] == "fit_failed" or row.get("metrics") is None:
            continue
        record = row.get("artifacts", {}).get("points")
        if record is None:
            raise ValueError("Saved metric summaries require a pointwise artifact.")
        cases.append(_read_points(project, record))
    if not cases:
        return None, 0
    pooled = {key: np.concatenate([case[key] for case in cases])
              for key in ("radius", "excess", "weights")}
    # Every fitted repetition contributes equal total diagnostic mass.
    pooled["weights"] /= len(cases)
    return pooled, len(cases)


def _coverage_figure(summary, counts, missing):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    probability, outcomes = axes
    target = summary["target"]
    scope = "all real actions" if target["kind"] == "real_line" else "the declared finite action set"
    fig.suptitle(f"Coverage at {target['context_count']} contexts over {scope}")
    analytical, empirical = (summary["coverage"].get(key) for key in ("analytical", "empirical"))
    labels = []
    if analytical is not None:
        if analytical["kind"] == "lower_bound":
            label, marker = "Analytical\nlower bound", "^"
        elif analytical["kind"] == "exact_probability":
            label, marker = "Analytical\nexact probability", "o"
        else:
            raise ValueError("Analytical coverage requires an explicit probability kind.")
        value = analytical["value"]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Analytical probability must lie in [0, 1].")
        probability.plot(len(labels), value, marker=marker, linestyle="none")
        labels.append(label)
    if empirical is not None:
        estimate, interval = empirical["estimate"], empirical["interval"]
        if estimate is None:
            labels.append("Empirical\nunavailable")
        else:
            low, high = interval
            if (empirical["interval_method"] != "Wilson"
                    or not 0 <= low <= estimate <= high <= 1
                    or not np.isclose(estimate, counts["covered"] / empirical["requested"])):
                raise ValueError("Invalid empirical probability or Wilson interval.")
            probability.errorbar(len(labels), estimate,
                                 yerr=[[estimate-low], [high-estimate]],
                                 marker="o", linestyle="none", capsize=3,
                                 label=f"{100*empirical['confidence']:g}% Wilson interval")
            labels.append("Empirical\nestimate")
            probability.legend()
    if not labels:
        labels = ["No probability\nstatement supplied"]
    probability.set_xticks(range(len(labels)), labels)
    probability.set_xlim(-.5, len(labels)-.5)
    probability.set_ylim(-.03, 1.03)
    probability.set_xlabel("Coverage statement")
    probability.set_ylabel("Simultaneous coverage probability")
    probability.set_title("Probability statements")
    bars = outcomes.bar(["Covered", "Violated", "Unknown", "Fit failed", "Missing"],
                        [*(counts[key] for key in _STATUSES), missing])
    outcomes.bar_label(bars, fontsize=10)
    outcomes.set_ylim(0, max(1, max(bar.get_height() for bar in bars))*1.15)
    outcomes.set_xlabel("Containment outcome")
    outcomes.set_ylabel("Repetitions")
    outcomes.set_title("Realized outcomes")
    return fig


def _tightness_figure(arrays, fitted, total):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    title = "Radius and excess over finite diagnostic locations"
    if fitted < total:
        title += f"\nConditional on {fitted}/{total} successful metric evaluations"
    else:
        title += f"\n{fitted} fitted repetition{'s' if fitted != 1 else ''}; equal repetition weight"
    fig.suptitle(title)
    for ax, key, label in zip(axes, ("radius", "excess"),
                              ("Envelope radius", "Signed excess radius")):
        if arrays is not None:
            ax.hist(arrays[key], bins=30, weights=arrays["weights"])
        ax.set_title(label if arrays is not None else f"{label}: no available evaluations")
        ax.set_xlabel("Radius r(x, a)" if key == "radius" else r"Excess $r(x,a)-|\hat f(x,a)-f(x,a)|$")
        ax.set_ylabel("Share of diagnostic weight")
    return fig


def write_reports(project: Path) -> list[Path]:
    """Write two vector PDFs from verified saved diagnostics, without refitting."""
    project = Path(project)
    summary = json.loads((project / "summary.json").read_text(encoding="utf-8"))
    if "repetition_records" in summary:
        saved_rows = []
        for record in summary["repetition_records"]:
            if file_record(record["path"]) != record:
                raise ValueError("Saved repetition summary hash/size mismatch.")
            saved_rows.append(json.loads(Path(record["path"]).read_text()))
        if saved_rows != summary["repetitions"]:
            raise ValueError("Project summary differs from saved repetition results.")
    counts, missing = _outcomes(summary)
    arrays, fitted = _metric_arrays(project, summary)
    destination = project / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "01_coverage.pdf", destination / "02_radius_excess.pdf"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figures = (_coverage_figure(summary, counts, missing),
                   _tightness_figure(arrays, fitted,
                                     len(summary["repetitions"])+missing))
        for figure, path in zip(figures, paths):
            try:
                figure.savefig(path, format="pdf")
            finally:
                plt.close(figure)
    return paths
