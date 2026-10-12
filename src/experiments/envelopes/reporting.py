"""Replay uncertainty-envelope coverage and finite-location diagnostics as PDFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.provenance import file_record, write_provenance

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


def _comparison_case(project):
    from .study import _read_completed_case

    project = Path(project).resolve()
    summary_path = project / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (not summary.get("complete") or not summary.get("successful")
            or summary.get("requested_repetitions") != 1
            or len(summary["repetitions"]) != 1
            or len(summary["repetition_records"]) != 1):
        raise ValueError("Comparison requires one completed, successful realization per project.")
    record = summary["repetition_records"][0]
    if file_record(record["path"]) != record:
        raise ValueError("Saved repetition summary hash/size mismatch.")
    row = _read_completed_case(Path(record["path"]))
    if (row != summary["repetitions"][0]
            or row["contract_sha256"] != summary["contract_sha256"]
            or row["status"] == "fit_failed" or row.get("metrics") is None
            or row["analytical"] != summary["coverage"]["analytical"]):
        raise ValueError("Comparison project summary differs from its completed realization.")
    contract_path = project / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract_payload = {key: value for key, value in contract.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(contract_payload, sort_keys=True).encode()).hexdigest()
    if (digest != contract["sha256"] or digest != summary["contract_sha256"]
            or summary["manifest"] != contract["manifest"]):
        raise ValueError("Comparison manifest or contract identity differs from the saved run.")
    _outcomes(summary)
    point_record = row["artifacts"]["points"]
    _read_points(project, point_record)
    with np.load(point_record["path"], allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key], dtype=float) for key in
                  ("X", "actions", "center", "truth", "radius", "excess", "weights")}
        if "component_auxiliary" in saved:
            arrays["auxiliary"] = np.asarray(saved["component_auxiliary"], dtype=float)
    size = len(arrays["actions"])
    if (arrays["X"].ndim != 2 or arrays["X"].shape[0] != size
            or any(value.shape != (size,) for key, value in arrays.items() if key != "X")
            or not all(np.isfinite(value).all() for value in arrays.values())
            or not np.allclose(arrays["excess"], arrays["radius"]
                               - np.abs(arrays["center"]-arrays["truth"]),
                               rtol=1e-12, atol=1e-12)):
        raise ValueError("Invalid saved envelope slice arrays.")
    with np.load(row["artifacts"]["calibration"]["path"], allow_pickle=False) as calibration:
        logged_contexts = np.asarray(calibration["X"], dtype=float)
    target = summary["target"]
    if (logged_contexts.ndim != 2 or len(logged_contexts) == 0
            or target["context_count"] != len(logged_contexts)
            or target["context_dim"] != logged_contexts.shape[1]
            or target["context_dim"] != arrays["X"].shape[1]
            or target["kind"] != contract["manifest"]["target"]["actions"]["type"]
            or size % len(logged_contexts)
            or not np.array_equal(arrays["X"], np.repeat(logged_contexts, size//len(logged_contexts), axis=0))):
        raise ValueError("Comparison target differs from saved calibration/diagnostic contexts.")
    method = summary["manifest"]["envelope"]
    if method["type"] == "constant_radius":
        label = f"Constant radius r = {method['radius']:g}"
    elif method["type"] == "linear_known_variance_transfer":
        multiplier = method.get("width_multiplier", 1.)
        label = ("Writeup transfer radius" if multiplier == 1 else
                 f"Linear transfer (width multiplier {multiplier:g})")
    else:
        label = summary["name"]
    provenance = {
        "name": summary["name"], "label": label,
        "summary": file_record(summary_path), "repetition": record,
        "completion": file_record(Path(record["path"]).with_name("completion.json")),
        "artifacts": row["artifacts"], "contract_sha256": summary["contract_sha256"],
        "manifest": summary["manifest"], "target": summary["target"],
        "coverage": summary["coverage"], "containment": row["containment"],
        "metrics": row["metrics"], "contract": file_record(contract_path),
    }
    return summary, arrays, provenance


def _comparison_slices(summary, arrays):
    count = summary["target"]["context_count"]
    if count < 3 or len(arrays["actions"]) % count:
        raise ValueError("Comparison requires at least three context-major diagnostic blocks.")
    actions = arrays["actions"].reshape(count, -1)
    contexts = arrays["X"].reshape(count, actions.shape[1], -1)
    if (not np.all(actions == actions[0]) or np.any(np.diff(actions[0]) <= 0)
            or not np.all(contexts == contexts[:, :1, :])):
        raise ValueError("Comparison requires shared increasing actions within fixed context blocks.")
    chosen = (actions[0] >= -2) & (actions[0] <= 2)
    window = actions[0, chosen]
    if len(window) < 2 or window[0] != -2 or window[-1] != 2:
        raise ValueError("Saved diagnostics must include the comparison window endpoints -2 and 2.")
    return window, contexts[:3, 0, :], {
        key: value.reshape(count, -1)[:3, chosen]
        for key, value in arrays.items() if key != "X"
    }


def _envelope_comparison_figure(cases, slices):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey="row", constrained_layout=True)
    fig.suptitle("Saved envelope slices at contexts 0, 1, and 2")
    for column, ((_, _, provenance), (actions, contexts, values)) in enumerate(zip(cases, slices)):
        for index in range(3):
            ax = axes[index, column]
            ax.plot(actions, values["truth"][index], label="True f")
            center = values["center"][index]
            line, = ax.plot(actions, center, label="Frozen predictor")
            radius = values["radius"][index]
            ax.fill_between(actions, center-radius, center+radius,
                            color=line.get_color(), alpha=.2, label="Predictor ± radius")
            if "auxiliary" in values:
                ax.plot(actions, values["auxiliary"][index], label="Auxiliary fit")
            coordinates = ", ".join(f"{value:.2f}" for value in contexts[index])
            ax.set_title(f"{provenance['label']}\nContext {index}: x = ({coordinates})")
            ax.set_xlabel("Action a")
            ax.set_ylabel("Response value")
            ax.set_xlim(-2, 2)
            ax.legend()
    return fig


def _radius_comparison_figure(cases, slices):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row", constrained_layout=True)
    fig.suptitle("Radius and signed excess at contexts 0, 1, and 2")
    for column, ((_, _, provenance), (actions, _, values)) in enumerate(zip(cases, slices)):
        for row, key in enumerate(("radius", "excess")):
            ax = axes[row, column]
            for index in range(3):
                ax.plot(actions, values[key][index], label=f"Context {index}")
            if key == "excess":
                ax.axhline(0, color="black", linestyle="--", label="Zero excess")
            ax.set_title(provenance["label"])
            ax.set_xlabel("Action a")
            ax.set_ylabel("Radius r(x, a)" if key == "radius" else
                          r"Signed excess $r(x,a)-|\hat f(x,a)-f(x,a)|$")
            ax.set_xlim(-2, 2)
            ax.legend()
    return fig


def write_comparison(projects, destination: Path) -> list[Path]:
    """Compare two saved single realizations at contexts 0–2 on actions [-2, 2]."""
    projects = list(projects)
    if len(projects) != 2:
        raise ValueError("Comparison requires exactly two saved projects.")
    cases = [_comparison_case(project) for project in projects]
    for key in ("X", "actions", "center", "truth", "weights"):
        if not np.array_equal(cases[0][1][key], cases[1][1][key]):
            raise ValueError(f"Comparison requires identical saved diagnostic {key}.")
    slices = [_comparison_slices(summary, arrays) for summary, arrays, _ in cases]
    destination = Path(destination)
    if any(destination.resolve() == Path(project).resolve() for project in projects):
        raise ValueError("Use a separate comparison output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "01_envelope_slices.pdf", destination / "02_radius_excess.pdf"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        for create, path in zip((_envelope_comparison_figure, _radius_comparison_figure), paths):
            figure = create(cases, slices)
            try:
                figure.savefig(path, format="pdf")
            finally:
                plt.close(figure)
    write_provenance(destination / "comparison.json", {
        "projects": [provenance for _, _, provenance in cases],
        "context_indices": [0, 1, 2], "context_values": slices[0][1],
        "action_window": [-2., 2.], "source": file_record(__file__),
        "plots": [file_record(path) for path in paths],
        "semantics": "Curves join saved finite diagnostic values for display. "
                     "No refitting, action selection, or continuum coverage inference is performed. "
                     "Each saved analytical probability statement, realized containment decision, "
                     "target domain, and finite diagnostic metric retains its original meaning.",
    })
    return paths
