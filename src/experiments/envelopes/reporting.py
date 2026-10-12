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
        for key in ("auxiliary", "width", "discrepancy"):
            if f"component_{key}" in saved:
                arrays[key] = np.asarray(saved[f"component_{key}"], dtype=float)
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
        label = f"constant radius = {method['radius']:g}"
    elif method["type"] == "linear_known_variance_transfer":
        multiplier = method.get("width_multiplier", 1.)
        label = ("analytical-from-linear-f radius" if multiplier == 1 else
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


def _aligned_comparison(summary, arrays):
    count = summary["target"]["context_count"]
    if count < 1 or len(arrays["actions"]) % count:
        raise ValueError("Comparison requires context-major diagnostic blocks.")
    actions = arrays["actions"].reshape(count, -1)
    contexts = arrays["X"].reshape(count, actions.shape[1], -1)
    if (not np.all(actions == actions[0]) or np.any(np.diff(actions[0]) <= 0)
            or not np.all(contexts == contexts[:, :1, :])):
        raise ValueError("Comparison requires shared increasing actions within fixed context blocks.")
    spec = summary["manifest"]["problem"]["truth"]
    if spec["type"] != "linear":
        raise ValueError("Context adjustment requires a declared additive linear truth.")
    coefficients = np.asarray(spec["coefficients"], dtype=float)
    X = contexts[:, 0, :]
    if coefficients.shape != (X.shape[1]+2,) or not np.isfinite(coefficients).all():
        raise ValueError("Context adjustment requires dimension-matched finite truth coefficients.")
    g = coefficients[0] + X @ coefficients[1:-1]
    if not np.allclose(arrays["truth"].reshape(count, -1),
                       g[:, None] + coefficients[-1]*actions, rtol=1e-12, atol=1e-12):
        raise ValueError("Saved truth does not match the declared additive linear truth.")
    chosen = (actions[0] >= -2) & (actions[0] <= 2)
    window = actions[0, chosen]
    if len(window) < 2 or window[0] != -2 or window[-1] != 2:
        raise ValueError("Saved diagnostics must include the comparison window endpoints -2 and 2.")
    center = arrays["center"].reshape(count, -1)[:, chosen] - g[:, None]
    radius = arrays["radius"].reshape(count, -1)[:, chosen]
    # Intersect response intervals across every context. These endpoint
    # reductions do not select an action, optimizer solution, or context to plot.
    lower = np.maximum.reduce(center-radius, axis=0)
    upper = np.minimum.reduce(center+radius, axis=0)
    truth = coefficients[-1]*window
    empty = lower > upper
    return {"actions": window, "contexts": X, "truth": truth,
            "mean_predictor": center.mean(axis=0), "lower": lower, "upper": upper,
            "empty": empty, "covered": (~empty) & (lower <= truth) & (truth <= upper),
            "g": g, "coefficients": coefficients}


def _context_formula(coefficients):
    terms = []
    if coefficients[0] != 0:
        terms.append(f"{coefficients[0]:g}")
    for index, coefficient in enumerate(coefficients[1:-1], start=1):
        if coefficient == 0:
            continue
        magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient):g}"
        term = magnitude + f"x_{index}"
        terms.append(("-" if coefficient < 0 else ("+" if terms else "")) + term)
    return "g(x)=" + ("".join(terms) if terms else "0")


def _coverage_comparison_figure(cases, aligned):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, constrained_layout=True)
    count = cases[0][0]["target"]["context_count"]
    fig.suptitle(f"Simultaneous envelope containment across all {count} contexts")
    formula = _context_formula(aligned[0]["coefficients"])
    fig.supxlabel(
        "Display adjustment only: subtract $" + formula + "$ from truth, predictions, and envelope endpoints.\n"
        "Radii and coverage are unchanged. Shading is the common overlap, not an individual radius.\n"
        "Blue inside the overlap means every context is covered at that action; drawn curves join sampled actions.",
        fontsize=10,
    )
    for ax, (summary, _, provenance), values in zip(axes, cases, aligned):
        a = values["actions"]
        ax.plot(a, values["truth"], label="True f, context-adjusted")
        line, = ax.plot(a, values["mean_predictor"], label="Mean frozen predictor, context-adjusted")
        ax.fill_between(a, values["lower"], values["upper"], where=~values["empty"],
                        color=line.get_color(), alpha=.2, label=f"Common envelope across {count} contexts")
        outside = ~values["covered"] & ~values["empty"]
        if np.any(outside):
            ax.plot(a[outside], values["truth"][outside], "o", color="C3", label="At least one context not covered")
        if np.any(values["empty"]):
            ax.plot(a[values["empty"]], values["truth"][values["empty"]], "x", color="C3",
                    label="Empty common overlap")
        containment = provenance["containment"]
        status = {"covered": "verified", "violated": "fails", "unknown": "unknown"}[containment["status"]]
        domain = "All real actions" if containment["domain"] == "real_line" else "Declared finite target"
        statement = summary["coverage"]["analytical"]
        probability = "No analytical probability guarantee supplied"
        if statement is not None:
            relation = "at least" if statement["kind"] == "lower_bound" else "exactly"
            probability = f"Coverage probability: {relation} {100*statement['value']:g}%"
        ax.set_title(f"{provenance['label']}\n{domain}: {status}\n{probability}")
        metrics = provenance["metrics"]
        ax.legend(title=f"Mean radius: {metrics['radius_mean']:.3f}; mean excess: {metrics['excess_mean']:.3f}")
        ax.set_xlabel("Action a")
        ax.set_ylabel("Context-adjusted response value")
        ax.set_xlim(-2, 2)
    return fig


def _collapsed_tightness(aligned):
    return {"half_span": (aligned["upper"]-aligned["lower"])/2,
            "excess": np.minimum(aligned["truth"]-aligned["lower"],
                                 aligned["upper"]-aligned["truth"])}


def _radius_excess_comparison_figure(cases, aligned, tightness):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    count = len(aligned[0]["contexts"])
    fig.suptitle(f"Radius and signed excess after collapsing all {count} contexts")
    for (_, _, provenance), values, metrics in zip(cases, aligned, tightness):
        half_width = np.ma.masked_where(values["empty"], metrics["half_span"])
        axes[0].plot(values["actions"], half_width, label=provenance["label"])
        axes[1].plot(values["actions"], metrics["excess"], label=provenance["label"])
    axes[0].set_title("Half-width of the common envelope in plot 1")
    axes[0].set_ylabel("Common-envelope half-width (U - L) / 2")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Signed excess relative to the common truth")
    axes[1].set_ylabel("Signed excess min(f - L, U - f), context-adjusted")
    axes[1].axhline(0, color="0.5", linestyle="--", label="Zero excess: coverage boundary")
    for ax in axes:
        ax.set_xlabel("Action a")
        ax.set_xlim(-2, 2)
        ax.legend()
    fig.supxlabel(
        "Same context adjustment and interval intersection as plot 1; one curve per construction.\n"
        "Common half-width is not an individual radius. Signed excess equals the smallest excess across all contexts.\n"
        "Negative excess means at least one context is not covered; half-width is omitted for empty intersections.",
        fontsize=10,
    )
    return fig


def write_comparison(projects, destination: Path) -> list[Path]:
    """Plot common envelopes and their radius/excess diagnostics after removing known context shifts."""
    projects = list(projects)
    if len(projects) != 2:
        raise ValueError("Comparison requires exactly two saved projects.")
    cases = [_comparison_case(project) for project in projects]
    for key in ("X", "actions", "center", "truth", "weights"):
        if not np.array_equal(cases[0][1][key], cases[1][1][key]):
            raise ValueError(f"Comparison requires identical saved diagnostic {key}.")
    if cases[0][0]["manifest"]["problem"]["truth"] != cases[1][0]["manifest"]["problem"]["truth"]:
        raise ValueError("Comparison requires the same declared truth for context adjustment.")
    aligned = [_aligned_comparison(summary, arrays) for summary, arrays, _ in cases]
    tightness = [_collapsed_tightness(values) for values in aligned]
    destination = Path(destination)
    if any(destination.resolve() == Path(project).resolve() for project in projects):
        raise ValueError("Use a separate comparison output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "01_coverage_comparison.pdf", destination / "02_radius_excess.pdf"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figures = (_coverage_comparison_figure(cases, aligned),
                   _radius_excess_comparison_figure(cases, aligned, tightness))
        for figure, path in zip(figures, paths):
            try:
                figure.savefig(path, format="pdf")
            finally:
                plt.close(figure)
    arrays_path = destination / "adjusted_coverage.npz"
    np.savez_compressed(arrays_path,
                        **{key: aligned[0][key] for key in ("actions", "contexts", "truth", "mean_predictor", "g")},
                        **{key: np.stack([row[key] for row in aligned])
                           for key in ("lower", "upper", "empty", "covered")})
    tightness_path = destination / "collapsed_tightness.npz"
    np.savez_compressed(tightness_path, actions=aligned[0]["actions"],
                        empty=np.stack([values["empty"] for values in aligned]),
                        **{key: np.stack([metrics[key] for metrics in tightness])
                           for key in ("half_span", "excess")})
    write_provenance(destination / "comparison.json", {
        "projects": [provenance for _, _, provenance in cases],
        "context_count": len(aligned[0]["contexts"]), "context_values": aligned[0]["contexts"],
        "context_adjustment": {"formula": _context_formula(aligned[0]["coefficients"]),
                               "truth_coefficients": aligned[0]["coefficients"],
                               "source": "Known synthetic truth, not estimated from observations."},
        "sampled_action_coverage": [bool(row["covered"].all()) for row in aligned],
        "action_window": [-2., 2.], "source": file_record(__file__),
        "plots": [file_record(path) for path in paths], "aligned_arrays": file_record(arrays_path),
        "collapsed_tightness": file_record(tightness_path),
        "semantics": "Subtract the same context-only truth contribution from truth, predictions, and interval endpoints. "
                     "Shading intersects the adjusted intervals across every saved context; empty intersections are not shaded. "
                     "The mean adjusted predictor is for orientation only. Radii and original finite diagnostic metrics are unchanged. "
                     "The radius/excess PDF uses this same intersection: half-width (U-L)/2 is masked when empty, "
                     "while signed excess min(truth-L,U-truth) is preserved even when negative. "
                     "This excess equals the smallest original pointwise excess across contexts; it is not a context average. "
                     "Curves join saved finite diagnostic values. Floating-point alignment is not a new continuous certificate. "
                     "Saved full-domain containment and analytical probability statements are displayed separately, without refitting or optimization.",
    })
    return paths


def _calibration_action_view(summary, arrays, data):
    if summary["manifest"]["envelope"]["type"] != "linear_known_variance_transfer":
        raise ValueError("Action-support display requires the saved linear transfer construction.")
    view = _aligned_comparison(summary, arrays)
    if not np.array_equal(view["contexts"], data.X):
        raise ValueError("Calibration contexts must match the saved diagnostic contexts.")
    if not {"width", "discrepancy", "auxiliary"} <= arrays.keys():
        raise ValueError("Action-support display requires saved width and discrepancy components.")
    if (np.any(arrays["width"] < 0) or np.any(arrays["discrepancy"] < 0)
            or not np.allclose(arrays["radius"], arrays["width"]+arrays["discrepancy"],
                               rtol=1e-12, atol=1e-12)
            or not np.allclose(arrays["discrepancy"], np.abs(arrays["center"]-arrays["auxiliary"]),
                               rtol=1e-12, atol=1e-12)):
        raise ValueError("Saved radius components disagree with the linear transfer construction.")
    count = len(data.X)
    diagnostic_actions = arrays["actions"].reshape(count, -1)[0]
    window = (diagnostic_actions >= -2) & (diagnostic_actions <= 2)
    view.update({f"mean_{key}": arrays[key].reshape(count, -1)[:, window].mean(axis=0)
                 for key in ("radius", "width", "discrepancy")})
    view["adjusted_responses"] = data.y-view["g"]
    view["observed_in_window"] = (data.actions >= -2) & (data.actions <= 2)
    view["histogram_counts"], view["histogram_edges"] = np.histogram(data.actions, bins=np.linspace(-2, 2, 9))
    return view


def _calibration_figure(data, view):
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    fig.suptitle("Calibration action draws and the analytical-from-linear-f radius")
    a = view["actions"]
    observed = view["observed_in_window"]
    axes[0].plot(a, view["truth"], label="True f, context-adjusted")
    predictor, = axes[0].plot(a, view["mean_predictor"], label="Mean frozen predictor, context-adjusted")
    axes[0].fill_between(a, view["lower"], view["upper"], where=~view["empty"],
                         color=predictor.get_color(), alpha=.2, label="Common envelope from plot 1")
    axes[0].scatter(data.actions[observed], view["adjusted_responses"][observed],
                    color="C2", s=20, label="Observed responses, context-adjusted")
    axes[0].set_title("Analytical envelope and the observed calibration responses")
    axes[0].set_ylabel("Context-adjusted response")
    axes[0].legend()
    axes[1].plot(a, view["mean_radius"], label="Total radius r = discrepancy + w")
    axes[1].plot(a, view["mean_width"], label="Analytical uncertainty width w")
    axes[1].plot(a, view["mean_discrepancy"], label="Predictor–OLS discrepancy")
    axes[1].set_title(f"Radius components averaged across all {len(data.X)} contexts")
    axes[1].set_ylabel("Mean radius component")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()
    edges = view["histogram_edges"]
    axes[2].stairs(view["histogram_counts"], edges, fill=True, alpha=.4,
                   label="Calibration action counts (bin width 0.5)")
    axes[2].plot(data.actions[observed], np.zeros(observed.sum()), "|",
                  label="One tick per observed action")
    below, above = int((data.actions < -2).sum()), int((data.actions > 2).sum())
    axes[2].set_title(f"Action draws: {observed.sum()} of {len(data.y)} in the displayed window; "
                      f"{below} below -2, {above} above 2")
    axes[2].set_ylabel("Calibration observations per bin")
    axes[2].set_ylim(bottom=0)
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Action a (same window as plot 1)")
        ax.set_xlim(-2, 2)
    fig.supxlabel(
        "Display only: subtract $" + _context_formula(view["coefficients"]) + "$ from responses and model values.\n"
        "The shaded band targets the true mean response, not noisy observations; its common overlap is not the mean radius.\n"
        "Width w uses the full context–action design; histogram density alone does not determine it.", fontsize=10,
    )
    return fig


def write_calibration_plot(project: Path, destination: Path) -> Path:
    """Align saved calibration action counts, radius components, and the common analytical envelope."""
    from .core import CalibrationData

    summary, arrays, provenance = _comparison_case(project)
    artifacts = summary["repetitions"][0]["artifacts"]
    with np.load(artifacts["calibration"]["path"], allow_pickle=False) as saved:
        data = CalibrationData(saved["X"], saved["actions"], saved["y"])
    view = _calibration_action_view(summary, arrays, data)
    destination = Path(destination)
    if destination.resolve() == Path(project).resolve():
        raise ValueError("Use a separate calibration report output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "03_calibration_dataset.pdf"
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figure = _calibration_figure(data, view)
        try:
            figure.savefig(path, format="pdf")
        finally:
            plt.close(figure)
    write_provenance(destination / "calibration_action_support.json", {
        "project": provenance, "view": view, "source": file_record(__file__),
        "plot": file_record(path), "outside_window": int((~view["observed_in_window"]).sum()),
        "semantics": "Replay saved diagnostic arrays on the existing [-2,2] action window. "
                     "The common adjusted envelope matches plot 1 and is not an individual or mean radius. "
                     "Observed y-g(x) retains observation noise; coverage concerns the true mean response. "
                     "Width, discrepancy, and total radius are equally averaged across all target contexts. "
                     "Histogram bins count actual calibration actions; out-of-window draws are explicitly counted. "
                     "The linear width uses global joint context-action leverage, not inverse local histogram density. "
                     "No new fits, observations, PCA, or coverage claims are produced.",
    })
    return path
