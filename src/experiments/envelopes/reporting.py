"""Replay uncertainty-envelope coverage and finite-location diagnostics as PDFs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def write_comparison(projects, destination: Path) -> list[Path]:
    """Plot two common envelopes after subtracting known context-only truth shifts."""
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
    destination = Path(destination)
    if any(destination.resolve() == Path(project).resolve() for project in projects):
        raise ValueError("Use a separate comparison output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "01_coverage_comparison.pdf"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figure = _coverage_comparison_figure(cases, aligned)
        try:
            figure.savefig(paths[0], format="pdf")
        finally:
            plt.close(figure)
    arrays_path = destination / "adjusted_coverage.npz"
    np.savez_compressed(arrays_path,
                        **{key: aligned[0][key] for key in ("actions", "contexts", "truth", "mean_predictor", "g")},
                        **{key: np.stack([row[key] for row in aligned])
                           for key in ("lower", "upper", "empty", "covered")})
    write_provenance(destination / "comparison.json", {
        "projects": [provenance for _, _, provenance in cases],
        "context_count": len(aligned[0]["contexts"]), "context_values": aligned[0]["contexts"],
        "context_adjustment": {"formula": _context_formula(aligned[0]["coefficients"]),
                               "truth_coefficients": aligned[0]["coefficients"],
                               "source": "Known synthetic truth, not estimated from observations."},
        "sampled_action_coverage": [bool(row["covered"].all()) for row in aligned],
        "action_window": [-2., 2.], "source": file_record(__file__),
        "plots": [file_record(path) for path in paths], "aligned_arrays": file_record(arrays_path),
        "semantics": "Subtract the same context-only truth contribution from truth, predictions, and interval endpoints. "
                     "Shading intersects the adjusted intervals across every saved context; empty intersections are not shaded. "
                     "The mean adjusted predictor is for orientation only. Radii and original finite diagnostic metrics are unchanged. "
                     "Curves join saved finite diagnostic values. Floating-point alignment is not a new continuous certificate. "
                     "Saved full-domain containment and analytical probability statements are displayed separately, without refitting or optimization.",
    })
    return paths


def _calibration_projection(summary, data, fit):
    problem = summary["manifest"]["problem"]
    if any(problem[name]["type"] != "linear" for name in ("truth", "predictor")):
        raise ValueError("Calibration reference planes require declared linear models.")
    if fit["metadata"].get("construction") != "linear_known_variance_transfer":
        raise ValueError("Calibration reference planes require the saved linear auxiliary fit.")
    mean = data.X.mean(axis=0)
    centered = data.X-mean
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] <= 0:
        raise ValueError("Calibration PCA requires nonzero context variation.")
    pc1 = directions[0].copy()
    # Fix the otherwise arbitrary SVD sign, without changing the projection.
    if pc1[np.flatnonzero(pc1)[0]] < 0:
        pc1 *= -1
    coefficients = {name: np.asarray(problem[name]["coefficients"], dtype=float)
                    for name in ("truth", "predictor")}
    coefficients["auxiliary"] = np.asarray(fit["state"]["beta"], dtype=float)
    planes = {}
    for name, beta in coefficients.items():
        if beta.shape != (data.X.shape[1]+2,) or not np.isfinite(beta).all():
            raise ValueError("Calibration planes require dimension-matched finite coefficients.")
        planes[name] = {"intercept": float(beta[0] + mean @ beta[1:-1]),
                        "action_slope": float(beta[-1]),
                        "pc1_slope": float(pc1 @ beta[1:-1])}
    return {"context_mean": mean, "pc1": pc1, "scores": centered @ pc1,
            "singular_values": singular_values,
            "explained_variance_ratio": float(singular_values[0]**2 / (singular_values**2).sum()),
            "planes": planes}


def _calibration_figure(data, projection):
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    scores = projection["scores"]
    actions, contexts = np.meshgrid(np.linspace(data.actions.min(), data.actions.max(), 9),
                                   np.linspace(scores.min(), scores.max(), 9))
    handles = []
    for index, (name, label) in enumerate((("truth", "True f"),
                                         ("predictor", "Frozen predictor"),
                                         ("auxiliary", "Saved auxiliary OLS fit"))):
        plane = projection["planes"][name]
        response = plane["intercept"] + plane["action_slope"]*actions + plane["pc1_slope"]*contexts
        ax.plot_wireframe(actions, contexts, response, rstride=4, cstride=4,
                          color=f"C{index}", alpha=.7, zorder=1)
        handles.append(Line2D([], [], color=f"C{index}",
                              label=f"{label}: slopes a = {plane['action_slope']:.3f}, PC1 = {plane['pc1_slope']:.3f}"))
    # Keep the actual observations visible above reference wireframes; no
    # coordinates, response values, camera angles, or box aspects are adjusted.
    scatter = ax.scatter(data.actions, scores, data.y, color="C3", s=20,
                         depthshade=False, zorder=10, label=f"{len(data.y)} observed responses")
    ax.legend(handles=[scatter, *handles])
    ax.set_xlabel("Action a")
    ax.set_ylabel("Centered context PC1 score")
    ax.set_zlabel("Observed / model response")
    fig.suptitle("Single calibration dataset and linear model slopes")
    fig.supxlabel(
        f"PCA uses centered contexts only; PC1 explains {100*projection['explained_variance_ratio']:.1f}% of their sample variation.\n"
        "Reference planes set the other PC scores to zero; observed points retain those components and noise.\n"
        "This PCA direction is sample-specific. All models are replayed without refitting.", fontsize=10,
    )
    return fig


def write_calibration_plot(project: Path, destination: Path) -> Path:
    """Replay raw calibration responses against action and context PC1, with saved linear model slices."""
    from .core import CalibrationData

    summary, _, provenance = _comparison_case(project)
    artifacts = summary["repetitions"][0]["artifacts"]
    with np.load(artifacts["calibration"]["path"], allow_pickle=False) as saved:
        data = CalibrationData(saved["X"], saved["actions"], saved["y"])
    fit = json.loads(Path(artifacts["fit"]["path"]).read_text())
    projection = _calibration_projection(summary, data, fit)
    destination = Path(destination)
    if destination.resolve() == Path(project).resolve():
        raise ValueError("Use a separate calibration report output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "03_calibration_dataset.pdf"
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figure = _calibration_figure(data, projection)
        try:
            figure.savefig(path, format="pdf")
        finally:
            plt.close(figure)
    write_provenance(destination / "calibration_pca.json", {
        "project": provenance, "projection": projection, "source": file_record(__file__),
        "plot": file_record(path),
        "semantics": "Full deterministic SVD of centered, unscaled calibration contexts only. "
                     "PC1 has its first nonzero loading positive. Actions and responses are excluded from PCA. "
                     "Scatter coordinates are the original actions, projected centered contexts, and raw observed responses. "
                     "Reference planes evaluate the original models at x = context_mean + score * pc1, "
                     "setting all omitted centered PC scores to zero. Observations retain omitted-context effects and noise. "
                     "Coefficients are replayed, not refitted. Isotropic population contexts have no preferred PCA direction. "
                     "This visualization does not change the envelope construction, coverage target, or guarantees.",
    })
    return path
