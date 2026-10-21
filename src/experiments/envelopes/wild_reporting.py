"""Replay clean comparisons of analytical and wild-bootstrap envelope widths."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

from experiments.provenance import file_record
from experiments.sweep_reporting import write_rows_csv

from .diagnostics import _slice_indices
from .reporting import _STYLE
from .wild import WildLinearDesign, residual_radii

_METHODS = (("analytical", "Analytical", "C0"),
            ("wild_bootstrap", "Wild bootstrap", "C1"))


def write_leverage_report(summary_paths, destination, *, diagnostics=False) -> Path:
    """Compare fixed-n leverage studies, optionally replaying moved-point bands and radii."""
    summary_paths = list(map(Path, summary_paths))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows, inputs, signatures, conditions, axes = [], [], [], [], []
    for path in summary_paths:
        summary = json.loads(path.read_text())
        p = summary["manifest"]
        if (p["design"]["distribution"] != "independent_truncated_standard_normal"
                and not (p["design"]["distribution"] == "independent_standard_normal"
                         and "action_override" in p["design"])):
            raise ValueError("Leverage report requires truncated-normal or normal moved-point studies.")
        signature = json.loads(json.dumps(p))
        signature.pop("name")
        signature["band_target"] = summary.get("band_target", p.get("band_target", "function"))
        bound = p["design"].get("truncation_bound")
        if "action_override" in signature["design"]:
            axis = "outlier_action"
            condition = signature["design"]["action_override"].pop("value")
        else:
            axis = "truncation_bound"
            condition = signature["design"].pop("truncation_bound")
        signatures.append(signature)
        conditions.append(condition)
        axes.append(axis)
        metrics = [row for row in summary["metrics"] if row["method"] == "wild_bootstrap"]
        if len(metrics) != 1:
            raise ValueError("Require one fixed n, B, and delta per study.")
        row = dict(metrics[0], truncation_bound=bound)
        row[axis] = condition
        if row["repetitions"] != p["repetitions"]:
            raise ValueError("Coverage repetition counts disagree.")
        for event in ("auxiliary", "transferred"):
            if row[f"{event}_coverage"] != row[f"{event}_covered_count"] / p["repetitions"]:
                raise ValueError("Coverage estimate disagrees with saved count.")
        rows.append(row)
        inputs.append(file_record(path))
    if (len(rows) < 2 or len(set(axes)) != 1 or len(set(conditions)) != len(conditions)
            or any(s != signatures[0] for s in signatures)):
        raise ValueError("Require distinct sweep values and otherwise identical study manifests.")
    axis = axes[0]
    # This orders a diagnostic axis, never an optimizer solution.
    rows.sort(key=lambda row: np.inf if row[axis] is None else row[axis])
    # Coincident estimates share one marker, without jittering measured leverage.
    groups = {}
    for row in rows:
        key = tuple(round(row[name], 12) for name in
                    ("max_leverage", "auxiliary_coverage", "auxiliary_lower", "auxiliary_upper"))
        groups.setdefault(key, []).append(row)
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for group in groups.values():
            row = group[0]
            coverage = row["auxiliary_coverage"]
            low, high = row["auxiliary_lower"], row["auxiliary_upper"]
            if not 0 <= low <= coverage <= high <= 1:
                raise ValueError("Invalid auxiliary coverage interval.")
            if axis == "outlier_action":
                label = "Moved observation a = " + ", ".join(f"{item[axis]:g}" for item in group)
            else:
                labels = ["Unbounded" if item[axis] is None
                          else f"±{item[axis]:g}" for item in group]
                label = "Bounds " + ", ".join(labels) if labels != ["Unbounded"] else "Unbounded normal"
            ax.errorbar(row["max_leverage"], coverage,
                        yerr=[[coverage-low], [high-coverage]],
                        fmt="o", capsize=3, label=label)
        ax.axhline(1-rows[0]["delta"], color="0.5", linestyle="--", label="Nominal coverage")
        ax.set_xlabel(r"Maximum observed leverage $h_n$")
        residual = signatures[0]["band_target"] == "residual"
        ax.set_ylabel("Simultaneous residual-band coverage" if residual else "Simultaneous auxiliary coverage")
        ax.set_title("Wild-bootstrap coverage versus maximum leverage")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        output = destination / "coverage_vs_leverage.pdf"
        fig.savefig(output, format="pdf")
        plt.close(fig)
    csv = destination / "coverage_vs_leverage.csv"
    write_rows_csv(csv, rows, list(rows[0]))
    (destination / "provenance.json").write_text(json.dumps({
        "inputs": inputs, "source": file_record(Path(__file__)),
        "outputs": [file_record(output), file_record(csv)],
        "intervals": "Pointwise 95% Wilson intervals across independent response/bootstrap repetitions.",
        "coverage_event": "Auxiliary band contains its target jointly at every calibration context and every real action.",
        "interpretation": "Fixed designs; maximum leverage does not determine coverage uniquely.",
        "sweep_axis": axis,
        "coincident_points": [[item[axis] for item in group]
                               for group in groups.values()],
        "coincidence_tolerance": "Displayed coordinates and interval endpoints agree to 12 decimal places.",
    }, indent=2) + "\n")
    if diagnostics:
        _write_moved_diagnostics(summary_paths, destination)
    return output


def _verified_arrays(project, record, inputs):
    path = project / Path(record["path"]).name
    actual = file_record(path)
    if any(actual[key] != record[key] for key in ("sha256", "bytes")):
        raise ValueError(f"Saved artifact changed: {path}")
    inputs.append(actual)
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def _moved_examples(summary_paths):
    items, inputs = [], []
    for path in summary_paths:
        summary = json.loads(path.read_text())
        p, project = summary["manifest"], path.parent
        if p.get("band_target") != "residual" or "action_override" not in p["design"]:
            raise ValueError("Moved-point diagnostics require saved residual studies.")
        inputs.append(file_record(path))
        metadata = []
        for record, relative in ((summary["environment"], "environment.json"),
                                 (summary["repetitions"][0], "repetitions/rep-00000.json")):
            source = project / relative
            actual = file_record(source)
            if any(actual[key] != record[key] for key in ("sha256", "bytes")):
                raise ValueError(f"Saved artifact changed: {source}")
            inputs.append(actual)
            metadata.append(json.loads(source.read_text()))
        environment, repetition = metadata
        if (environment["contract"]["manifest"] != p or repetition["index"] != 0
                or any(m["contract_sha256"] != summary["contract_sha256"] for m in metadata)):
            raise ValueError("Saved environment and repetition disagree with summary.")
        snapshot = _verified_arrays(project, environment["arrays"], inputs)
        first = _verified_arrays(project / "repetitions", repetition["arrays"], inputs)
        example = _verified_arrays(project, summary["example"], inputs)
        _validate_example(snapshot, example)
        design = WildLinearDesign(snapshot["X"], snapshot["actions"])
        row = next(r for r in summary["metrics"] if r["method"] == "wild_bootstrap")
        if not np.isclose(np.max(design.leverage), row["max_leverage"], atol=1e-12, rtol=0):
            raise ValueError("Saved maximum leverage disagrees with design.")
        moved = p["design"]["action_override"]["index"]
        items.append(dict(p=p, snapshot=snapshot, example=example, first=first,
                          repetition=repetition, moved=moved, leverage=design.leverage,
                          action=float(snapshot["actions"][moved])))
    items.sort(key=lambda item: item["action"])
    base = items[0]
    for item in items[1:]:
        mask = np.arange(len(item["snapshot"]["actions"])) != item["moved"]
        for key in ("X", "gp_cos", "gp_sin", "actions_grid"):
            if not np.array_equal(base["snapshot"][key], item["snapshot"][key]):
                raise ValueError(f"Moved-point studies must share {key}.")
        for key in ("actions", "sigma"):
            if not np.array_equal(base["snapshot"][key][mask], item["snapshot"][key][mask]):
                raise ValueError(f"Moved-point studies must share bulk {key}.")
        if (base["repetition"]["seeds"] != item["repetition"]["seeds"]
                or base["repetition"]["multiplier_sha256"] != item["repetition"]["multiplier_sha256"]
                or not np.array_equal(base["first"]["innovations"], item["first"]["innovations"])):
            raise ValueError("Moved-point studies must share innovations and multipliers.")
    return items, inputs


def _mean_bands(example):
    keys = ("truth", "predictor", "residual_truth", "residual_fit")
    means = {key: example[key].mean(axis=0) for key in keys}
    for method in ("analytical", "bootstrap"):
        means[f"{method}_width"] = example[f"{method}_width"].mean(axis=0)
        for side, sign in (("lower", -1), ("upper", 1)):
            means[f"{method}_residual_{side}"] = (
                example["residual_fit"] + sign * example[f"{method}_width"]).mean(axis=0)
            means[f"{method}_transferred_{side}"] = example[f"{method}_{side}"].mean(axis=0)
            means[f"{method}_radius_{side}"] = example[f"{method}_radius_{side}"].mean(axis=0)
    return means


def _moved_distribution(items):
    from scipy.stats import norm, truncnorm

    fig, axes = plt.subplots(len(items), 1, figsize=(10, 2.5*len(items)),
                              sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    fig.suptitle("Fixed normal bulk and one moved observation")
    bulk = np.delete(items[0]["snapshot"]["actions"], items[0]["moved"])
    a = np.linspace(min(-4., float(bulk.min())), max(4., float(bulk.max())), 401)
    bound = items[0]["p"]["design"].get("truncation_bound")
    density = norm.pdf(a) if bound is None else truncnorm.pdf(a, -bound, bound)
    for ax, item in zip(axes[:, 0], items):
        ax.hist(bulk, bins=20, density=True, alpha=.3, label=f"Fixed bulk: {len(bulk)} observations")
        ax.plot(a, density, label="Parent normal: mean 0, SD 1" if bound is None
                else f"Standard normal truncated to ±{bound:g}")
        ax.scatter([item["action"]], [0.], color="C2", zorder=3, clip_on=False)
        h = item["leverage"][item["moved"]]
        # Explicitly requested label above the moved dot; no figure-level text layout.
        ax.annotate(rf"$h_0={h:.5f}$", (item["action"], 0), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.set_xscale("symlog", linthresh=4)
        ax.set_xlim(min(-4., float(bulk.min())-1), max(i["action"] for i in items)*3)
        ax.set_title(rf"$n={len(bulk)+1}$, moved $a={item['action']:g}$; maximum $h_n={max(item['leverage']):.5f}$")
        ax.set_xlabel(r"Action $a$ (symlog; linear for $|a|\leq4$)")
        ax.set_ylabel("Bulk density")
    axes[0, 0].legend()
    return fig


def _moved_band_figure(items, means, *, radii=False):
    fig, axes = plt.subplots(len(items), 2, figsize=(12, 3.2*len(items)),
                            sharex=True, sharey="col", squeeze=False, constrained_layout=True)
    delta = items[0]["p"]["baseline"]["delta"]
    title = "Context-averaged radii" if radii else "Context-averaged bands"
    fig.suptitle(rf"{title}: $\delta={delta:g}$, response repetition 0")
    for row_axes, item, mean in zip(axes, items, means):
        a = item["example"]["actions_grid"]
        for col, ax in enumerate(row_axes):
            for method, label, color in (("analytical", "OLS variance bound", "C0"),
                                         ("bootstrap", "Wild bootstrap", "C1")):
                if radii:
                    if col == 0:
                        ax.plot(a, mean[f"{method}_width"], color=color, label=label)
                    else:
                        for side, style in (("lower", "-"), ("upper", "--")):
                            ax.plot(a, mean[f"{method}_radius_{side}"], color=color,
                                    linestyle=style, label=f"{label}: {side}")
                else:
                    target = "residual" if col == 0 else "transferred"
                    ax.fill_between(a, mean[f"{method}_{target}_lower"],
                                    mean[f"{method}_{target}_upper"], color=color, alpha=.2, label=label)
            if not radii:
                lines = (("residual_truth", r"Mean residual $\overline{g}$"),
                         ("residual_fit", r"Mean OLS fit $\overline{\widetilde g}$")) if col == 0 else (
                         ("truth", r"Mean truth $\overline{f}$"), ("predictor", r"Mean predictor $\overline{\widehat f}$"))
                for (key, label), color, style in zip(lines, ("C2", "C3"), ("-", "--")):
                    ax.plot(a, mean[key], color=color, linestyle=style, label=label)
            target = (("Residual half-width", "Transferred radii") if radii else
                      ("Residual band", "Transferred envelope"))[col]
            ax.set_title(rf"{target}; moved $a={item['action']:g}$, $h_n={max(item['leverage']):.4f}$")
            ax.set_xlabel("Action $a$")
            ax.set_ylabel("Context-averaged radius" if radii else
                          ("Context-averaged residual" if col == 0 else "Context-averaged response"))
    for ax in axes[0]:
        ax.legend()
    return fig


def _write_moved_diagnostics(summary_paths, destination):
    items, inputs = _moved_examples(summary_paths)
    means = [_mean_bands(item["example"]) for item in items]
    outputs = []
    with plt.style.context("default"), plt.rc_context(_STYLE):
        for stem, draw in (("data_distribution", lambda: _moved_distribution(items)),
                           ("context_averaged_bands", lambda: _moved_band_figure(items, means)),
                           ("context_averaged_radii", lambda: _moved_band_figure(items, means, radii=True))):
            output = destination / f"{stem}.pdf"
            fig = draw()
            try:
                fig.savefig(output, format="pdf")
            finally:
                plt.close(fig)
            outputs.append(file_record(output))
    curves = [{"moved_action": item["action"], "action": float(a),
               **{key: float(value[k]) for key, value in mean.items()}}
              for item, mean in zip(items, means)
              for k, a in enumerate(item["example"]["actions_grid"])]
    design_rows = [{"moved_action": item["action"], "row": k, "is_moved": k == item["moved"],
                    "action": float(a), "leverage": float(item["leverage"][k]),
                    "sigma": float(item["snapshot"]["sigma"][k]),
                    **{f"x{j+1}": float(x) for j, x in enumerate(item["snapshot"]["X"][k])}}
                   for item in items for k, a in enumerate(item["snapshot"]["actions"])]
    for stem, rows in (("context_averaged_curves", curves), ("design_points", design_rows)):
        output = destination / f"{stem}.csv"
        write_rows_csv(output, rows, list(rows[0]))
        outputs.append(file_record(output))
    (destination / "diagnostics_provenance.json").write_text(json.dumps({
        "inputs": inputs, "outputs": outputs, "source": file_record(__file__),
        "geometry_source": file_record(Path(__file__).with_name("wild.py")),
        "response_repetition": 0,
        "context_average": "Equal weights on the same calibration contexts; transfer before averaging.",
        "display_domain": "Saved diagnostic window only; coverage covers all real actions.",
        "leverage": "Full contextual hat-matrix diagonal; moved-row and maximum leverage are distinct.",
        "replay": "Saved examples and critical values; no new fits, draws, or optimizer actions.",
    }, indent=2) + "\n")


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
    title = (r"Auxiliary coverage: $\widetilde f\pm w_\delta$" if event == "auxiliary"
             else r"Transferred-envelope coverage: $\widehat f\pm r_\delta$")
    if summary.get("band_target") == "residual":
        title = (r"Residual-band coverage: $\widetilde g\pm w_\delta$" if event == "auxiliary"
                 else "Asymmetric transferred-envelope coverage")
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
                        capsize=3, label=label)
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
    fig.suptitle("Mean auxiliary half-width")
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
    if "residual_fit" in example:
        if not np.allclose(example["residual_truth"], example["truth"]-example["predictor"]):
            raise ValueError("Saved residual truth disagrees with truth minus predictor.")
        for method in ("analytical", "bootstrap"):
            lower, upper = residual_radii(example["residual_fit"], example[f"{method}_width"])
            if (not np.allclose(example[f"{method}_lower"], example["predictor"]-lower)
                    or not np.allclose(example[f"{method}_upper"], example["predictor"]+upper)):
                raise ValueError("Saved bounds disagree with the asymmetric residual transfer.")
    return (0,) if X.shape[1] == 0 else _slice_indices(X)


def _slice_panels(example, indices):
    fig, axes = plt.subplots(1, len(indices), figsize=(4.7*len(indices), 4.8),
                             sharey=True, squeeze=False, constrained_layout=True)
    ranks = ("Central", "Median norm", "Outer") if len(indices) == 3 else ("Central", "Outer")
    for ax, index, rank in zip(axes[0], indices, ranks):
        norm = np.linalg.norm(example["X"][index])
        ax.set_title("Action-only model" if example["X"].shape[1] == 0 else rank + rf": $\|x_i\|_2={norm:.2f}$")
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
        residual_target = summary.get("band_target") == "residual"
        ax.plot(a, auxiliary, ":", color="C2",
                label=r"$\widehat f+\widetilde g$" if residual_target else r"$\widetilde f(x_i,a)$")
        prefix = "analytical" if method == "analytical" else "bootstrap"
        lower, upper = ((example[f"{prefix}_lower"][index], example[f"{prefix}_upper"][index])
                        if residual_target else (predictor-radius, predictor+radius))
        ax.fill_between(a, lower, upper, color="C1", alpha=.2,
                        label="Transferred envelope" if residual_target else r"$\widehat f(x_i,a)\pm r_\delta(x_i,a)$")
        ax.set_ylabel(r"Response $f(x_i,a)$")
    axes[0].legend()
    fig.supxlabel(
        rf"First saved dataset; $n={p['baseline']['n']}$, $B={p['baseline']['B']}$;"
        + (" one action-only realization." if example["X"].shape[1] == 0 else
           " contexts selected by smallest, median, and largest norm."
           "\nContext norm measures centrality; these slices illustrate one realization."),
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
        ("Constant noise scale; independent Gaussian observation errors."
         if p["noise"].get("type") == "constant" else
         "One fixed GP volatility profile; independent Gaussian observation errors conditional on that profile.")
        + "\nThe known variance cap holds for all inputs; it is not estimated from the displayed actions.",
        fontsize=10,
    )
    return fig


def write_reports(summary, snapshot, example, destination, *, paired_example=None) -> list[Path]:
    """Render grid or paired-size coverage, widths, and saved context slices as PDFs."""
    if summary["manifest"].get("sweep_mode") == "paired":
        if paired_example is None:
            raise ValueError("Paired reporting requires saved shared-context examples.")
        return _write_paired_reports(summary, snapshot, paired_example, destination)
    if np.asarray(example["X"]).shape[1] == 0 and summary.get("band_target") != "residual":
        # Action-only conditions are compared by write_leverage_report; there
        # are no distinct context slices to display.
        return []
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
            ("05_analytical_context_slices", _slices_figure(summary, example, indices, "analytical", "Analytical")),
            ("06_wild_bootstrap_context_slices", _slices_figure(summary, example, indices, "wild_bootstrap", "Wild bootstrap")),
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



def _paired_rows(summary, delta, method):
    p = summary["manifest"]
    expected = list(zip(p["n_values"], p["B_values"]))
    rows = [r for r in summary["metrics"] if r["delta"] == delta and r["method"] == method]
    indexed = {(r["n"], r["B"]): r for r in rows}
    if len(rows) != len(expected) or set(indexed) != set(expected):
        raise ValueError("Require exactly the declared paired conditions in the report.")
    return [indexed[pair] for pair in expected]


def _paired_axis(ax, p):
    positions = np.arange(len(p["n_values"]))
    ax.set_xticks(positions, [f"({n},\n{b})" for n, b in zip(p["n_values"], p["B_values"])])
    ax.set_xlabel("Paired sizes (n, B)")


def _paired_metrics_figure(summary, event=None):
    p = summary["manifest"]
    fig, axes = _panels(p["deltas"])
    title = ("Mean residual-band half-width" if event is None else
             "Residual-band coverage" if event == "auxiliary" else "Transferred-envelope coverage")
    fig.suptitle(title)
    for ax, delta in zip(axes, p["deltas"]):
        for method, label, color in _METHODS:
            label = "OLS variance bound" if method == "analytical" else label
            rows = _paired_rows(summary, delta, method)
            x = np.arange(len(rows))
            if event is None:
                ax.plot(x, [r["mean_width"] for r in rows], marker="o", color=color, label=label)
            else:
                y, lower, upper = (np.array([r[f"{event}_{key}"] for r in rows])
                                   for key in ("coverage", "lower", "upper"))
                if (not np.isfinite([y, lower, upper]).all() or np.any(lower > y)
                        or np.any(y > upper) or np.any(lower < 0) or np.any(upper > 1)):
                    raise ValueError("Invalid paired coverage or Wilson intervals.")
                ax.errorbar(x, y, yerr=[y-lower, upper-y], marker="o", capsize=3, color=color, label=label)
        if event is not None:
            ax.axhline(1-delta, linestyle="--", color="0.5", label=r"Target $1-\delta$")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        _paired_axis(ax, p)
        ax.set_ylabel("Simultaneous coverage" if event else r"Mean half-width $\overline{w}_\delta$")
    axes[0].legend()
    if event is None:
        axes[0].set_ylim(bottom=0)
    else:
        # Keep uncertainty intervals visible while retaining headroom above 100%.
        axes[0].set_ylim(top=1.015)
    description = ("All calibration contexts and every real action; 95% Wilson intervals."
                   if event else "Mean over calibration contexts and the action window [-2,2].")
    fig.supxlabel(f"{p['repetitions']:,} response repetitions per pair. " + description, fontsize=10)
    return fig


def _validate_paired_example(summary, snapshot, example):
    p = summary["manifest"]
    indices = np.asarray(example["context_indices"])
    expected = _slice_indices(np.asarray(snapshot["X"])[:min(p["n_values"])])
    if (not np.array_equal(indices, expected)
            or not np.array_equal(example["X"], np.asarray(snapshot["X"])[indices])
            or not np.array_equal(example["n_values"], p["n_values"])
            or not np.array_equal(example["B_values"], p["B_values"])):
        raise ValueError("Paired slices must retain the declared pairs and fixed smallest-prefix contexts.")
    count = len(p["n_values"])
    local_snapshot = {"X": example["X"]}
    slices = []
    for j in range(count):
        local = {key: np.asarray(example[key])[j] for key in
                 ("truth", "predictor", "auxiliary", "residual_truth", "residual_fit",
                  "analytical_width", "bootstrap_width", "analytical_lower", "analytical_upper",
                  "bootstrap_lower", "bootstrap_upper", "sigma_grid")}
        local.update(X=example["X"], actions_grid=example["actions_grid"])
        _validate_example(local_snapshot, local)
        slices.append(local)
    return slices


def _paired_slices_figure(summary, example, slices, *, residual):
    p = summary["manifest"]
    count, k = len(slices), len(example["X"])
    fig, axes = plt.subplots(count, k, figsize=(4.7*k, 3.2*count), sharex=True,
                             sharey="col", squeeze=False, constrained_layout=True)
    title = "Residual bands" if residual else "Transferred function envelopes"
    fig.suptitle(title + rf": $\delta={p['baseline']['delta']:g}$, response repetition 0")
    a = example["actions_grid"]
    for j, (n, b, item) in enumerate(zip(p["n_values"], p["B_values"], slices)):
        for c, index in enumerate(example["context_indices"]):
            ax = axes[j, c]
            for prefix, label, color in (("analytical", "OLS variance bound", "C0"),
                                         ("bootstrap", "Wild bootstrap", "C1")):
                if residual:
                    lower = item["residual_fit"][c]-item[f"{prefix}_width"][c]
                    upper = item["residual_fit"][c]+item[f"{prefix}_width"][c]
                else:
                    lower, upper = item[f"{prefix}_lower"][c], item[f"{prefix}_upper"][c]
                ax.fill_between(a, lower, upper, color=color, alpha=.2, label=label)
            if residual:
                ax.plot(a, item["residual_truth"][c], color="C2", label=r"$g=f-\widehat f$")
                ax.plot(a, item["residual_fit"][c], color="C3", linestyle="--", label=r"OLS fit $\widetilde g$")
            else:
                ax.plot(a, item["truth"][c], color="C2", label=r"Truth $f$")
                ax.plot(a, item["predictor"][c], color="C3", linestyle="--", label=r"Predictor $\widehat f$")
            norm = np.linalg.norm(example["X"][c])
            ax.set_title(rf"$x_{{{index}}}$, $\|x\|={norm:.2f}$; $(n,B)=({n},{b})$")
            ax.set_xlabel("Action $a$")
            ax.set_ylabel("Mean residual" if residual else "Response")
    axes[0, 0].legend()
    return fig


def _write_paired_reports(summary, snapshot, example, destination):
    slices = _validate_paired_example(summary, snapshot, example)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figures = [
            ("01_residual_coverage_vs_pairs", _paired_metrics_figure(summary, "auxiliary")),
            ("02_transferred_coverage_vs_pairs", _paired_metrics_figure(summary, "transferred")),
            ("03_mean_width_vs_pairs", _paired_metrics_figure(summary)),
            ("04_residual_bands_by_pair", _paired_slices_figure(summary, example, slices, residual=True)),
            ("05_transferred_envelopes_by_pair", _paired_slices_figure(summary, example, slices, residual=False)),
            ("06_heteroskedastic_noise", _noise_figure(summary, slices[0], np.arange(len(example["X"])))),
        ]
        for name, fig in figures:
            path = destination / f"{name}.pdf"
            try:
                fig.savefig(path, format="pdf")
            finally:
                plt.close(fig)
            paths.append(path)
    return paths
