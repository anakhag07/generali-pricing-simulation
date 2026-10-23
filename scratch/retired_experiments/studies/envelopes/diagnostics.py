"""Standard context coverage, mean-envelope, component, and slice reports."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from experiments.envelopes.plotting import _slice_indices

from experiments.provenance import file_record, write_provenance
from .reporting import _STYLE, _outcomes, _read_points


def _load_project(project):
    from .study import _read_completed_case

    project = Path(project).resolve()
    summary = json.loads((project / "summary.json").read_text())
    contract = json.loads((project / "contract.json").read_text())
    payload = {key: value for key, value in contract.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if (digest != contract["sha256"] or digest != summary["contract_sha256"]
            or summary["manifest"] != contract["manifest"]):
        raise ValueError("Saved experiment contract does not match its summary.")
    calibration = summary["manifest"]["coverage"]
    expected = 1 if calibration["mode"] == "analytical" else calibration["repetitions"]
    if summary["requested_repetitions"] != expected:
        raise ValueError("Requested repetitions differ from the calibration contract.")
    if summary["target"]["kind"] != summary["manifest"]["target"]["actions"]["type"]:
        raise ValueError("Summary target domain differs from the experiment contract.")
    records = summary["repetition_records"]
    rows = []
    for record in records:
        if file_record(record["path"]) != record:
            raise ValueError("Saved repetition identity changed.")
        rows.append(_read_completed_case(Path(record["path"])))
    if rows != summary["repetitions"]:
        raise ValueError("Saved repetition results differ from the project summary.")
    indices = [row["index"] for row in rows]
    if (len(set(indices)) != len(indices)
            or any(not 0 <= index < summary["requested_repetitions"] for index in indices)
            or any(row["contract_sha256"] != digest for row in rows)):
        raise ValueError("Invalid repetition indices or contract identities.")
    _outcomes(summary)
    return summary


def _load_grid(row, target):
    record = row["artifacts"]["points"]
    _read_points(Path(record["path"]).parent, record)
    with np.load(record["path"], allow_pickle=False) as saved:
        arrays = {key: np.asarray(saved[key], dtype=float) for key in
                  ("X", "actions", "center", "truth", "radius", "excess")}
        for key in ("auxiliary", "width", "discrepancy"):
            if f"component_{key}" in saved:
                arrays[key] = np.asarray(saved[f"component_{key}"], dtype=float)
    count, dimension = target["context_count"], target["context_dim"]
    size = len(arrays["actions"])
    if (count < 1 or size < count or size % count
            or arrays["X"].shape != (size, dimension)
            or any(v.shape != (size,) for k, v in arrays.items() if k != "X")
            or not all(np.isfinite(v).all() for v in arrays.values())):
        raise ValueError("Invalid saved context/action diagnostics.")
    actions = arrays["actions"].reshape(count, -1)
    contexts = arrays["X"].reshape(count, -1, dimension)
    if (not np.all(actions == actions[0])
            or not np.all(contexts == contexts[:, :1])):
        raise ValueError("Diagnostics require shared actions at fixed contexts.")
    calibration_record = row["artifacts"]["calibration"]
    if file_record(calibration_record["path"]) != calibration_record:
        raise ValueError("Saved calibration identity changed.")
    with np.load(calibration_record["path"], allow_pickle=False) as calibration:
        if not np.array_equal(contexts[:, 0], calibration["X"]):
            raise ValueError("Diagnostic contexts differ from the calibration target.")
    grid = {k: v.reshape(count, -1) for k, v in arrays.items() if k not in {"X", "actions"}}
    # Sorting is only for drawing curves; it never selects an optimizer action.
    order = np.argsort(actions[0], kind="stable")
    sorted_actions = actions[0, order]
    duplicates = np.diff(sorted_actions) == 0
    keep = np.r_[True, ~duplicates]
    for key, value in grid.items():
        ordered = value[:, order]
        if not np.array_equal(ordered[:, 1:][:, duplicates], ordered[:, :-1][:, duplicates]):
            raise ValueError("Repeated diagnostic actions have conflicting saved values.")
        grid[key] = ordered[:, keep]
    grid.update(X=contexts[:, 0], actions=sorted_actions[keep])
    if not np.allclose(grid["excess"], grid["radius"]-np.abs(grid["center"]-grid["truth"]),
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Saved excess does not match radius minus absolute error.")
    if {"width", "discrepancy"} <= grid.keys():
        if (np.any(grid["width"] < 0) or np.any(grid["discrepancy"] < 0)
                or not np.allclose(grid["radius"], grid["width"]+grid["discrepancy"],
                                   rtol=1e-12, atol=1e-12)):
            raise ValueError("Saved radius components must be nonnegative and sum to radius.")
        if "auxiliary" in grid and not np.allclose(
                grid["discrepancy"], np.abs(grid["center"]-grid["auxiliary"]),
                rtol=1e-12, atol=1e-12):
            raise ValueError("Saved discrepancy must equal the absolute predictor/auxiliary gap.")
    return grid


def _coverage(summary, grids):
    count = summary["target"]["context_count"]
    statuses = np.full((summary["requested_repetitions"], count), -1, dtype=int)
    for row in summary["repetitions"]:
        decision = row.get("containment")
        if row["status"] == "fit_failed" or decision is None:
            continue
        if decision["domain"] != summary["target"]["kind"]:
            raise ValueError("Containment domain differs from the experiment target.")
        details = decision.get("details", {})
        values = statuses[row["index"]]
        if "contexts" in details:
            records = details["contexts"]
            if (details.get("context_count", count) != count or len(records) != count
                    or sorted(record["context_index"] for record in records) != list(range(count))
                    or any(record["covered"] is not None
                           and type(record["covered"]) is not bool for record in records)):
                raise ValueError("Invalid per-context containment certificate.")
            for record in records:
                if record["covered"] is not None:
                    values[record["context_index"]] = int(record["covered"])
        elif decision["status"] == "covered":
            values[:] = 1
        elif (decision["method"] == "constant_affine_all_real"
              and decision["domain"] == "real_line"
              and details.get("slope_mismatch") is True
              and details.get("context_count") == count
              and Fraction(details["action_slope_difference"]) != 0):
            # Replay the old exact certificate, which proves failure at every x.
            values[:] = 0
        elif decision["domain"] == "finite_set" and row["index"] in grids:
            grid = grids[row["index"]]
            target = np.asarray(summary["manifest"]["target"]["actions"]["values"])
            matches = [np.flatnonzero(grid["actions"] == a) for a in target]
            if all(len(match) == 1 for match in matches):
                positions = [int(match[0]) for match in matches]
                values[:] = np.all(grid["excess"][:, positions] >= 0, axis=1)
        if ((decision["status"] == "covered" and np.any(values == 0))
                or (decision["status"] == "violated" and np.all(values == 1))):
            raise ValueError("Per-context coverage contradicts the saved global decision.")
    return {"statuses": statuses,
            "mean": float(statuses.mean()) if np.all(statuses >= 0) else None,
            "covered_fraction": float(np.mean(statuses == 1)),
            "unresolved_fraction": float(np.mean(statuses < 0))}




def _averages(grid):
    return {key: grid[key].mean(axis=0) for key in
            ("center", "truth", "radius", "auxiliary", "width", "discrepancy") if key in grid}


def _average_repetition_grids(summary, grids):
    expected = set(range(summary["requested_repetitions"]))
    if (not expected or set(grids) != expected
            or any(row["status"] == "fit_failed" for row in summary["repetitions"])):
        raise ValueError("Averaged diagnostics require every requested repetition to have a successful saved fit.")
    first = grids[0]
    fixed = {"X", "actions", "center", "truth"}
    result = {key: value.copy() for key, value in first.items()}
    for index in range(1, summary["requested_repetitions"]):
        grid = grids[index]
        if (grid.keys() != first.keys()
                or any(not np.array_equal(grid[key], first[key]) for key in fixed)):
            raise ValueError("Averaged diagnostics require identical contexts, actions, truth, center, and component keys.")
        for key in grid.keys() - fixed:
            # Discrepancies were made absolute for each fit before averaging.
            result[key] += grid[key]
    for key in result.keys() - fixed:
        result[key] /= len(grids)
    return result


def _radius_label(summary):
    envelope = summary["manifest"]["envelope"]
    if envelope["type"] == "constant_radius":
        return f"constant radius = {envelope['radius']:g}"
    if envelope["type"] == "linear_known_variance_transfer":
        return "analytical-radius"
    return "Envelope radius"


def _coverage_figure(summary, coverage):
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    real_line = summary["target"]["kind"] == "real_line"
    domain = r"$\mathcal{A}=\mathbb{R}$" if real_line else r"$\mathcal{A}_{\mathrm{target}}$"
    height = coverage["covered_fraction"]
    bars = ax.bar([domain], [height], width=.3, label="Covered")
    if coverage["mean"] is not None:
        ax.bar_label(bars, labels=[f"{height:.0%}"], padding=3)
    else:
        ax.bar([domain], [coverage["unresolved_fraction"]], bottom=[height], width=.3, label="Unresolved")
        ax.legend()
    ax.set_xlim(-.6, .6)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Action domain")
    ax.set_ylabel(r"Context-averaged coverage $\overline{c}$")
    ax.set_title(f"Simultaneous coverage over actions\n{_radius_label(summary)}")
    count, repeats = summary["target"]["context_count"], summary["requested_repetitions"]
    scope = "all real actions" if real_line else "every declared target action"
    note = (f"For each context, check containment for {scope} using saved certificates.\n"
            f"Average equally over {count} contexts and {repeats} calibration realization(s).\n"
            "This context average is not the joint all-context coverage probability.")
    if coverage["mean"] is None:
        note += "\nUnresolved cases retain their weight; the mean is unavailable."
    fig.supxlabel(note, fontsize=10)
    return fig


def _mean_figure(grid, radius_label=None):
    fig, (ax, radius_ax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)
    means = _averages(grid)
    a = grid["actions"]
    marker = "o" if len(a) == 1 else None
    ax.plot(a, means["truth"], marker=marker, label=r"$\overline{f}(a)$")
    line, = ax.plot(a, means["center"], "--", marker=marker, label=r"$\overline{\widehat f}(a)$")
    _draw_band(ax, a, means["center"], means["radius"], line.get_color(),
               r"$\overline{\widehat f}(a)\pm\overline{r}_\delta(a)$")
    ax.set_title("Context-averaged envelope" + (f": {radius_label}" if radius_label else ""))
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"Mean response $\overline{f}(a)$")
    ax.legend()
    radius_ax.plot(a, means["radius"], marker=marker)
    radius_ax.set_title(r"Mean radius $\overline{r}_\delta(a)$")
    radius_ax.set_xlabel(r"$a$")
    radius_ax.set_ylabel(r"$\overline{r}_\delta(a)$")
    radius_ax.set_ylim(bottom=0)
    return fig


def _components_figure(grid, constant, radius_label=None):
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    means, a = _averages(grid), grid["actions"]
    if {"width", "discrepancy"} <= means.keys():
        labels = [r"$\overline{|\widehat f-\widetilde f|}(a)$", r"$\overline{w}_\delta(a)$"]
        if len(a) == 1:
            ax.bar(a, means["discrepancy"], alpha=.35, label=labels[0])
            ax.bar(a, means["width"], bottom=means["discrepancy"], alpha=.35, label=labels[1])
        else:
            ax.stackplot(a, means["discrepancy"], means["width"], alpha=.35, labels=labels)
        ax.plot(a, means["radius"], color="C2", marker="o" if len(a) == 1 else None,
                label=radius_label or r"$\overline{r}_\delta(a)$")
        ax.set_title(r"$\overline{r}_\delta=\overline{|\widehat f-\widetilde f|}+\overline{w}_\delta$")
    else:
        ax.plot(a, means["radius"], marker="o" if len(a) == 1 else None,
                label=radius_label or r"$\overline{r}_\delta(a)$")
        ax.set_title(radius_label or ("Constant radius" if constant else "Envelope radius"))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$a$")
    ax.set_ylabel("Mean radius contribution")
    ax.legend()
    return fig


def _draw_band(ax, actions, center, radius, color, label):
    if len(actions) == 1:
        ax.errorbar(actions, center, yerr=radius, fmt="none", capsize=4,
                    color=color, label=label)
    else:
        ax.fill_between(actions, center-radius, center+radius, color=color, alpha=.2, label=label)


def _slices_figure(grid, indices, external, radius_label=None):
    fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 4.8),
                             sharey=True, squeeze=False, constrained_layout=True)
    ranks = ("Central", "Median norm", "Outer") if len(indices) == 3 else ("Central", "Outer")
    a = grid["actions"]
    marker = "o" if len(a) == 1 else None
    for ax, index, rank in zip(axes[0], indices, ranks):
        ax.plot(a, grid["truth"][index], marker=marker, label=r"$f(x_i,a)$")
        center, = ax.plot(a, grid["center"][index], "--", marker=marker, label=r"$\widehat f(x_i,a)$")
        if "auxiliary" in grid:
            label = r"$\widetilde f(x_i,a)$" + (" (reference)" if external else "")
            ax.plot(a, grid["auxiliary"][index], ":", marker=marker, label=label)
        _draw_band(ax, a, grid["center"][index], grid["radius"][index], center.get_color(),
                   radius_label or r"$\widehat f(x_i,a)\pm r_\delta(x_i,a)$")
        ax.set_title(rank + rf": $\|x_{{{index}}}\|_2={np.linalg.norm(grid['X'][index]):.2f}$")
        ax.set_xlabel(r"$a$")
        ax.set_ylabel(r"$f(x_i,a)$")
        ax.legend()
    return fig


def write_diagnostic_reports(project: Path, *, auxiliary_project: Path | None = None,
                             repetition_index: int | None = None,
                             average_repetitions: bool = False) -> list[Path]:
    """Replay context diagnostics for one saved fit or all requested repetitions, with an optional paired auxiliary."""
    if average_repetitions and repetition_index is not None:
        raise ValueError("Choose either an explicit repetition or an average across repetitions.")
    project = Path(project).resolve()
    summary = _load_project(project)
    grids = {row["index"]: _load_grid(row, summary["target"])
             for row in summary["repetitions"]
             if row["status"] != "fit_failed" and "points" in row["artifacts"]}
    coverage = _coverage(summary, grids)
    selected = (None if average_repetitions else
                min(grids, default=None) if repetition_index is None else repetition_index)
    if repetition_index is not None and selected not in grids:
        raise ValueError("Requested repetition has no saved fitted diagnostics.")
    grid = (_average_repetition_grids(summary, grids) if average_repetitions else
            None if selected is None else grids[selected].copy())
    external = None
    if auxiliary_project is not None:
        other = _load_project(auxiliary_project)
        own = {row["index"]: row for row in summary["repetitions"]
               if average_repetitions or row["index"] == selected}
        rows = [row for row in other["repetitions"] if row["index"] in own]
        if (grid is None or not rows or len(rows) != len(own)
                or any(row["status"] == "fit_failed" for row in rows)
                or (average_repetitions and other["requested_repetitions"] != len(own))):
            raise ValueError("No matching saved auxiliary realization.")
        if "auxiliary" in grid:
            raise ValueError("An external reference cannot replace a construction's own auxiliary fit.")
        auxiliary = np.zeros_like(grid["center"])
        for row in rows:
            reference = _load_grid(row, other["target"])
            if ("auxiliary" not in reference
                    or any(not np.array_equal(grid[key], reference[key])
                           for key in ("X", "actions", "truth", "center"))
                    or any(own[row["index"]]["artifacts"]["calibration"][key]
                           != row["artifacts"]["calibration"][key] for key in ("sha256", "bytes"))):
                raise ValueError("Auxiliary reference must use identical calibration data and diagnostic models/locations.")
            auxiliary += reference["auxiliary"]
        grid["auxiliary"] = auxiliary / len(rows)
        external = {"source_project": str(Path(auxiliary_project).resolve()),
                    "contract": file_record(Path(auxiliary_project)/"contract.json"),
                    **({"repetitions": rows} if average_repetitions else {"repetition": rows[0]})}
    destination = project / "reports" / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    if external is not None:
        external_snapshot = write_provenance(destination / "auxiliary_summary.json", other)
        external["summary"] = file_record(external_snapshot)
    names = ("01_context_coverage", "02_mean_envelope", "03_radius_decomposition", "04_context_slices")
    paths = [destination / f"{name}.pdf" for name in names]
    indices = [] if grid is None else _slice_indices(grid["X"])
    with plt.style.context("default"), plt.rc_context(_STYLE):
        figures = [_coverage_figure(summary, coverage)]
        if grid is not None:
            constant = summary["manifest"]["envelope"]["type"] == "constant_radius"
            label = _radius_label(summary)
            figures += [_mean_figure(grid, label), _components_figure(grid, constant, label),
                        _slices_figure(grid, indices, external is not None, label)]
            realization = (f"Mean over {len(grids)} calibration datasets" if average_repetitions
                           else f"Realization {selected}")
            base = f"{realization}; equal averages over {len(grid['X'])} contexts."
            figures[1].supxlabel(base + "\nCoverage of the mean does not imply coverage at every context.", fontsize=10)
            component_note = base
            if not {"width", "discrepancy"} <= grid.keys():
                component_note += "\nRadius specified directly; no discrepancy/width decomposition." if constant else "\nNo saved discrepancy/width decomposition."
            figures[2].supxlabel(component_note, fontsize=10)
            slice_note = f"{realization}; contexts selected by norm, with stable ties and the lower median rank."
            if external is not None:
                slice_note += "\nThe auxiliary curve is a paired analytical reference; it does not construct this radius."
            elif "auxiliary" not in grid:
                slice_note += "\nNo auxiliary model is saved for this construction."
            figures[3].supxlabel(slice_note, fontsize=10)
        else:
            for title in ("Context-averaged envelope", "Radius decomposition", "Context slices"):
                fig, ax = plt.subplots(constrained_layout=True)
                ax.set(title=title+"\nNo saved fitted diagnostics", xlabel=r"$a$", ylabel="Response / radius")
                figures.append(fig)
        for fig, path in zip(figures, paths):
            try:
                fig.savefig(path, format="pdf")
            finally:
                plt.close(fig)
    # collect() subsequently updates the project's plot list. Keep the exact
    # input snapshot so that report provenance does not point at a changing hash.
    snapshot = write_provenance(destination / "source_summary.json", summary)
    write_provenance(destination / "diagnostics.json", {
        "summary": file_record(snapshot), "source_project": str(project),
        "contract": file_record(project/"contract.json"),
        "repetitions": summary["repetition_records"], "coverage": coverage,
        "average_repetitions": average_repetitions,
        "averaged_repetition_count": len(grids) if average_repetitions else None,
        "selected_repetition": selected, "slice_indices": indices,
        "slice_norms": [] if grid is None else np.linalg.norm(grid["X"][indices], axis=1),
        "grid": grid, "means": None if grid is None else _averages(grid),
        "external_auxiliary": external, "plots": [file_record(path) for path in paths],
        "sources": [file_record(__file__), file_record(Path(__file__).with_name("reporting.py"))],
        "semantics": "Coverage is simultaneous over declared target actions at each fixed context, "
                     "averaged equally over contexts and requested calibration repetitions, not the "
                     "joint all-context probability. Unknown, failed, and missing cases keep their "
                     "denominator weight. Other plots use "
                     + (f"all {len(grids)} requested calibration datasets" if average_repetitions
                        else "one explicit saved realization") + " and equal "
                     "context weights. The mean band averages endpoints; it is not an intersection. "
                     "Components use the mean absolute discrepancy, not the absolute mean discrepancy. "
                     "Norm is a centrality proxy, not certified difficulty. Curves join saved diagnostic "
                     "samples for visualization only. No new fit, draw, or optimizer is used.",
    })
    return paths
