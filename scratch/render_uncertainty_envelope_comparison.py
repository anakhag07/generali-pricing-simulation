"""Replay the radius-3/writeup comparison and certify the displayed interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from experiments.envelopes.reporting import (
    _STYLE, _comparison_case, write_calibration_plot, write_comparison,
)
from experiments.envelopes.diagnostics import _load_project, _load_grid
from experiments.paths import results_root
from experiments.provenance import file_record, write_provenance
from experiments.sweep_reporting import write_rows_csv


def _simple_figure(curves):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for row in curves:
        axes[0].plot(row["actions"], row["minimum_excess"], label=row["label"])
        axes[1].plot(row["actions"], row["mean_radius"], label=row["label"])
    axes[0].axhline(0, color="0.5", linestyle="--", label="Coverage boundary")
    axes[0].set_title("Does the envelope contain the truth?")
    axes[0].set_ylabel("Smallest coverage margin")
    axes[1].set_title("How wide is the envelope?")
    axes[1].set_ylabel("Mean envelope half-width")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Action a")
        ax.set_xlim(-2, 2)
        ax.legend()
    return fig


def _write_simple_comparison(projects, destination):
    cases = [_comparison_case(project) for project in projects]
    for key in ("X", "actions", "center", "truth", "weights"):
        if not np.array_equal(cases[0][1][key], cases[1][1][key]):
            raise ValueError(f"Comparison requires identical saved diagnostic {key}.")
    curves = []
    for label, (summary, arrays, _) in zip(("constant radius = 3", "analytical-radius"), cases):
        count = summary["target"]["context_count"]
        actions = arrays["actions"].reshape(count, -1)
        contexts = arrays["X"].reshape(count, actions.shape[1], -1)
        if (not np.all(actions == actions[0]) or np.any(np.diff(actions[0]) <= 0)
                or not np.all(contexts == contexts[:, :1, :])):
            raise ValueError("Expected shared increasing actions at fixed saved contexts.")
        mask = (actions[0] >= -2) & (actions[0] <= 2)
        window = actions[0, mask]
        if len(window) < 2 or window[0] != -2 or window[-1] != 2:
            raise ValueError("Saved diagnostics must span the displayed [-2,2] window.")
        # These summarize contexts at each action; no action is selected.
        curves.append({
            "label": label, "context_count": count, "actions": window,
            "minimum_excess": arrays["excess"].reshape(count, -1)[:, mask].min(axis=0),
            "mean_radius": arrays["radius"].reshape(count, -1)[:, mask].mean(axis=0),
        })
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "04_simple_comparison.pdf"
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig = _simple_figure(curves)
        try:
            fig.savefig(path, format="pdf")
        finally:
            plt.close(fig)
    write_provenance(destination / "simple_comparison.json", {
        "projects": [provenance for _, _, provenance in cases], "curves": curves,
        "source": file_record(__file__), "plot": file_record(path),
        "semantics": "Replay the saved 60-context diagnostics on [-2,2], without context "
                     "adjustment, intersection, fitting, simulation, or optimization. "
                     "At each sampled action the smallest signed excess is positive if "
                     "every context strictly covers its truth, zero if at least one "
                     "touches its boundary and none fails, and negative if any fails. "
                     "The mean half-width averages original individual radii equally "
                     "across contexts. It is neither overlap width nor excess radius. "
                     "Lines connect diagnostic samples; this is one realized dataset, "
                     "not a coverage probability estimate or a new continuum certificate. "
                     "Separate saved all-real certificates and guarantees remain in projects.",
    })
    return path


def _average_curves(projects):
    summaries = [_load_project(project) for project in projects]
    total = summaries[0]["requested_repetitions"]
    if any(not s["complete"] or not s["successful"]
           or s["requested_repetitions"] != total
           or len(s["repetitions"]) != total for s in summaries):
        raise ValueError("Averaging requires equally many complete, successful repetitions.")
    curves = []
    reference = None
    for label, summary in zip(("constant radius = 3", "analytical-radius"), summaries):
        radius_sum = excess_sum = None
        for row in summary["repetitions"]:
            grid = _load_grid(row, summary["target"])
            if reference is None:
                reference = grid
            if any(not np.array_equal(grid[key], reference[key])
                   for key in ("X", "actions", "truth", "center")):
                raise ValueError("Repeated comparison requires fixed, identical diagnostic models and locations.")
            radius = grid["radius"].mean(axis=0)
            # Reduce contexts within each dataset before averaging datasets.
            excess = grid["excess"].min(axis=0)
            radius_sum = radius.copy() if radius_sum is None else radius_sum + radius
            excess_sum = excess.copy() if excess_sum is None else excess_sum + excess
        curves.append({"label": label, "actions": reference["actions"],
                       "mean_radius": radius_sum / total,
                       "minimum_excess": excess_sum / total})
    for left, right in zip(summaries[0]["repetitions"], summaries[1]["repetitions"]):
        if left["index"] != right["index"] or any(
                left["artifacts"]["calibration"][key] != right["artifacts"]["calibration"][key]
                for key in ("sha256", "bytes")):
            raise ValueError("Both constructions must use the same response datasets.")
    return summaries, curves


def _write_average_comparison(projects, destination):
    summaries, curves = _average_curves(projects)
    total = summaries[0]["requested_repetitions"]
    coverage = [s["coverage"]["empirical"] for s in summaries]
    if any(c is None or c["estimate"] is None for c in coverage):
        raise ValueError("Joint coverage requires resolved empirical certificates for every dataset.")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / "01_coverage_comparison.pdf", destination / "02_radius_excess.pdf"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        heights = [c["estimate"] for c in coverage]
        bars = ax.bar([c["label"] for c in curves], heights, width=.3)
        for i, c in enumerate(coverage):
            lo, hi = c["interval"]
            ax.errorbar(i, c["estimate"], yerr=[[c["estimate"]-lo], [hi-c["estimate"]]],
                        fmt="none", color="black", capsize=3, elinewidth=.7, capthick=.7)
        ax.bar_label(bars, labels=[f"{x:.1%}" for x in heights], padding=5)
        ax.set(xlabel="Radius construction", ylabel="Joint simultaneous coverage",
               title=f"Coverage across {total:,} datasets", ylim=(0, 1.12), xlim=(-.7, 1.7))
        fig.supxlabel("Containment at all 60 fixed contexts and every real action.\n"
                      "Error bars: 95% Wilson intervals across response datasets.", fontsize=10)
        fig.savefig(paths[0], format="pdf")
        plt.close(fig)
        fig = _simple_figure(curves)
        fig.axes[0].set_title("Mean smallest coverage margin")
        fig.axes[0].set_ylabel("Mean of within-dataset smallest margins")
        fig.axes[1].set_title("Mean envelope radius")
        fig.suptitle(f"Equal averages across {total:,} response datasets")
        fig.supxlabel("Margins first take the minimum over contexts within each dataset, then average datasets.\n"
                      "A positive mean margin is not a coverage certificate; curves join diagnostic samples.", fontsize=10)
        fig.savefig(paths[1], format="pdf")
        plt.close(fig)
    write_provenance(destination / "comparison.json", {
        "projects": [file_record(project / "summary.json") for project in projects],
        "contracts": [file_record(project / "contract.json") for project in projects],
        "repetitions": total, "curves": curves, "joint_coverage": coverage,
        "source": file_record(__file__), "plots": [file_record(p) for p in paths],
        "semantics": "Paired response datasets at the original fixed n=60 design with Gaussian sigma=1. "
                     "Contexts, truth, predictor, constant radius 3, and named seed streams are unchanged. "
                     "Each dataset has equal weight. Radius is averaged over contexts and datasets; "
                     "minimum excess is reduced over contexts before averaging datasets. Coverage uses "
                     "saved all-real certificates, not averaged curves or a finite diagnostic grid. "
                     "No action optimization is involved.",
    })
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    parser.add_argument("--simple", action="store_true",
                        help="Replay only the concise coverage-margin and individual-radius view.")
    parser.add_argument("--average", action="store_true",
                        help="Replay the paired 2000-response experiments with equal dataset averages.")
    args = parser.parse_args()
    if args.average:
        projects = [args.runs_root / name for name in (
            "uncertainty-envelopes-constant-3-2000", "uncertainty-envelopes-analytical-radius-2000",
        )]
        destination = args.runs_root / "uncertainty-envelopes-comparison-2000"
        print(*_write_average_comparison(projects, destination), sep="\n")
        return
    projects = [args.runs_root / name for name in (
        "uncertainty-envelopes-constant-3", "uncertainty-envelopes-writeup",
    )]
    destination = args.runs_root / "uncertainty-envelopes-comparison"
    if args.simple:
        print(_write_simple_comparison(projects, destination))
        return
    previous_plots = [destination / name for name in ("01_envelope_slices.pdf",)
                      if (destination / name).exists()]
    previous_summary = destination / "comparison.json"
    previous_metadata = previous_summary.read_bytes() if previous_plots and previous_summary.exists() else None
    paths = write_comparison(projects, destination)
    paths.append(write_calibration_plot(projects[1], destination))
    summaries = [json.loads((project / "summary.json").read_text()) for project in projects]
    constant = summaries[0]
    config = constant["manifest"]
    if any(config["problem"][key]["type"] != "linear" for key in ("truth", "predictor")):
        raise ValueError("This interval certificate requires the declared linear models.")
    if config["envelope"]["type"] != "constant_radius":
        raise ValueError("This replay expects the literal constant-radius construction.")
    diagnostic = config["tightness"]["actions"]
    if diagnostic["type"] != "linspace":
        raise ValueError("Expected the saved interval visualization window.")
    endpoints = [sp.Rational(float(diagnostic[key])) for key in ("min", "max")]
    radius = sp.Rational(float(config["envelope"]["radius"]))
    coefficients = {
        key: [sp.Rational(float(v)) for v in config["problem"][key]["coefficients"]]
        for key in ("truth", "predictor")
    }
    difference = [p-t for p, t in zip(coefficients["predictor"], coefficients["truth"])]
    point_record = constant["repetitions"][0]["artifacts"]["points"]
    with np.load(point_record["path"], allow_pickle=False) as data:
        contexts = data["X"][::diagnostic["count"]]
    checks = []
    for index, context in enumerate(contexts):
        intercept = difference[0] + sum(c * sp.Rational(float(x))
                                       for c, x in zip(difference[1:-1], context))
        errors = [intercept + difference[-1]*action for action in endpoints]
        checks.append({"context_index": index, "endpoint_errors": list(map(str, errors)),
                       "covered": all(bool(abs(error) <= radius) for error in errors)})
    certificate = {
        "scope": "Every saved calibration context, every action in the closed displayed interval.",
        "interval": list(map(str, endpoints)), "radius": str(radius),
        "covered": all(check["covered"] for check in checks), "checks": checks,
        "proof": "The absolute value of an affine error is convex. Containment at both endpoints implies containment throughout their interval.",
        "numerics": "Exact rational arithmetic on stored floating-point coefficients and contexts; no rounding enclosure for floating-point function evaluation.",
        "probability_semantics": "The constant radius and predictor are fixed: this truth-based interval event does not depend on calibration responses. This is not a generic 95% calibration guarantee.",
        "action_slope_difference": str(difference[-1]),
        "all_real_containment": constant["repetitions"][0]["containment"],
        "inputs": [file_record(project / "summary.json") for project in projects] + [point_record],
        "source": file_record(__file__),
    }
    (destination / "window_certificate.json").write_text(json.dumps(certificate, indent=2)+"\n")
    table = []
    for label, summary in zip(("constant radius = 3", "analytical-from-linear-f radius"), summaries):
        row = summary["repetitions"][0]
        table.append({"construction": label, "all_real_containment": row["containment"]["status"],
                      **row["metrics"]})
    write_rows_csv(destination / "metrics.csv", table, list(table[0]))
    # Retire the old coverage slices after successful replay. The current
    # 02_radius_excess.pdf is now part of the paired comparison, not an archive.
    if previous_plots:
        archive = destination / "previous-context-slices"
        archive.mkdir(exist_ok=True)
        for previous in previous_plots:
            archived = archive / previous.name
            if archived.exists():
                archived = archive / f"{previous.stem}-{file_record(previous)['sha256'][:12]}.pdf"
            previous.replace(archived)
        if previous_metadata is not None:
            archived_metadata = archive / "comparison.json"
            if not archived_metadata.exists():
                archived_metadata.write_bytes(previous_metadata)
    print(json.dumps({"metrics": table, "constant_interval_covered": certificate["covered"],
                      "plots": list(map(str, paths))}, indent=2))


if __name__ == "__main__":
    main()
