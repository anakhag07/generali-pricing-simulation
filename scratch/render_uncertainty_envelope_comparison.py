"""Replay the radius-3/writeup comparison and certify the displayed interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp

from experiments.envelopes.reporting import write_comparison
from experiments.paths import results_root
from experiments.provenance import file_record
from experiments.sweep_reporting import write_rows_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    args = parser.parse_args()
    projects = [args.runs_root / name for name in (
        "uncertainty-envelopes-constant-3", "uncertainty-envelopes-writeup",
    )]
    destination = args.runs_root / "uncertainty-envelopes-comparison"
    previous_plots = [destination / name for name in ("01_envelope_slices.pdf", "02_radius_excess.pdf")
                      if (destination / name).exists()]
    previous_summary = destination / "comparison.json"
    previous_metadata = previous_summary.read_bytes() if previous_plots and previous_summary.exists() else None
    paths = write_comparison(projects, destination)
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
    # Keep the earlier context-slice figures available, but present only the
    # new two-panel comparison in the result directory after successful replay.
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
