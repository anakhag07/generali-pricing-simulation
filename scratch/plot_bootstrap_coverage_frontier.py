"""Replay saved pairs-bootstrap prefixes and plot the empirical coverage frontier.

Run with PYTHONPATH=src python scratch/plot_bootstrap_coverage_frontier.py.
No data generation, model refitting, or optimizer action selection is performed.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.bootstrap_band_continuous import QuadraticErrorCertificate
from experiments.paths import results_root
from experiments.policy_lcb.common import read_json, wilson_interval, write_json_atomic
from experiments.provenance import file_record
from experiments.sweep_reporting import write_rows_csv


def verify(record):
    # ORCD compute and login nodes expose the same files under different mounts.
    path = Path(record["path"])
    if not path.exists():
        path = Path(str(path).replace("/orcd/home/002/anakhag/", "/home/anakhag/"))
    actual = file_record(path)
    if any(actual[k] != record[k] for k in ("bytes", "sha256")):
        raise ValueError(f"Changed source artifact: {path}")
    return path


def replay(project):
    summary = read_json(project/"summary.json")
    payload = summary["manifest"]
    if payload["bootstrap_method"] != "pairs" or payload["baseline"]["sigma"] != 1:
        raise ValueError("Require the saved pairs-bootstrap experiment at sigma=1.")
    ns, bs = payload["axes"]["N"], payload["axes"]["B"]
    records = summary["datasets"]
    if len(records) != payload["datasets"]:
        raise ValueError("Incomplete source dataset collection.")
    covered = np.zeros((len(records), len(ns), len(bs)), dtype=bool)
    checked_sources, exact_fallbacks = set(), 0
    for i, record in enumerate(records):
        saved = read_json(verify(record))
        if saved["contract"]["manifest"] != payload or saved["contract"]["dataset"] != i:
            raise ValueError("Source dataset contract differs.")
        for source in saved["contract"]["sources"]:
            key = (source["path"], source["sha256"])
            if key not in checked_sources:
                verify(source)
                checked_sources.add(key)
        n_rows = {r["N"]: r for r in saved["rows"] if r["axis"] == "N"}
        with np.load(verify(saved["arrays"])) as arrays:
            for ni, n in enumerate(ns):
                row = n_rows[n]
                bounds = arrays[f"N{n}_bootstrap_supremum_bounds"]
                for bi, b in enumerate(bs):
                    if b > len(bounds):
                        raise ValueError("Requested B exceeds saved bootstrap draws.")
                    critical = float(np.quantile(bounds[:b, 1], 1-payload["delta"], method="higher"))
                    if row["observed_statistic_upper"] <= critical:
                        hit = True
                    elif row["observed_statistic_lower"] > critical:
                        hit = False
                    else:
                        exact_fallbacks += 1
                        cert = QuadraticErrorCertificate(arrays[f"N{n}_covariance"], row["sigma_hat"])
                        error = cert.difference(arrays[row["array_prefix"]+"_beta_hat"],
                                                np.array(payload["truth"]["coefficients"]))
                        hit = cert.contains(error, critical)
                    covered[i, ni, bi] = hit
            # Independently check every previously reported N/B coverage event.
            for row in saved["rows"]:
                if row["axis"] in {"N", "B"}:
                    assert covered[i, ns.index(row["N"]), bs.index(row["B"])] == row["covered"]
    target = Fraction(1)-Fraction(str(payload["delta"]))
    rows, costs = [], []
    for ni, n in enumerate(ns):
        for bi, b in enumerate(bs):
            hits = int(covered[:, ni, bi].sum())
            lo, hi = wilson_interval(hits, len(records))
            # Integer arithmetic preserves ties equally far above/below target.
            deviation = abs(hits*target.denominator-len(records)*target.numerator)
            costs.append((n, b, deviation))
            rows.append({"N": n, "B": b, "sigma": 1., "datasets": len(records),
                         "covered_count": hits, "coverage": hits/len(records),
                         "coverage_lower": lo, "coverage_upper": hi,
                         "absolute_coverage_error": deviation/(len(records)*target.denominator)})
    costs = np.asarray(costs, dtype=np.int64)
    # Descriptive dominance among measured configurations, not action optimization.
    for row, candidate in zip(rows, costs):
        dominated = np.all(costs <= candidate, axis=1) & np.any(costs < candidate, axis=1)
        row["pareto"] = not bool(np.any(dominated))
    return summary, rows, covered, exact_fallbacks


def plot(rows, payload, path):
    frontier = [r for r in rows if r["pareto"]]
    style = {"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12,
             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
             "figure.titlesize": 16}
    with plt.rc_context(style):
        fig, (ax, ci) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(f"Empirical coverage Pareto frontier: {payload['datasets']} datasets, "
                     r"$\sigma=1$, target coverage $95\%$")
        points = ax.scatter([r["N"] for r in rows], [r["B"] for r in rows],
                            c=[r["coverage"] for r in rows], cmap="viridis")
        ax.scatter([r["N"] for r in frontier], [r["B"] for r in frontier],
                   facecolors="none", edgecolors="black", s=100,
                   label="Empirical Pareto points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Training sample size $N$")
        ax.set_ylabel(r"Bootstrap refit count $B$")
        ax.set_title(r"Minimize $N$, $B$, and $|\widehat C-0.95|$")
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label(r"Simultaneous coverage $\widehat C$")
        ax.legend()
        y = np.arange(len(frontier))
        coverage = np.array([r["coverage"] for r in frontier])
        errors = np.array([[r["coverage"]-r["coverage_lower"] for r in frontier],
                           [r["coverage_upper"]-r["coverage"] for r in frontier]])
        ci.errorbar(coverage, y, xerr=errors, fmt="o", capsize=3, label="95% Wilson intervals")
        ci.axvline(1-payload["delta"], color="C1", linestyle="--", label="Target 95%")
        ci.set_yticks(y, [f"({r['N']}, {r['B']})" for r in frontier])
        ci.set_xlabel("Simultaneous coverage")
        ci.set_ylabel(r"Empirical Pareto configuration $(N,B)$")
        ci.set_title("Pointwise intervals; selection is exploratory")
        ci.legend()
        fig.savefig(path, format="pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path,
                        default=results_root()/"bootstrap-ols-pairs-controlled-sweep")
    args = parser.parse_args()
    summary, rows, covered, fallbacks = replay(args.project)
    output = args.project/"reports"/"coverage_frontier"
    output.mkdir(parents=True, exist_ok=True)
    csv = output/"coverage_grid.csv"
    write_rows_csv(csv, rows, tuple(rows[0]))
    indicators = output/"coverage_indicators.npz"
    np.savez_compressed(indicators, covered=covered,
                        N=summary["manifest"]["axes"]["N"], B=summary["manifest"]["axes"]["B"])
    pdf = output/"coverage_pareto_frontier.pdf"
    plot(rows, summary["manifest"], pdf)
    write_json_atomic(output/"provenance.json", {
        "source_summary": file_record(args.project/"summary.json"),
        "source_datasets": summary["datasets"], "script": file_record(Path(__file__)),
        "outputs": [file_record(p) for p in (csv, indicators, pdf)],
        "settings": len(rows), "frontier": [r for r in rows if r["pareto"]],
        "exact_containment_fallbacks": fallbacks,
        "dominance": "Minimize N, B, and absolute deviation of empirical coverage from 1-delta.",
        "coverage": "Whole-real-line containment; original upper-certified bootstrap quantile convention.",
        "uncertainty": "100 independent outer datasets, paired across N/B. Wilson intervals are pointwise, not selection-adjusted.",
        "replay": "Existing bootstrap prefixes only; no new random draws, refits, or optimizer actions.",
    })
    print(pdf)
    print(f"{len(rows)} configurations; {sum(r['pareto'] for r in rows)} empirical Pareto points")
    for row in rows:
        if row["pareto"]:
            print(row)


if __name__ == "__main__":
    main()
