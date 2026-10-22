"""Illustrate dimension scaling at n=12p using the repository's linear adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.envelopes.core import CalibrationData, TargetDomain
from experiments.envelopes.linear import LinearModel, fit_linear_transfer
from experiments.envelopes.reporting import _STYLE, _comparison_case
from experiments.paths import results_root
from experiments.provenance import file_record, write_provenance
from experiments.seeds import derive_seed
from experiments.sweep_reporting import write_rows_csv

P_VALUES = np.arange(5, 101, 5)


def _coefficients(original, dimension):
    # Keep the original signal and predictor mismatch; add nuisance coordinates.
    return np.r_[original[:-1], np.zeros(dimension-(len(original)-2)), original[-1]]


def _grid_means(fitted, X, actions):
    totals = {key: np.zeros(len(actions)) for key in ("radius", "width", "discrepancy")}
    for offset in range(0, len(X), 64):
        block = X[offset:offset+64]
        values = fitted.evaluate(np.repeat(block, len(actions), axis=0),
                                 np.tile(actions, len(block)))
        quantities = {"radius": values.radius, **values.components}
        for key in totals:
            totals[key] += quantities[key].reshape(len(block), -1).sum(axis=0)
    return {key: value/len(X) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=results_root())
    args = parser.parse_args()
    source = args.runs_root / "uncertainty-envelopes-writeup"
    summary, original_points, provenance = _comparison_case(source)
    config = summary["manifest"]
    if (config["calibration"]["n"] != 60 or config["calibration"]["context_dim"] != 3
            or config["envelope"]["type"] != "linear_known_variance_transfer"):
        raise ValueError("This illustration extends the saved n=60, d=3 linear example.")
    settings = config["envelope"]
    if settings.get("width_multiplier", 1) != 1:
        raise ValueError("The displayed scaling identity assumes width multiplier one.")
    seeds = config["seeds"]
    sigma, delta = settings["sigma_bound"], settings["delta"]
    n_max, d_max = 12*int(P_VALUES[-1]), int(P_VALUES[-1])-2
    contexts_rng = np.random.default_rng(seeds["contexts"])
    # Drawing original coordinates first preserves the original 60x3 dataset.
    original_X = contexts_rng.normal(size=(n_max, 3))
    extra_X = contexts_rng.normal(size=(n_max, d_max-3))
    X = np.column_stack((original_X, extra_X))
    actions = np.random.default_rng(seeds["actions"]).normal(size=n_max)
    observation_seed = derive_seed(seeds["observations"], "repetition:0")
    errors = config["calibration"]["noise"]["std"] * np.random.default_rng(observation_seed).normal(size=n_max)
    grid = original_points["actions"].reshape(60, -1)[0]
    truth_beta = np.asarray(config["problem"]["truth"]["coefficients"])
    predictor_beta = np.asarray(config["problem"]["predictor"]["coefficients"])
    baseline_y = LinearModel(truth_beta)(X[:60, :3], actions[:60])+errors[:60]
    with np.load(provenance["artifacts"]["calibration"]["path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(X[:60, :3], saved["X"])
        np.testing.assert_array_equal(actions[:60], saved["actions"])
        np.testing.assert_allclose(baseline_y, saved["y"], rtol=1e-13, atol=1e-13)

    destination = args.runs_root / "uncertainty-envelopes-dimension-sweep"
    destination.mkdir(parents=True, exist_ok=True)
    inputs = destination / "paired_inputs.npz"
    np.savez_compressed(inputs, X=X, actions=actions, errors=errors, diagnostic_actions=grid)
    records, artifacts = [], []
    for p in P_VALUES:
        p = int(p)
        dimension, n = p-2, 12*p
        contexts, logged_actions = X[:n, :dimension], actions[:n]
        truth = LinearModel(_coefficients(truth_beta, dimension))
        predictor = LinearModel(_coefficients(predictor_beta, dimension))
        data = CalibrationData(contexts, logged_actions,
                               truth(contexts, logged_actions)+errors[:n])
        fitted = fit_linear_transfer(data, predictor, TargetDomain(contexts), settings, None)
        means = _grid_means(fitted, contexts, grid)
        np.testing.assert_allclose(means["radius"], means["width"]+means["discrepancy"],
                                   rtol=1e-12, atol=1e-12)
        logged_width = fitted.evaluate(contexts, logged_actions).components["width"]
        logged_rms = float(np.sqrt(np.mean(logged_width**2)))
        exact_logged_rms = sigma*p/np.sqrt(n*delta)
        np.testing.assert_allclose(logged_rms, exact_logged_rms, rtol=1e-11, atol=1e-11)
        if p == 5:
            for key, saved_key in (("radius", "radius"), ("width", "width"),
                                   ("discrepancy", "discrepancy")):
                np.testing.assert_allclose(means[key], original_points[saved_key].reshape(60, -1).mean(axis=0),
                                           rtol=1e-11, atol=1e-11)
        record = {"p": p, "d": dimension, "n": n,
                  **{f"mean_{key}": float(value.mean()) for key, value in means.items()},
                  "logged_width_rms": logged_rms, "exact_logged_width_rms": exact_logged_rms}
        records.append(record)
        fit_path = destination / f"p{p:03d}.npz"
        np.savez_compressed(fit_path, **fitted.state, **means,
                            truth_coefficients=truth.coefficients,
                            predictor_coefficients=predictor.coefficients)
        artifacts.append(file_record(fit_path))
        print(f"Completed p={p}, n={n}", flush=True)

    csv_path = destination / "dimension_sweep.csv"
    write_rows_csv(csv_path, records, list(records[0]))
    pdf_path = destination / "mean_radius_vs_dimension.pdf"
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        p = np.array([row["p"] for row in records])
        width = np.array([row["mean_width"] for row in records])
        ax.plot(p, width, "--", color="C1", label=r"$\overline{w}_\delta$")
        ax.plot(p, width[0]*np.sqrt(p/p[0]), ":", color="0.5", label=r"$\sqrt{p}$ reference")
        ax.set(title=r"Linear-envelope dimension scaling: $n=12p$", xlabel=r"$p=d+2$", ylabel="Mean half-width")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.supxlabel(
            rf"Means over calibration contexts and {len(grid)} actions in $[-2,2]$; $\sigma={sigma:g}$, $\delta={delta:g}$."
            "\n" + r"Each $\sqrt{p}$ reference matches $\overline{w}_\delta$ at $p=5$; datasets are paired across the sweep.",
            fontsize=10,
        )
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)

    experiment = destination / "EXPERIMENT.md"
    experiment.write_text(
        "# Linear-envelope dimension sweep\n\n"
        "Vary p=d+2 over 5,10,...,100, keeping sigma=1 and delta=.05 with n=12p. "
        "The original truth and frozen predictor retain their intercepts, "
        "first three context coefficients, and action slopes; extra coefficients are zero. "
        "The auxiliary OLS model estimates all p coefficients using the repository's "
        "fit_linear_transfer adapter. No optimizer or optimum is involved.\n\n"
        "Every plotted mean weights calibration contexts and the 101 diagnostic actions "
        "in [-2,2] equally. The radius is a half-width. The total radius is absolute "
        "predictor/auxiliary discrepancy plus w_delta. These arithmetic means are not "
        "the logged-row RMS. The separate exact RMS identity sigma*p/sqrt(n*delta) is "
        "checked at every setting. The sqrt(p) reference is anchored at p=5, not fitted.\n\n"
        "Contexts and logged actions are independent standard normals; observation errors "
        "are independent Gaussian draws with SD 1. There is one paired calibration draw, "
        "not a repeated-dataset coverage experiment. Existing separate context/action/" 
        "observation streams are retained; the deterministic OLS fit consumes no method RNG. "
        "Generate all original three context coordinates first, then extra coordinates, "
        "and use row/column prefixes across n and p. Actions and errors use the same "
        "row prefixes. The n=60,p=5 case reproduces the saved source calibration and "
        "radius-component curves; no extra signal is introduced by the nuisance dimensions.\n\n"
        "Truth, predictor, auxiliary model, and radius remain analytic functions of real "
        "actions. The formal linear-model guarantee is over all real actions at the target "
        "contexts under the original assumptions; this sweep does not rerun continuum "
        "certification or estimate coverage. Diagnostic actions only define the averaging "
        "measure. Lines between sampled p values are visualization, not an off-grid "
        "experiment. There are no optimization queries, action boundaries, or global "
        "reference computations.\n\n"
        "Regenerate with `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 "
        "python scratch/plot_linear_envelope_dimension_sweep.py`. Inputs, fit states, "
        "component means, and source identities are retained beside the vector PDF.\n"
    )
    write_provenance(destination / "provenance.json", {
        "source_experiment": provenance, "settings": settings, "seed_roots": seeds,
        "observation_seed": observation_seed, "p_values": P_VALUES,
        "sample_size": "n=12*p",
        "averaging": "Uniform calibration contexts crossed with the saved diagnostic actions.",
        "paired_inputs": file_record(inputs), "fits": artifacts,
        "summary": file_record(csv_path), "plot": file_record(pdf_path),
        "documentation": file_record(experiment),
        "sources": [file_record(__file__), file_record(Path(__file__).parents[1]/"src/experiments/envelopes/linear.py"),
                    file_record(Path(__file__).parents[1]/"src/experiments/envelopes/core.py")],
    })
    print(pdf_path)


if __name__ == "__main__":
    main()
