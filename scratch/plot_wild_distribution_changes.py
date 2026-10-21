"""Replay distribution changes and paired mean-half-width comparisons (MATH §7.16)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, norm, t

from experiments.envelopes.reporting import _STYLE
from experiments.envelopes.wild_reporting import _distribution_comparison_label, _paired_axis
from experiments.provenance import file_record, write_provenance
from experiments.sweep_reporting import write_rows_csv


def _verified(path, expected):
    actual = file_record(path)
    if any(actual[key] != expected[key] for key in ("sha256", "bytes")):
        raise ValueError(f"Saved artifact changed: {path}")
    return actual


def _load_study(summary_path):
    summary_path = Path(summary_path)
    project = summary_path.parent
    summary = json.loads(summary_path.read_text())
    p = summary["manifest"]
    if p.get("band_target") != "residual" or p.get("sweep_mode") != "paired":
        raise ValueError("Require a completed paired residual sweep.")
    inputs = [file_record(summary_path)]
    metadata_path = project / "environment.json"
    inputs.append(_verified(metadata_path, summary["environment"]))
    environment = json.loads(metadata_path.read_text())
    if (environment["contract"]["manifest"] != p
            or environment["contract_sha256"] != summary["contract_sha256"]):
        raise ValueError("Environment and summary contracts disagree.")
    inputs.append(_verified(project / "environment.npz", environment["arrays"]))
    with np.load(project / "environment.npz", allow_pickle=False) as data:
        snapshot = dict(data)
    repetitions, pairs, deltas = p["repetitions"], len(p["n_values"]), len(p["deltas"])
    if len(summary["repetitions"]) != repetitions:
        raise ValueError("Incomplete saved repetition list.")
    widths = np.empty((repetitions, pairs, deltas))
    pairing = []
    for index, expected in enumerate(summary["repetitions"]):
        path = project / "repetitions" / f"rep-{index:05d}.json"
        inputs.append(_verified(path, expected))
        metadata = json.loads(path.read_text())
        if metadata["index"] != index or metadata["contract_sha256"] != summary["contract_sha256"]:
            raise ValueError("Repetition and summary contracts disagree.")
        inputs.append(_verified(path.with_suffix(".npz"), metadata["arrays"]))
        pairing.append((metadata["seeds"], metadata["multiplier_shape"], metadata["multiplier_sha256"]))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            if not np.array_equal(data["active_conditions"], np.eye(pairs, dtype=bool)):
                raise ValueError("Saved active conditions must follow the paired schedule.")
            critical = data["critical"]
            widths[index] = critical[np.arange(pairs), np.arange(pairs), :, 1] * snapshot["mean_ell"][:, None]
            if index == 0:
                innovations = data["innovations"].copy()
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("Invalid residual half-widths.")
    # This also guards the intended residual-width target and diagonal indexing.
    for j, (n, B) in enumerate(zip(p["n_values"], p["B_values"])):
        for k, delta in enumerate(p["deltas"]):
            saved = [r for r in summary["metrics"] if (r["n"], r["B"], r["delta"], r["method"])
                     == (n, B, delta, "wild_bootstrap")]
            if len(saved) != 1 or not np.isclose(widths[:, j, k].mean(), saved[0]["mean_width"], rtol=1e-12, atol=1e-14):
                raise ValueError("Reconstructed widths disagree with the original summary.")
    print(f"Verified {repetitions} saved repetitions: {p['name']}", flush=True)
    return {"summary": summary, "snapshot": snapshot, "innovations": innovations,
            "widths": widths, "pairing": pairing, "inputs": inputs, "project": project}


def _noise_figure(baseline, variant):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle("Noise change: Gaussian to centered exponential")
    z = np.unique(np.r_[np.linspace(-4, 8, 1601), -1.000001, -1.])
    for law, label in ((norm, r"Gaussian $N(0,1)$"), (expon(loc=-1), r"Centered exponential $E-1$")):
        axes[0].plot(z, law.pdf(z), label=label)
    axes[0].set(title="Standardized noise: mean 0, variance 1",
                xlabel=r"Standardized innovation $\xi$", ylabel="Probability density")
    axes[0].legend()
    errors = [study["snapshot"]["sigma"] * study["innovations"] for study in (baseline, variant)]
    bins = np.histogram_bin_edges(np.concatenate(errors), bins=50)
    for values, label in zip(errors, ("Gaussian noise", "Centered exponential noise")):
        axes[1].hist(values, bins=bins, density=True, histtype="step", label=label)
    count = len(errors[0])
    axes[1].set(title=f"Saved heteroskedastic noise: n={count}, repetition 0",
                xlabel=r"Observation error $\epsilon_i=\sigma_i\xi_i$", ylabel="Empirical density")
    axes[1].legend()
    return fig


def _source_values(study):
    snapshot = study["snapshot"]
    return np.concatenate((snapshot["X"].ravel(), snapshot["actions"]))


def _design_figure(baseline, variant):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    fig.suptitle(r"Source-data change: Gaussian to standardized $t_3$")
    x = np.linspace(-6, 6, 1601)
    axes[0].plot(x, norm.pdf(x), label=r"Gaussian $N(0,1)$")
    axes[0].plot(x, t.pdf(x, df=3, scale=1/np.sqrt(3)), label=r"Standardized $T_3/\sqrt{3}$")
    axes[0].set(title="Generating densities: mean 0, variance 1",
                xlabel="Source coordinate or logged action", ylabel="Probability density")
    axes[0].legend()
    for study, label in ((baseline, "Gaussian design"), (variant, r"Standardized $t_3$ design")):
        values = np.sort(np.abs(_source_values(study)))
        survival = np.arange(len(values), 0, -1) / len(values)
        # P(|V| >= v), including every saved extreme; no clipping or new draws.
        axes[1].step(np.r_[0., values], np.r_[1., survival], where="pre", label=label)
    axes[1].set_yscale("log")
    axes[1].set(title=f"Saved data: {len(values):,} pooled context/action values",
                xlabel=r"Absolute source value $v$", ylabel=r"Observed fraction with $|V|\geq v$")
    axes[1].legend()
    return fig


def _width_statistics(baseline, variant, comparisons):
    a, b = baseline["widths"], variant["widths"]
    if a.shape != b.shape or baseline["pairing"] != variant["pairing"]:
        raise ValueError("Paired width inference requires identical repetitions, seeds and multipliers.")
    p = variant["summary"]["manifest"]
    repetitions = len(a)
    point_quantile = t.ppf(.975, df=repetitions-1)
    adjusted_quantile = t.ppf(1-.05/(2*comparisons), df=repetitions-1)
    difference = b-a
    base_mean, changed_mean = a.mean(axis=0), b.mean(axis=0)
    base_se, changed_se, difference_se = (v.std(axis=0, ddof=1)/np.sqrt(repetitions) for v in (a, b, difference))
    rows = []
    for j, (n, B) in enumerate(zip(p["n_values"], p["B_values"])):
        for k, delta in enumerate(p["deltas"]):
            mean = difference[:, j, k].mean()
            se = difference_se[j, k]
            low, high = mean-adjusted_quantile*se, mean+adjusted_quantile*se
            rows.append({"n": n, "B": B, "delta": delta, "repetitions": repetitions,
                         "gaussian_mean_half_width": base_mean[j, k],
                         "variant_mean_half_width": changed_mean[j, k],
                         "gaussian_mean_ci95_half_width": point_quantile*base_se[j, k],
                         "variant_mean_ci95_half_width": point_quantile*changed_se[j, k],
                         "paired_difference": mean, "relative_change_percent": 100*mean/base_mean[j, k],
                         "paired_standard_error": se,
                         "paired_ci95_lower": mean-point_quantile*se,
                         "paired_ci95_upper": mean+point_quantile*se,
                         "adjusted_ci95_lower": low, "adjusted_ci95_upper": high,
                         "family_comparisons": comparisons,
                         "different_from_zero_adjusted": bool(low > 0 or high < 0)})
    return rows


def _width_figure(variant, label, rows):
    p = variant["summary"]["manifest"]
    fig, axes = plt.subplots(2, len(p["deltas"]), figsize=(5*len(p["deltas"]), 8),
                             sharex=True, sharey="row", squeeze=False, constrained_layout=True)
    fig.suptitle("Mean residual-band half-width: wild bootstrap")
    for col, delta in enumerate(p["deltas"]):
        data = [row for row in rows if row["delta"] == delta]
        x = np.arange(len(data))
        top, bottom = axes[:, col]
        for prefix, name in (("gaussian", "Gaussian baseline"), ("variant", label)):
            top.errorbar(x, [r[f"{prefix}_mean_half_width"] for r in data],
                         yerr=[r[f"{prefix}_mean_ci95_half_width"] for r in data],
                         marker="o", capsize=3, label=name)
        difference = np.array([r["paired_difference"] for r in data])
        interval = np.array([difference-np.array([r["adjusted_ci95_lower"] for r in data]),
                             np.array([r["adjusted_ci95_upper"] for r in data])-difference])
        bottom.errorbar(x, difference, yerr=interval, marker="o", capsize=3,
                        label="Paired change with adjusted 95% CI")
        bottom.axhline(0, color="0.5", linestyle="--", label="No change")
        for ax in (top, bottom):
            _paired_axis(ax, p)
            ax.set_title(rf"$\delta={delta:g}$")
        top.set_ylabel("Mean half-width")
        bottom.set_ylabel("Half-width change (variant − Gaussian)")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.supxlabel(f"Means over contexts and actions in [−2, 2], then {p['repetitions']:,} paired repetitions.\n"
                  f"Monte Carlo intervals: pointwise 95% above; Bonferroni 95% across {rows[0]['family_comparisons']} comparisons below.",
                  fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    baseline = _load_study(args.baseline)
    variants = [_load_study(path) for path in args.summaries]
    labels = [_distribution_comparison_label(baseline["summary"], v["summary"]) for v in variants]
    p = baseline["summary"]["manifest"]
    comparisons = len(variants)*len(p["n_values"])*len(p["deltas"])
    with plt.style.context("default"), plt.rc_context(_STYLE):
        for variant, label in zip(variants, labels):
            destination = variant["project"] / "plots/comparison"
            destination.mkdir(parents=True, exist_ok=True)
            noise_variant = variant["summary"]["manifest"]["noise"].get("innovation_distribution") == "centered_exponential"
            if noise_variant:
                for key in ("X", "actions", "sigma"):
                    np.testing.assert_array_equal(baseline["snapshot"][key], variant["snapshot"][key])
                distribution = _noise_figure(baseline, variant)
                name = "09_noise_distribution.pdf"
            else:
                np.testing.assert_array_equal(baseline["innovations"], variant["innovations"])
                distribution = _design_figure(baseline, variant)
                name = "09_source_data_distribution.pdf"
            rows = _width_statistics(baseline, variant, comparisons)
            outputs = []
            for filename, fig in ((name, distribution),
                                  ("10_mean_half_width_vs_pairs.pdf", _width_figure(variant, label, rows))):
                path = destination / filename
                try:
                    fig.savefig(path, format="pdf")
                finally:
                    plt.close(fig)
                outputs.append(path)
            csv = destination / "10_mean_half_width_statistics.csv"
            write_rows_csv(csv, rows, list(rows[0]))
            outputs.append(csv)
            write_provenance(destination / "distribution_and_width_provenance.json", {
                "source": file_record(__file__), "math": "MATH §7.16",
                "inputs": [*baseline["inputs"], *variant["inputs"]],
                "outputs": [file_record(path) for path in outputs],
                "width_target": "Wild-bootstrap residual half-width, mean over saved contexts and 101 actions in [-2,2], then repetitions.",
                "paired_inference": "Student-t Monte Carlo intervals on within-repetition width differences; Bonferroni over all supplied variant/pair/delta comparisons.",
                "conditioning": "Frozen designs and scale profile; does not account for redrawing designs or certify band coverage.",
                "distribution_plot": "Known generating densities and saved repetition-0 noise or all saved source coordinates/actions. No new random draws.",
                "family_comparisons": comparisons,
            })
            print(label, flush=True)
            for row in rows:
                if row["delta"] == p["baseline"]["delta"]:
                    print({key: row[key] for key in ("n", "B", "gaussian_mean_half_width", "variant_mean_half_width",
                          "relative_change_percent", "adjusted_ci95_lower", "adjusted_ci95_upper", "different_from_zero_adjusted")}, flush=True)
            for path in outputs:
                print(path, flush=True)


if __name__ == "__main__":
    main()
