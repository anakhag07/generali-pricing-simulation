"""Replay context-averaged residual and transferred bands from a paired sweep."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.envelopes.reporting import _STYLE
from experiments.envelopes.wild import WildLinearDesign
from experiments.envelopes.wild_reporting import _validate_example
from experiments.envelopes.wild_study import _example, _sources
from experiments.provenance import file_record, write_provenance
from experiments.sweep_reporting import write_rows_csv


def _verified(path, record):
    actual = file_record(path)
    if any(actual[key] != record[key] for key in ("sha256", "bytes")):
        raise ValueError(f"Saved artifact changed: {path}")
    return actual


def _context_means(example):
    means = {key: example[key].mean(axis=0) for key in
             ("truth", "predictor", "residual_truth", "residual_fit")}
    for method in ("analytical", "bootstrap"):
        for side, sign in (("lower", -1), ("upper", 1)):
            means[f"{method}_residual_{side}"] = (
                example["residual_fit"] + sign * example[f"{method}_width"]
            ).mean(axis=0)
            # The saved per-context endpoints already include positive-part transfer.
            means[f"{method}_transferred_{side}"] = example[f"{method}_{side}"].mean(axis=0)
    return means


def render(summary_path, destination=None):
    summary_path = Path(summary_path)
    project = summary_path.parent
    summary = json.loads(summary_path.read_text())
    p = summary["manifest"]
    if p.get("sweep_mode") != "paired" or p.get("band_target") != "residual":
        raise ValueError("Require a paired residual-band sweep.")
    inputs = [file_record(summary_path)]
    metadata = []
    for path, record in ((project / "environment.json", summary["environment"]),
                         (project / "repetitions/rep-00000.json", summary["repetitions"][0])):
        inputs.append(_verified(path, record))
        metadata.append(json.loads(path.read_text()))
    environment, repetition = metadata
    if (environment["contract"]["manifest"] != p or repetition["index"] != 0
            or any(item["contract_sha256"] != summary["contract_sha256"] for item in metadata)):
        raise ValueError("Saved environment and repetition must match the summary.")
    sources = _sources()
    saved_sources = environment["contract"]["sources"]
    if (len(sources) != len(saved_sources) or any(
            actual[key] != saved[key] for actual, saved in zip(sources, saved_sources)
            for key in ("sha256", "bytes"))):
        raise ValueError("Replay requires the original mathematical implementation.")
    arrays = []
    for path, item in ((project / "environment.npz", environment),
                       (project / "repetitions/rep-00000.npz", repetition)):
        inputs.append(_verified(path, item["arrays"]))
        with np.load(path, allow_pickle=False) as saved:
            arrays.append(dict(saved))
    snapshot, first = arrays
    count = len(p["n_values"])
    if (len(p["B_values"]) != count
            or not np.array_equal(first["active_conditions"], np.eye(count, dtype=bool))):
        raise ValueError("Saved active conditions must match the paired schedule.")
    designs = [WildLinearDesign(snapshot["X"][:n], snapshot["actions"][:n],
                                action_degree=p["design"].get("action_degree", 1),
                                supremum_tolerance=p.get("supremum_tolerance", 1e-8))
               for n in p["n_values"]]
    delta = p["baseline"]["delta"]
    means = []
    for n, b in zip(p["n_values"], p["B_values"]):
        example = _example(p, snapshot, designs, first, baseline={"n": n, "B": b, "delta": delta})
        _validate_example({"X": snapshot["X"][:n]}, example)
        means.append(_context_means(example))

    destination = Path(destination) if destination else project / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    stem = "07_context_averaged_bands_by_pair"
    output = destination / f"{stem}.pdf"
    a = snapshot["actions_grid"]
    with plt.style.context("default"), plt.rc_context(_STYLE):
        fig, axes = plt.subplots(count, 2, figsize=(12, 3.2 * count), sharex=True,
                                 sharey="col", squeeze=False, constrained_layout=True)
        fig.suptitle(rf"Context-averaged bands: $\delta={delta:g}$, response repetition 0")
        for j, (n, b, item) in enumerate(zip(p["n_values"], p["B_values"], means)):
            for col, target in enumerate(("residual", "transferred")):
                ax = axes[j, col]
                for method, label, color in (("analytical", "OLS variance bound", "C0"),
                                             ("bootstrap", "Wild bootstrap", "C1")):
                    ax.fill_between(a, item[f"{method}_{target}_lower"],
                                    item[f"{method}_{target}_upper"],
                                    color=color, alpha=.2, label=label)
                if col == 0:
                    lines = (("residual_truth", r"Mean residual $\overline{g}$"),
                             ("residual_fit", r"Mean OLS fit $\overline{\widetilde g}$"))
                else:
                    lines = (("truth", r"Mean truth $\overline{f}$"),
                             ("predictor", r"Mean predictor $\overline{\widehat f}$"))
                for (key, label), color, style in zip(lines, ("C2", "C3"), ("-", "--")):
                    ax.plot(a, item[key], color=color, linestyle=style, label=label)
                title = "Residual band" if col == 0 else "Transferred envelope"
                ax.set_title(title + rf"; $(n,B)=({n},{b})$")
                ax.set_xlabel("Action $a$")
                ax.set_ylabel("Context-averaged residual" if col == 0 else "Context-averaged response")
        for ax in axes[0]:
            ax.legend()
        try:
            fig.savefig(output, format="pdf")
        finally:
            plt.close(fig)

    rows = [{"n": n, "B": b, "delta": delta, "action": float(action),
             **{key: float(value[k]) for key, value in item.items()}}
            for n, b, item in zip(p["n_values"], p["B_values"], means)
            for k, action in enumerate(a)]
    csv = destination / f"{stem}.csv"
    write_rows_csv(csv, rows, list(rows[0]))
    write_provenance(destination / f"{stem}.json", {
        "inputs": inputs, "source": file_record(__file__), "mathematical_sources": sources,
        "outputs": [file_record(output), file_record(csv)],
        "response_repetition": 0, "delta": delta,
        "context_average": "Equal weights over all n calibration contexts at each pair.",
        "transfer_order": "Transfer each context's endpoints, then average endpoints.",
        "interpretation": "Empirical-context mean bands; not averages over response repetitions.",
        "replay": "Saved fits and critical values; no new random draws or optimizer queries.",
    })
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(render(args.summary, args.destination))
