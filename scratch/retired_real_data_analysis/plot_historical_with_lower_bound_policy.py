"""Overlay historical prices with a saved lower-bound optimized policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.loader import load_observed_u_array


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT.parent.parent / "results" if ROOT.parent.name == "worktrees" else ROOT / "results"
)
DEFAULT_POLICY_DIR = (
    RESULTS_ROOT / "spline-xgb-synthetic-tail-140-lower-bound-policy-20k"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "spline-xgb-synthetic-tail-140-historical-overlay"
DEFAULT_REFERENCE_PDF = Path(
    "/home/anakhag/.codex/attachments/"
    "52144191-70c9-44b0-a06b-198f569a8d95/"
    "historical_vs_optimized_decimal_minus0.1_to_0.2_overlay (1).pdf"
)
BOUNDS = (-0.1, 0.2)
N_BINS = 30
HISTORICAL_COLOR = "#737373"
OPTIMIZED_COLOR = "#86002d"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        default=DEFAULT_POLICY_DIR / "support_lower_bound_policy_full_actions.npz",
    )
    parser.add_argument(
        "--policy-summary",
        type=Path,
        default=DEFAULT_POLICY_DIR / "summary.json",
    )
    parser.add_argument("--reference-pdf", type=Path, default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _load_policy(
    artifact_path: Path,
    summary_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(artifact_path, allow_pickle=False) as loaded:
        actions = np.asarray(loaded["actions"], dtype=float)
        row_indices = np.asarray(loaded["row_indices"], dtype=np.int64)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary["full_application"]
    if len(actions) != int(expected["n_customers"]):
        raise ValueError("Saved actions do not match the full-population summary.")
    if actions.shape != row_indices.shape or len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("Policy rows must be unique and aligned one-to-one with actions.")
    if _sha256_indices(row_indices) != expected["row_indices_sha256"]:
        raise ValueError("Policy row-index checksum does not match its summary.")
    if not np.all(np.isfinite(actions)) or not np.all(
        (actions >= BOUNDS[0]) & (actions <= BOUNDS[1])
    ):
        raise ValueError("Saved policy actions must be finite and inside [-0.1, 0.2].")
    return actions, row_indices, summary


def _histogram_frame(
    historical: np.ndarray,
    optimized: np.ndarray,
) -> pd.DataFrame:
    edges = np.linspace(BOUNDS[0], BOUNDS[1], N_BINS + 1)
    rows = []
    for series, values in (("Historical", historical), ("Optimized", optimized)):
        counts, _ = np.histogram(values, bins=edges)
        density, _ = np.histogram(values, bins=edges, density=True)
        for left, right, count, value in zip(
            edges[:-1], edges[1:], counts, density, strict=True
        ):
            rows.append(
                {
                    "series": series,
                    "bin_left": left,
                    "bin_right": right,
                    "count": int(count),
                    "density": value,
                }
            )
    return pd.DataFrame(rows)


def _plot(
    historical: np.ndarray,
    optimized: np.ndarray,
    output_path: Path,
) -> None:
    edges = np.linspace(BOUNDS[0], BOUNDS[1], N_BINS + 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    ax.hist(
        historical,
        bins=edges,
        density=True,
        color=HISTORICAL_COLOR,
        edgecolor=HISTORICAL_COLOR,
        linewidth=0.5,
        alpha=0.55,
        label="Historical",
    )
    ax.hist(
        optimized,
        bins=edges,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
        label="Optimized",
    )
    ax.set_xlim(*BOUNDS)
    ax.set_title("Historical and Optimized Price Changes", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> list[Path]:
    actions, row_indices, summary = _load_policy(
        args.policy_artifact,
        args.policy_summary,
    )
    historical = np.asarray(load_observed_u_array("xgb", n_rows=None), dtype=float)
    if historical.shape != actions.shape:
        raise ValueError("Historical and optimized arrays must cover the same customers.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "historical_vs_optimized_target140_overlay.pdf"
    csv_path = output_dir / "historical_vs_optimized_target140_histogram.csv"
    manifest_path = output_dir / "run_manifest.json"
    histogram = _histogram_frame(historical, actions)
    histogram.to_csv(csv_path, index=False)
    _plot(historical, actions, pdf_path)

    manifest = {
        "analysis": "historical-vs-synthetic-tail-140-lower-bound-policy-overlay",
        "display": {
            "title": "Historical and Optimized Price Changes",
            "x_axis": "Price Change",
            "y_axis": "Density",
            "bounds": list(BOUNDS),
            "n_bins": N_BINS,
            "historical": {
                "label": "Historical",
                "color": HISTORICAL_COLOR,
                "alpha": 0.55,
            },
            "optimized": {
                "label": "Optimized",
                "color": OPTIMIZED_COLOR,
                "alpha": 0.68,
            },
        },
        "population": {
            "n_customers": int(len(actions)),
            "row_indices_sha256": _sha256_indices(row_indices),
        },
        "policy": {
            "source": str(args.policy_artifact.resolve()),
            "source_sha256": _sha256_file(args.policy_artifact),
            "summary": str(args.policy_summary.resolve()),
            "summary_sha256": _sha256_file(args.policy_summary),
            "optimizer_entry_point": summary["optimizer"]["entry_point"],
            "optimizer_success": bool(summary["optimizer"]["success"]),
            "target_lower_profit": 140.0,
        },
        "reference_pdf": {
            "path": str(args.reference_pdf.resolve()),
            "sha256": _sha256_file(args.reference_pdf),
            "preserved": "title, axes, bounds, bins, colors, alpha, and legend labels",
        },
        "outputs": {
            pdf_path.name: _sha256_file(pdf_path),
            csv_path.name: _sha256_file(csv_path),
        },
        "optimizer": "not rerun; exact saved repository-optimizer output replayed",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = [pdf_path, csv_path, manifest_path]
    for output in outputs:
        print(output, flush=True)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
