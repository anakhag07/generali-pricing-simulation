"""Create an explicitly synthetic declining tail for the profit lower bound."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_SOURCE_DIR = RESULTS_ROOT / "monotone-spline-xgb-support-cloud"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "spline-xgb-synthetic-tail-support"
DEFAULT_CUTOFF = 0.12
DEFAULT_TARGET_U = 0.20
DEFAULT_TARGET_LOWER_PROFIT = 120.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "monotone_spline_xgb_local_support_cloud.csv",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "run_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--target-u", type=float, default=DEFAULT_TARGET_U)
    parser.add_argument(
        "--target-lower-profit",
        type=float,
        default=DEFAULT_TARGET_LOWER_PROFIT,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_tail_lower_bound(
    frame: pd.DataFrame,
    *,
    cutoff: float,
    target_u: float,
    target_lower_profit: float,
) -> tuple[pd.DataFrame, float]:
    """Subtract a linear post-cutoff penalty calibrated at one target action."""
    required = {
        "u",
        "smoothed_mean_profit",
        "smoothed_support_half_width",
        "support_cloud_lower_profit",
        "support_cloud_upper_profit",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Source support CSV is missing columns: {missing}")
    result = frame.sort_values("u").reset_index(drop=True).copy()
    u = result["u"].to_numpy(dtype=float)
    if not cutoff < target_u or cutoff < u[0] or target_u > u[-1]:
        raise ValueError("Require grid_min <= cutoff < target_u <= grid_max.")
    matches = np.flatnonzero(np.isclose(u, target_u, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError("target_u must be represented exactly once on the grid.")
    original_lower = result["support_cloud_lower_profit"].to_numpy(dtype=float)
    endpoint_lower = float(original_lower[int(matches[0])])
    required_drop = endpoint_lower - float(target_lower_profit)
    if required_drop <= 0.0:
        raise ValueError("target_lower_profit must be below the original lower bound.")
    slope = required_drop / (float(target_u) - float(cutoff))
    tail_penalty = slope * np.maximum(u - float(cutoff), 0.0)
    synthetic_lower = original_lower - tail_penalty

    result["original_support_cloud_lower_profit"] = original_lower
    result["synthetic_tail_penalty"] = tail_penalty
    result["optimization_support_penalty"] = (
        result["smoothed_support_half_width"].to_numpy(dtype=float) + tail_penalty
    )
    result["support_cloud_lower_profit"] = synthetic_lower
    if not np.isclose(synthetic_lower[int(matches[0])], target_lower_profit):
        raise AssertionError("Synthetic tail calibration failed.")
    return result, float(slope)


def _plot(frame: pd.DataFrame, output_path: Path) -> None:
    u = frame["u"].to_numpy(dtype=float)
    mean = frame["smoothed_mean_profit"].to_numpy(dtype=float)
    lower = frame["support_cloud_lower_profit"].to_numpy(dtype=float)
    upper = frame["support_cloud_upper_profit"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(u, lower, upper, color="C0", alpha=0.2)
    ax.plot(u, mean, color="C0", linewidth=2.0)
    ax.set_title("Mean Predicted Profit Per Customer vs. Price Change", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> list[Path]:
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("support", {}).get("baseline_subtracted") is not False:
        raise ValueError("Source support must retain absolute baseline risk.")
    source = pd.read_csv(args.source_csv)
    synthetic, slope = synthetic_tail_lower_bound(
        source,
        cutoff=float(args.cutoff),
        target_u=float(args.target_u),
        target_lower_profit=float(args.target_lower_profit),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "synthetic_tail_support_cloud.csv"
    pdf_path = output_dir / "synthetic_tail_support_cloud.pdf"
    manifest_path = output_dir / "run_manifest.json"
    synthetic.to_csv(csv_path, index=False)
    _plot(synthetic, pdf_path)
    manifest = {
        **source_manifest,
        "analysis": "explicitly-synthetic-tail-support-cloud",
        "synthetic_modification": {
            "interpretation": "illustrative counterfactual, not estimated uncertainty",
            "formula": "tail_penalty(u) = slope * max(u - cutoff, 0)",
            "cutoff": float(args.cutoff),
            "target_u": float(args.target_u),
            "target_lower_profit": float(args.target_lower_profit),
            "slope_profit_per_decimal_u": slope,
            "mean_profit_modified": False,
            "upper_envelope_modified": False,
            "lower_envelope_modified": True,
        },
        "inputs": {
            "source_csv": {
                "path": str(args.source_csv.resolve()),
                "sha256": _sha256_file(args.source_csv),
            },
            "source_manifest": {
                "path": str(args.source_manifest.resolve()),
                "sha256": _sha256_file(args.source_manifest),
            },
        },
        "outputs": {
            csv_path.name: _sha256_file(csv_path),
            pdf_path.name: _sha256_file(pdf_path),
        },
        "optimizer": "not used by this construction step",
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
