"""Overlay saved optimized price changes on the spline/XGBoost support cloud.

The optimized-price histogram is an exact replay of a saved repository
first-order policy output. This script performs no optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.provenance import file_sha256 as _sha256_file
from reporting.real_data import plot_support_action_overlay as _plot_overlay


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_SUPPORT_DIR = RESULTS_ROOT / "monotone-spline-xgb-support-cloud"
DEFAULT_HISTOGRAM_DIR = (
    RESULTS_ROOT
    / "policy-histogram-analysis"
    / "glm-softmax-80-20-first-order"
    / "decimal-range-minus0.1-to-0.2"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUPPORT_DIR
OUTPUT_NAME = "mean_profit_support_cloud_with_optimized_price_changes.pdf"
MANIFEST_NAME = "mean_profit_support_cloud_with_optimized_price_changes_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support-csv",
        type=Path,
        default=DEFAULT_SUPPORT_DIR / "monotone_spline_xgb_local_support_cloud.csv",
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=DEFAULT_SUPPORT_DIR / "run_manifest.json",
    )
    parser.add_argument(
        "--histogram-csv",
        type=Path,
        default=DEFAULT_HISTOGRAM_DIR / "histogram_bin_data_decimal.csv",
    )
    parser.add_argument(
        "--histogram-config",
        type=Path,
        default=DEFAULT_HISTOGRAM_DIR / "analysis_config.json",
    )
    parser.add_argument(
        "--source-pdf",
        type=Path,
        default=(
            DEFAULT_HISTOGRAM_DIR
            / "plots"
            / "optimized_price_changes_decimal_minus0.1_to_0.2.pdf"
        ),
        help="Original optimized-price PDF retained for provenance.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _load_support(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "u",
        "smoothed_mean_profit",
        "support_cloud_lower_profit",
        "support_cloud_upper_profit",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Support-cloud CSV is missing columns: {missing}")
    frame = frame.sort_values("u").reset_index(drop=True)
    if len(frame) != 301 or not np.allclose(
        frame["u"].to_numpy(dtype=float),
        np.linspace(-0.1, 0.2, 301),
    ):
        raise ValueError("Support cloud must use the 301-point [-0.1, 0.2] grid.")
    if not np.isfinite(frame.loc[:, sorted(required)].to_numpy(dtype=float)).all():
        raise ValueError("Support-cloud values must be finite.")
    return frame


def _load_optimized_histogram(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"plot", "series", "bin_left", "bin_right", "density"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Histogram CSV is missing columns: {missing}")
    selected = frame.loc[
        (frame["plot"] == "optimized") & (frame["series"] == "Optimized")
    ].sort_values("bin_left")
    if len(selected) != 30:
        raise ValueError("Expected 30 optimized-policy histogram bins.")
    left = selected["bin_left"].to_numpy(dtype=float)
    right = selected["bin_right"].to_numpy(dtype=float)
    density = selected["density"].to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([left, right, density])).all():
        raise ValueError("Optimized histogram values must be finite.")
    if not np.all(right > left) or not np.allclose(right[:-1], left[1:]):
        raise ValueError("Optimized histogram bins must be positive and contiguous.")
    if not np.isclose(left[0], -0.1) or not np.isclose(right[-1], 0.2):
        raise ValueError("Optimized histogram must span [-0.1, 0.2].")
    if np.any(density < 0.0):
        raise ValueError("Optimized histogram density must be non-negative.")
    return selected.reset_index(drop=True)


def _validate_provenance(support_manifest_path: Path, histogram_config_path: Path) -> dict:
    support_manifest = json.loads(support_manifest_path.read_text(encoding="utf-8"))
    histogram_config = json.loads(histogram_config_path.read_text(encoding="utf-8"))
    if support_manifest.get("model", {}).get("acceptance") != "monotone_spline_xgb":
        raise ValueError("Support manifest must identify monotone-spline acceptance.")
    if support_manifest.get("model", {}).get("loss") != "xgb":
        raise ValueError("Support manifest must identify XGBoost loss.")
    if support_manifest.get("support", {}).get("baseline_subtracted") is not False:
        raise ValueError("Support width must retain absolute baseline risk.")
    if histogram_config.get("estimator") != "first_order":
        raise ValueError("Optimized prices must come from the saved first-order policy.")
    if histogram_config.get("range") != [-0.1, 0.2]:
        raise ValueError("Optimized-price configuration must span [-0.1, 0.2].")
    return {
        "support": support_manifest,
        "histogram": histogram_config,
    }


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    provenance: dict,
    output_path: Path,
) -> None:
    payload = {
        "analysis": "spline-support-cloud-with-optimized-price-overlay",
        "model": provenance["support"]["model"],
        "sample": provenance["support"]["sample"],
        "support": provenance["support"]["support"],
        "optimized_policy": {
            "estimator": provenance["histogram"]["estimator"],
            "policy_artifact": provenance["histogram"]["policy_artifact"],
            "population": "training customers restricted to [-0.1, 0.2]",
            "representation": "saved 0.01-width density histogram",
        },
        "display": {
            "blue": "monotone-spline/XGBoost mean profit and local-support cloud",
            "red": "saved first-order GLM-policy optimized price changes",
            "legend": ["Mean Profit", "Optimized Price Changes"],
            "shared_x_axis": "decimal price change on [-0.1, 0.2]",
        },
        "inputs": {
            "support_csv": {
                "path": str(args.support_csv.resolve()),
                "sha256": _sha256_file(args.support_csv),
            },
            "support_manifest": {
                "path": str(args.support_manifest.resolve()),
                "sha256": _sha256_file(args.support_manifest),
            },
            "histogram_csv": {
                "path": str(args.histogram_csv.resolve()),
                "sha256": _sha256_file(args.histogram_csv),
            },
            "histogram_config": {
                "path": str(args.histogram_config.resolve()),
                "sha256": _sha256_file(args.histogram_config),
            },
            "source_pdf": {
                "path": str(args.source_pdf.resolve()),
                "sha256": _sha256_file(args.source_pdf),
            },
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": _sha256_file(output_path),
        },
        "optimizer": "not rerun; exact saved first-order policy histogram replayed",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> list[Path]:
    support = _load_support(args.support_csv)
    histogram = _load_optimized_histogram(args.histogram_csv)
    provenance = _validate_provenance(args.support_manifest, args.histogram_config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    manifest_path = output_dir / MANIFEST_NAME
    _plot_overlay(support, histogram, output_path)
    _write_manifest(
        manifest_path,
        args=args,
        provenance=provenance,
        output_path=output_path,
    )
    outputs = [output_path, manifest_path]
    for output in outputs:
        print(output, flush=True)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
