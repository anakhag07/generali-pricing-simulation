"""Plot the exact monotone-spline/XGBoost profit curve with local-support cloud.

This analysis reuses the deterministic 20,000-customer sample and exact-spline
``-ModelBasedObjective`` curve. It recomputes the customer-specific local
support over the requested wide action range, but does not refit models or
select an optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.coverage import (
    DEFAULT_ACTION_BANDWIDTH as ACTION_BANDWIDTH,
    local_support_matrix as _local_support_matrix,
    mixed_customer_embedding as _mixed_customer_embedding,
)
from data.loader import load_model_artifacts, load_x_frame
from reporting.profit_dispersion import row_index_sha256


DEFAULT_RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_CURVE_DIR = (
    DEFAULT_RESULTS_ROOT / "glm-spline-objective-dispersion-minus010-plus020"
)
DEFAULT_DIAGNOSTICS = (
    DEFAULT_RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "monotone-spline-xgb-support-cloud"
GAUSSIAN_SMOOTH_SIGMA = 1.25
GAUSSIAN_SMOOTH_TRUNCATE = 4.0
SUPPORT_BAND_MAX_HALF_WIDTH = 10.0
EXPECTED_SAMPLE_SIZE = 20_000
SUPPORT_GRID = np.linspace(-0.10, 0.20, 301)
PLOT_TITLE = "Mean Predicted Profit Per Customer vs. Price Change"
X_AXIS_LABEL = "Price Change"
Y_AXIS_LABEL = "Mean Predicted Profit Per Customer"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=DEFAULT_CURVE_DIR / "profit_dispersion_curves.csv",
    )
    parser.add_argument(
        "--curve-manifest",
        type=Path,
        default=DEFAULT_CURVE_DIR / "run_manifest.json",
    )
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smooth(values: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(
        np.asarray(values, dtype=float),
        sigma=GAUSSIAN_SMOOTH_SIGMA,
        mode="nearest",
        truncate=GAUSSIAN_SMOOTH_TRUNCATE,
    )


def _absolute_support_half_width(
    median_support: np.ndarray,
    *,
    max_half_width: float = SUPPORT_BAND_MAX_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map absolute inverse-root support risk to a positive display width."""
    support = np.asarray(median_support, dtype=float)
    if support.ndim != 1 or support.size == 0 or not np.isfinite(support).all():
        raise ValueError("median_support must be a non-empty finite 1D array.")
    if np.any(support <= 0.0):
        raise ValueError("median_support must be positive everywhere.")
    if max_half_width <= 0.0:
        raise ValueError("max_half_width must be positive.")

    relative_support = support / float(np.max(support))
    absolute_risk = np.sqrt(1.0 / relative_support)
    half_width = max_half_width * absolute_risk / float(np.max(absolute_risk))
    return relative_support, absolute_risk, half_width


def _compute_median_local_support(
    row_indices: np.ndarray,
    observed_u: np.ndarray,
    *,
    n_neighbors: int,
    n_jobs: int,
) -> np.ndarray:
    """Reapply the coverage-slide local-support scaffold on the wide grid."""
    frame = load_x_frame("xgb", row_indices=row_indices)
    acceptance_artifact, _ = load_model_artifacts("xgb")
    embedding = _mixed_customer_embedding(acceptance_artifact, frame)
    support = _local_support_matrix(
        embedding,
        observed_u,
        SUPPORT_GRID,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    return np.median(support, axis=0)


def _load_cloud_data(
    curve_csv: Path,
    curve_manifest: Path,
    diagnostics_path: Path,
    *,
    n_neighbors: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = json.loads(curve_manifest.read_text(encoding="utf-8"))
    if manifest.get("objective", {}).get("class") != "ModelBasedObjective":
        raise ValueError("Curve manifest must identify ModelBasedObjective.")
    if manifest.get("objective", {}).get("plot_transform") != "profit_i(u) = -objective_i(u)":
        raise ValueError("Curve manifest must identify the objective-to-profit sign flip.")
    model_pair = manifest.get("model_pairs", {}).get("spline")
    if model_pair != {"acceptance": "monotone_spline_xgb", "loss": "xgb"}:
        raise ValueError("Curve manifest does not contain the canonical spline/XGBoost pair.")

    with np.load(diagnostics_path, allow_pickle=False) as diagnostics:
        required = {"row_indices", "observed_u"}
        missing = sorted(required.difference(diagnostics.files))
        if missing:
            raise ValueError(f"Coverage diagnostics are missing keys: {missing}")
        row_indices = diagnostics["row_indices"].astype(int)
        observed_u = diagnostics["observed_u"].astype(float)

    if row_indices.size != EXPECTED_SAMPLE_SIZE or np.unique(row_indices).size != row_indices.size:
        raise ValueError("Expected 20,000 unique deterministic diagnostic rows.")
    if observed_u.shape != row_indices.shape or not np.isfinite(observed_u).all():
        raise ValueError("Expected one finite historical action per diagnostic row.")
    expected_hash = manifest.get("sample", {}).get("row_indices_sha256")
    actual_hash = row_index_sha256(row_indices)
    if expected_hash != actual_hash:
        raise ValueError("Curve and coverage diagnostics use different customer samples.")

    median_support = _compute_median_local_support(
        row_indices,
        observed_u,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    if median_support.shape != SUPPORT_GRID.shape or not np.isfinite(median_support).all():
        raise ValueError("Local support must contain one finite value per wide-grid point.")
    relative_support, absolute_risk, raw_width = _absolute_support_half_width(
        median_support
    )
    display_width = _smooth(raw_width)

    curves = pd.read_csv(curve_csv)
    selected = curves.loc[
        (curves["model_family"] == "spline")
        & (curves["center_statistic"] == "mean")
        & (curves["acceptance_model"] == "monotone_spline_xgb")
        & (curves["loss_model"] == "xgb")
    ].sort_values("u")
    if selected.empty or selected["u"].duplicated().any():
        raise ValueError("Expected one exact-spline mean-profit row per curve-grid point.")
    curve_u = selected["u"].to_numpy(dtype=float)
    matched_positions: list[int] = []
    for proposed_u in SUPPORT_GRID:
        matches = np.flatnonzero(np.isclose(curve_u, proposed_u, rtol=0.0, atol=1e-12))
        if matches.size != 1:
            raise ValueError("The exact-spline curve does not cover the support grid.")
        matched_positions.append(int(matches[0]))
    positions = np.asarray(matched_positions, dtype=int)
    mean_profit = selected["center"].to_numpy(dtype=float)[positions]
    if not np.isfinite(mean_profit).all():
        raise ValueError("Mean profit must be finite on the support grid.")
    smoothed_profit = _smooth(mean_profit)

    cloud = pd.DataFrame(
        {
            "u": SUPPORT_GRID,
            "mean_profit": mean_profit,
            "smoothed_mean_profit": smoothed_profit,
            "median_effective_neighbor_count": median_support,
            "relative_local_support": relative_support,
            "absolute_inverse_root_support_risk": absolute_risk,
            "raw_absolute_support_half_width": raw_width,
            "smoothed_support_half_width": display_width,
            "support_cloud_lower_profit": smoothed_profit - display_width,
            "support_cloud_upper_profit": smoothed_profit + display_width,
        }
    )
    provenance = {
        "sample_row_indices_sha256": actual_hash,
        "n_customers": int(row_indices.size),
        "curve_manifest": manifest,
    }
    return cloud, provenance


def _plot_support_cloud(frame: pd.DataFrame, output_path: Path) -> None:
    u = frame["u"].to_numpy(dtype=float)
    mean_profit = frame["smoothed_mean_profit"].to_numpy(dtype=float)
    lower = frame["support_cloud_lower_profit"].to_numpy(dtype=float)
    upper = frame["support_cloud_upper_profit"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(
        u,
        lower,
        upper,
        alpha=0.2,
        label="Illustrative local-support cloud",
    )
    ax.plot(
        u,
        mean_profit,
        linewidth=2.0,
        label="Monotone-spline XGBoost mean profit",
    )
    ax.set_title(PLOT_TITLE, fontsize=16)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=12)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(float(u[0]), float(u[-1]))
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    provenance: dict[str, object],
    csv_path: Path,
    pdf_path: Path,
) -> None:
    source_manifest = provenance["curve_manifest"]
    assert isinstance(source_manifest, dict)
    manifest = {
        "analysis": "monotone-spline-xgb-local-support-cloud",
        "sample": {
            "n_customers": provenance["n_customers"],
            "row_indices_sha256": provenance["sample_row_indices_sha256"],
            "seed": source_manifest["sample"]["seed"],
        },
        "model": {
            "objective": "ModelBasedObjective",
            "acceptance": "monotone_spline_xgb",
            "loss": "xgb",
            "minimization_formula": source_manifest["objective"]["minimization_formula"],
            "plot_transform": "profit_i(u) = -objective_i(u)",
        },
        "support": {
            "source": str(args.diagnostics.resolve()),
            "source_sha256": _sha256_file(args.diagnostics),
            "grid": "u=-0.100,...,0.200",
            "definition": (
                "median customer-specific local support over nearest neighbors; "
                "absolute risk = sqrt(max(median_support) / median_support); "
                "illustrative width = 10 * absolute_risk / max(absolute_risk)"
            ),
            "n_neighbors": int(args.n_neighbors),
            "action_kernel_bandwidth": ACTION_BANDWIDTH,
            "baseline_subtracted": False,
            "interpretation": "illustrative extrapolation-support proxy, not a confidence interval",
        },
        "display": {
            "smoothing": {
                "method": "gaussian_filter1d",
                "sigma": GAUSSIAN_SMOOTH_SIGMA,
                "truncate": GAUSSIAN_SMOOTH_TRUNCATE,
                "mode": "nearest",
            },
            "cloud": "smoothed mean profit +/- smoothed illustrative support width",
            "orientation": "maximization; higher displayed profit is better",
        },
        "inputs": {
            "curve_csv": {
                "path": str(args.curve_csv.resolve()),
                "sha256": _sha256_file(args.curve_csv),
            },
            "curve_manifest": {
                "path": str(args.curve_manifest.resolve()),
                "sha256": _sha256_file(args.curve_manifest),
            },
        },
        "outputs": {
            csv_path.name: {"sha256": _sha256_file(csv_path)},
            pdf_path.name: {"sha256": _sha256_file(pdf_path)},
        },
        "optimizer": "not used; no optimum is computed or marked",
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> list[Path]:
    cloud, provenance = _load_cloud_data(
        args.curve_csv,
        args.curve_manifest,
        args.diagnostics,
        n_neighbors=args.n_neighbors,
        n_jobs=args.n_jobs,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "monotone_spline_xgb_local_support_cloud.csv"
    pdf_path = output_dir / "monotone_spline_xgb_local_support_cloud.pdf"
    manifest_path = output_dir / "run_manifest.json"
    cloud.to_csv(csv_path, index=False)
    _plot_support_cloud(cloud, pdf_path)
    _write_manifest(
        manifest_path,
        args=args,
        provenance=provenance,
        csv_path=csv_path,
        pdf_path=pdf_path,
    )
    outputs = [pdf_path, csv_path, manifest_path]
    for output in outputs:
        print(output.resolve(), flush=True)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
