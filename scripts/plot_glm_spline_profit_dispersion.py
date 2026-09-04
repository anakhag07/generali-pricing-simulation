"""Plot GLM and monotone-spline customer profit dispersion on a fixed action grid.

The two canonical model pairs are GLM acceptance with GLM financial loss, and
exact monotone-spline XGBoost acceptance with XGBoost financial loss. Curves
show predicted profit directly, so larger values remain better. This script
does not optimize, scan for, or mark an optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import (
    ACCEPTANCE_MODEL_ARTIFACTS,
    DATASET_PATH,
    LOSS_MODEL_ARTIFACTS,
    PREMIUM_COL,
)
from data.loader import (
    ModelArtifactBundle,
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_observed_u_array,
    load_x_frame,
)
from reporting.profit_dispersion import (
    ANCHOR_U,
    SPLINE_DENSE_GRID_SIZE,
    U_GRID,
    ProfitDispersion,
    customer_profit_matrix,
    exact_spline_acceptance_matrix,
    load_deterministic_sample_rows,
    predict_acceptance_matrix,
    predict_loss,
    row_index_sha256,
    summarize_profit,
)


DEFAULT_RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_DIAGNOSTICS = (
    DEFAULT_RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "glm-spline-profit-dispersion"
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLE_SEED = 20260831
MODEL_ORDER = ("glm", "spline")
MODEL_LABELS = {
    "glm": "GLM acceptance + GLM risk",
    "spline": "Spline acceptance + XGBoost risk",
}
MODEL_ARTIFACT_IDS = {
    "glm": {"acceptance": "linear", "risk": "linear"},
    "spline": {"acceptance": "monotone_spline_xgb", "risk": "xgb"},
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
        help="Saved diagnostics NPZ containing the deterministic row_indices sample.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spline_weights(eligible_rows: np.ndarray) -> np.ndarray:
    observed_u = load_observed_u_array("xgb", row_indices=eligible_rows)
    in_support = observed_u[(observed_u >= ANCHOR_U[0]) & (observed_u <= ANCHOR_U[-1])]
    frequencies = pd.Series(in_support).round(2).value_counts(normalize=True)
    weights = frequencies.reindex(ANCHOR_U.round(2), fill_value=0.0).to_numpy(float)
    if not np.isfinite(weights).all() or not np.any(weights > 0.0):
        raise ValueError("Could not construct finite spline-anchor weights.")
    return weights


def _load_canonical_model_pairs(
) -> tuple[
    tuple[ModelArtifactBundle, ModelArtifactBundle],
    tuple[ModelArtifactBundle, ModelArtifactBundle],
]:
    glm_pair = load_model_artifact_pair("linear", "linear")
    spline_source_pair = load_model_artifact_pair("xgb", "xgb")
    return glm_pair, spline_source_pair


def _set_model_jobs(artifact: ModelArtifactBundle, n_jobs: int) -> None:
    if hasattr(artifact.model, "set_params"):
        available = artifact.model.get_params(deep=False)
        if "n_jobs" in available:
            artifact.model.set_params(n_jobs=int(n_jobs))


def _curve_rows(
    summaries: Mapping[str, ProfitDispersion],
    *,
    n_customers: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    statistic_pairs = (
        ("mean", "std", "std"),
        ("median", "mad", "mad"),
    )
    for model_family in MODEL_ORDER:
        summary = summaries[model_family]
        for center_name, dispersion_attr, dispersion_name in statistic_pairs:
            center = np.asarray(getattr(summary, center_name), dtype=float)
            dispersion = np.asarray(getattr(summary, dispersion_attr), dtype=float)
            if center.shape != U_GRID.shape or dispersion.shape != U_GRID.shape:
                raise ValueError(
                    f"{model_family} {center_name}/{dispersion_name} must contain "
                    f"{U_GRID.size} grid points."
                )
            for index, proposed_u in enumerate(U_GRID):
                rows.append(
                    {
                        "model_family": model_family,
                        "acceptance_model": MODEL_ARTIFACT_IDS[model_family]["acceptance"],
                        "risk_model": MODEL_ARTIFACT_IDS[model_family]["risk"],
                        "u": float(proposed_u),
                        "center_statistic": center_name,
                        "center": float(center[index]),
                        "dispersion_statistic": dispersion_name,
                        "dispersion": float(dispersion[index]),
                        "lower": float(center[index] - dispersion[index]),
                        "upper": float(center[index] + dispersion[index]),
                        "n_customers": int(n_customers),
                    }
                )
    return rows


def _plot_comparison(
    summaries: Mapping[str, ProfitDispersion],
    *,
    center_attr: str,
    dispersion_attr: str,
    title: str,
    band_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    x_percent = 100.0 * U_GRID
    for model_family in MODEL_ORDER:
        summary = summaries[model_family]
        center = np.asarray(getattr(summary, center_attr), dtype=float)
        dispersion = np.asarray(getattr(summary, dispersion_attr), dtype=float)
        line = ax.plot(x_percent, center, label=MODEL_LABELS[model_family])[0]
        ax.fill_between(
            x_percent,
            center - dispersion,
            center + dispersion,
            color=line.get_color(),
            alpha=0.2,
            label=f"{MODEL_LABELS[model_family]} ± {band_label}",
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Proposed price change (%)", fontsize=12)
    ax.set_ylabel("Predicted profit per customer", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _write_manifest(
    output_path: Path,
    *,
    args: argparse.Namespace,
    eligible_rows: np.ndarray,
    sample_rows: np.ndarray,
    spline_weights: np.ndarray,
    generated_files: Sequence[Path],
) -> None:
    artifacts = {
        "dataset": DATASET_PATH,
        "glm_acceptance": ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"],
        "glm_risk": LOSS_MODEL_ARTIFACTS["linear"]["path"],
        "xgb_acceptance_spline_source": ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"],
        "spline_risk": LOSS_MODEL_ARTIFACTS["xgb"]["path"],
    }
    manifest = {
        "analysis": "glm-spline-profit-dispersion",
        "sample": {
            "diagnostics_path": str(args.diagnostics.resolve()),
            "seed": int(args.seed),
            "n_customers": int(sample_rows.size),
            "eligible_population_size": int(eligible_rows.size),
            "row_indices_sha256": row_index_sha256(sample_rows),
            "selection": "sorted choice without replacement from eligible rows",
        },
        "action_grid": {
            "u_min": float(U_GRID[0]),
            "u_max": float(U_GRID[-1]),
            "u_step": float(U_GRID[1] - U_GRID[0]),
            "n_points": int(U_GRID.size),
        },
        "model_pairs": MODEL_ARTIFACT_IDS,
        "artifacts": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": _sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "spline": {
            "anchor_u": ANCHOR_U.tolist(),
            "anchor_weights": spline_weights.tolist(),
            "anchor_weight_population": "all eligible canonical rows with U in [0, 0.16]",
            "dense_grid_size": SPLINE_DENSE_GRID_SIZE,
            "recipe": [
                "XGBoost acceptance at anchors",
                "weighted smoothing spline in churn space",
                "probability clipping",
                "increasing isotonic regression",
                "PCHIP interpolation",
                "convert churn to acceptance",
            ],
            "raw_xgboost_fallback": False,
        },
        "profit": {
            "formula": "acceptance_i(u) * (premium_i * (1 + u) - predicted_loss_i)",
            "orientation": "maximization; higher profit is better; no sign inversion",
        },
        "statistics": {
            "mean_std": "mean +/- population standard deviation across customers (ddof=0)",
            "median_mad": "median +/- raw median absolute deviation across customers",
            "mad_scale_factor": 1.0,
            "smoothing": "none",
            "band_clipping": "none",
        },
        "generated_files": {
            path.name: {"sha256": _sha256_file(path)} for path in generated_files
        },
        "optimizer": "not used; no optimum is computed or marked",
    }
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> list[Path]:
    if int(args.sample_size) <= 0:
        raise ValueError("sample_size must be positive.")
    if int(args.n_jobs) == 0:
        raise ValueError("n_jobs cannot be zero.")

    eligible_rows = eligible_csv_row_indices("xgb")
    sample_rows = load_deterministic_sample_rows(
        args.diagnostics,
        eligible_rows,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )
    if sample_rows.size != DEFAULT_SAMPLE_SIZE:
        print(
            f"Warning: running with {sample_rows.size:,} customers instead of the production default.",
            flush=True,
        )
    frame = load_x_frame("xgb", row_indices=sample_rows)
    weights = _spline_weights(eligible_rows)
    (glm_acceptance, glm_risk), (xgb_acceptance, xgb_risk) = (
        _load_canonical_model_pairs()
    )
    _set_model_jobs(xgb_acceptance, int(args.n_jobs))
    _set_model_jobs(xgb_risk, int(args.n_jobs))

    print("Evaluating GLM acceptance and risk...", flush=True)
    glm_probability = predict_acceptance_matrix(glm_acceptance, frame, U_GRID)
    glm_loss = predict_loss(glm_risk, frame)
    glm_profit = customer_profit_matrix(
        glm_probability,
        frame[PREMIUM_COL].to_numpy(dtype=float),
        glm_loss,
        U_GRID,
    )
    glm_summary = summarize_profit(glm_profit)
    del glm_probability, glm_loss, glm_profit

    print("Fitting exact per-customer monotone splines without fallback...", flush=True)
    spline_probability = exact_spline_acceptance_matrix(
        xgb_acceptance,
        frame,
        U_GRID,
        weights,
        n_jobs=int(args.n_jobs),
    )
    spline_loss = predict_loss(xgb_risk, frame)
    spline_profit = customer_profit_matrix(
        spline_probability,
        frame[PREMIUM_COL].to_numpy(dtype=float),
        spline_loss,
        U_GRID,
    )
    spline_summary = summarize_profit(spline_profit)
    del spline_probability, spline_loss, spline_profit

    summaries = {"glm": glm_summary, "spline": spline_summary}
    rows = _curve_rows(summaries, n_customers=sample_rows.size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "profit_dispersion_curves.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mean_pdf = output_dir / "profit_mean_std_glm_vs_spline.pdf"
    median_pdf = output_dir / "profit_median_mad_glm_vs_spline.pdf"
    _plot_comparison(
        summaries,
        center_attr="mean",
        dispersion_attr="std",
        title="Mean Predicted Profit with Customer Standard Deviation",
        band_label="1 SD",
        output_path=mean_pdf,
    )
    _plot_comparison(
        summaries,
        center_attr="median",
        dispersion_attr="mad",
        title="Median Predicted Profit with Customer MAD",
        band_label="1 MAD",
        output_path=median_pdf,
    )
    generated = [csv_path, mean_pdf, median_pdf]
    manifest_path = output_dir / "run_manifest.json"
    _write_manifest(
        manifest_path,
        args=args,
        eligible_rows=eligible_rows,
        sample_rows=sample_rows,
        spline_weights=weights,
        generated_files=generated,
    )
    generated.append(manifest_path)
    for path in generated:
        print(path.resolve(), flush=True)
    return generated


def main(argv: Sequence[str] | None = None) -> None:
    run_analysis(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
