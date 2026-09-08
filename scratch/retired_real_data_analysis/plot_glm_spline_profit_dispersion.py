"""Plot GLM and monotone-spline ModelBasedObjective profit dispersion.

The repository objective first computes the minimization cost
``acceptance * (loss - revenue)``. Curves negate that per-customer cost for the
profit/maximization display where larger values remain better. This script does
not optimize, scan for, or mark an optimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import (
    ACCEPTANCE_MODEL_ARTIFACTS,
    ACCEPTANCE_STATE_COLS,
    DATASET_PATH,
    LOSS_FEATURE_COLS,
    LOSS_MODEL_ARTIFACTS,
    PREMIUM_COL,
)
from data.loader import (
    ModelArtifactBundle,
    eligible_csv_row_indices,
    load_model_artifact_pair,
    load_x_frame,
)
from objective.objectives.generali.model_based import ModelBasedObjective
from objective.policy import ConstantPolicy
from experiments.provenance import file_sha256 as _sha256_file
from reporting.profit_dispersion import (
    ANCHOR_U,
    SPLINE_DENSE_GRID_SIZE,
    ProfitDispersion,
    exact_spline_acceptance_matrix,
    load_deterministic_sample_rows,
    model_based_objective_matrix,
    row_index_sha256,
    spline_anchor_weights as _spline_weights,
    summarize_profit,
)


DEFAULT_RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_DIAGNOSTICS = (
    DEFAULT_RESULTS_ROOT / "customer-coverage-envelope-slides" / "coverage_diagnostics.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "glm-spline-objective-dispersion-minus010-plus020"
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_SAMPLE_SEED = 20260831
DEFAULT_U_MIN = -0.10
DEFAULT_U_MAX = 0.20
DEFAULT_U_COUNT = 301
MODEL_ORDER = ("glm", "spline")
MODEL_LABELS = {
    "glm": "GLM",
    "spline": "Monotone-spline XGBoost",
}
MODEL_ARTIFACT_IDS = {
    "glm": {"acceptance": "linear", "loss": "linear"},
    "spline": {"acceptance": "monotone_spline_xgb", "loss": "xgb"},
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
    parser.add_argument("--u-min", type=float, default=DEFAULT_U_MIN)
    parser.add_argument("--u-max", type=float, default=DEFAULT_U_MAX)
    parser.add_argument("--u-count", type=int, default=DEFAULT_U_COUNT)
    return parser


def _resolve_u_grid(u_min: float, u_max: float, u_count: int) -> np.ndarray:
    if not np.isfinite([u_min, u_max]).all():
        raise ValueError("u_min and u_max must be finite.")
    if float(u_min) >= float(u_max):
        raise ValueError("u_min must be less than u_max.")
    if int(u_count) < 2:
        raise ValueError("u_count must be at least two.")
    return np.linspace(float(u_min), float(u_max), int(u_count))


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
    u_grid: np.ndarray,
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
            if center.shape != u_grid.shape or dispersion.shape != u_grid.shape:
                raise ValueError(
                    f"{model_family} {center_name}/{dispersion_name} must contain "
                    f"{u_grid.size} grid points."
                )
            for index, proposed_u in enumerate(u_grid):
                rows.append(
                    {
                        "model_family": model_family,
                        "acceptance_model": MODEL_ARTIFACT_IDS[model_family]["acceptance"],
                        "loss_model": MODEL_ARTIFACT_IDS[model_family]["loss"],
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
    u_grid: np.ndarray,
    *,
    center_attr: str,
    dispersion_attr: str,
    title: str,
    band_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(constrained_layout=True)
    x_percent = 100.0 * u_grid
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
    ax.set_ylabel("Predicted profit per customer (−objective)", fontsize=12)
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
    u_grid: np.ndarray,
    generated_files: Sequence[Path],
) -> None:
    artifacts = {
        "dataset": DATASET_PATH,
        "glm_acceptance": ACCEPTANCE_MODEL_ARTIFACTS["linear"]["path"],
        "glm_loss": LOSS_MODEL_ARTIFACTS["linear"]["path"],
        "xgb_acceptance_spline_source": ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"],
        "spline_loss": LOSS_MODEL_ARTIFACTS["xgb"]["path"],
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
            "u_min": float(u_grid[0]),
            "u_max": float(u_grid[-1]),
            "u_step": float(u_grid[1] - u_grid[0]),
            "n_points": int(u_grid.size),
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
            "boundary_behavior": {
                "below_0": "constant churn at the fitted lower boundary",
                "above_0.16": "linear churn using the fitted upper slope, clipped to [0, 1]",
            },
            "raw_xgboost_fallback": False,
        },
        "objective": {
            "class": "ModelBasedObjective",
            "minimization_formula": (
                "acceptance_i(u) * (predicted_loss_i - premium_i * (1 + u))"
            ),
            "plot_transform": "profit_i(u) = -objective_i(u)",
            "plot_orientation": "maximization; higher displayed profit is better",
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
    u_grid = _resolve_u_grid(args.u_min, args.u_max, args.u_count)

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
    (glm_acceptance, glm_loss), (xgb_acceptance, xgb_loss) = (
        _load_canonical_model_pairs()
    )
    _set_model_jobs(xgb_acceptance, int(args.n_jobs))
    _set_model_jobs(xgb_loss, int(args.n_jobs))

    glm_objective = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=glm_acceptance,
        loss_model=glm_loss,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
    )
    spline_objective = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=xgb_acceptance,
        loss_model=xgb_loss,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
    )

    print("Evaluating the GLM ModelBasedObjective minimization grid...", flush=True)
    glm_cost = model_based_objective_matrix(
        glm_objective,
        frame,
        u_grid,
    )
    glm_summary = summarize_profit(-glm_cost)
    del glm_cost

    print("Fitting exact per-customer monotone splines without fallback...", flush=True)
    spline_probability = exact_spline_acceptance_matrix(
        xgb_acceptance,
        frame,
        u_grid,
        weights,
        n_jobs=int(args.n_jobs),
    )
    print("Evaluating the spline ModelBasedObjective minimization grid...", flush=True)
    spline_cost = model_based_objective_matrix(
        spline_objective,
        frame,
        u_grid,
        acceptance=spline_probability,
    )
    spline_summary = summarize_profit(-spline_cost)
    del spline_probability, spline_cost

    summaries = {"glm": glm_summary, "spline": spline_summary}
    rows = _curve_rows(summaries, u_grid, n_customers=sample_rows.size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "profit_dispersion_curves.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    mean_pdf = output_dir / "profit_mean_std_glm_vs_spline.pdf"
    median_pdf = output_dir / "profit_median_mad_glm_vs_spline.pdf"
    _plot_comparison(
        summaries,
        u_grid,
        center_attr="mean",
        dispersion_attr="std",
        title="Mean Predicted Profit with Customer Standard Deviation",
        band_label="1 SD",
        output_path=mean_pdf,
    )
    _plot_comparison(
        summaries,
        u_grid,
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
        u_grid=u_grid,
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
