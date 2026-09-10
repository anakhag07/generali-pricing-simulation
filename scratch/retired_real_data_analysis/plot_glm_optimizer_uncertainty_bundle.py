#!/usr/bin/env python3
"""Recreate the five optimizer-uncertainty figures with the saved GLM curve.

No model or policy is fitted here. The figures replay the saved base-GLM and
20k GLM synthetic-tail optimizer actions with artifact provenance.
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
sys.path.insert(0, str(ROOT / "src"))

from data.loader import load_observed_u_array
from reporting.profit_dispersion import row_index_sha256


RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_CURVE_CSV = (
    RESULTS_ROOT
    / "glm-spline-objective-dispersion-minus010-plus020"
    / "profit_dispersion_curves.csv"
)
DEFAULT_CURVE_MANIFEST = DEFAULT_CURVE_CSV.with_name("run_manifest.json")
DEFAULT_SUPPORT_CSV = (
    RESULTS_ROOT
    / "monotone-spline-xgb-support-cloud"
    / "monotone_spline_xgb_local_support_cloud.csv"
)
DEFAULT_SUPPORT_MANIFEST = DEFAULT_SUPPORT_CSV.with_name("run_manifest.json")
DEFAULT_TAIL_CSV = (
    RESULTS_ROOT
    / "spline-xgb-synthetic-tail-140-support"
    / "synthetic_tail_support_cloud.csv"
)
DEFAULT_TAIL_MANIFEST = DEFAULT_TAIL_CSV.with_name("run_manifest.json")
DEFAULT_GLM_TAIL_DIR = RESULTS_ROOT / "glm-synthetic-tail-140-lower-bound-policy-20k"
DEFAULT_GLM_TAIL_ACTIONS = (
    DEFAULT_GLM_TAIL_DIR / "support_lower_bound_policy_full_actions.npz"
)
DEFAULT_GLM_TAIL_SUMMARY = DEFAULT_GLM_TAIL_DIR / "summary.json"
DEFAULT_BASE_GLM_HISTOGRAM = (
    RESULTS_ROOT
    / "policy-histogram-analysis"
    / "glm-softmax-80-20-first-order"
    / "decimal-range-minus0.1-to-0.2"
    / "histogram_bin_data_decimal.csv"
)
DEFAULT_BASE_GLM_HISTOGRAM_CONFIG = (
    DEFAULT_BASE_GLM_HISTOGRAM.parent / "analysis_config.json"
)
DEFAULT_BASE_GLM_SUMMARY = (
    RESULTS_ROOT
    / "glm-softmax-80-20-first-order"
    / "glm-softmax-80-20-first-order"
    / "summary-seed-8.json"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "optimizer_uncertainty"

ACTION_GRID = np.linspace(-0.1, 0.2, 301)
HISTOGRAM_EDGES = np.linspace(-0.1, 0.2, 31)
HISTORICAL_COLOR = "#737373"
OPTIMIZED_COLOR = "#86002d"
SUPPORT_COLOR = "C0"
SMOOTH_SIGMA = 1.25
SMOOTH_TRUNCATE = 4.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-csv", type=Path, default=DEFAULT_CURVE_CSV)
    parser.add_argument("--curve-manifest", type=Path, default=DEFAULT_CURVE_MANIFEST)
    parser.add_argument("--support-csv", type=Path, default=DEFAULT_SUPPORT_CSV)
    parser.add_argument("--support-manifest", type=Path, default=DEFAULT_SUPPORT_MANIFEST)
    parser.add_argument("--tail-csv", type=Path, default=DEFAULT_TAIL_CSV)
    parser.add_argument("--tail-manifest", type=Path, default=DEFAULT_TAIL_MANIFEST)
    parser.add_argument("--glm-tail-actions", type=Path, default=DEFAULT_GLM_TAIL_ACTIONS)
    parser.add_argument("--glm-tail-summary", type=Path, default=DEFAULT_GLM_TAIL_SUMMARY)
    parser.add_argument(
        "--base-glm-histogram", type=Path, default=DEFAULT_BASE_GLM_HISTOGRAM
    )
    parser.add_argument(
        "--base-glm-histogram-config",
        type=Path,
        default=DEFAULT_BASE_GLM_HISTOGRAM_CONFIG,
    )
    parser.add_argument("--base-glm-summary", type=Path, default=DEFAULT_BASE_GLM_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_clouds(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_manifest = json.loads(args.curve_manifest.read_text(encoding="utf-8"))
    support_manifest = json.loads(args.support_manifest.read_text(encoding="utf-8"))
    tail_manifest = json.loads(args.tail_manifest.read_text(encoding="utf-8"))
    expected_hash = curve_manifest.get("sample", {}).get("row_indices_sha256")
    if expected_hash != support_manifest.get("sample", {}).get("row_indices_sha256"):
        raise ValueError("GLM curve and support width do not use the same 20k sample.")
    if support_manifest.get("support", {}).get("baseline_subtracted") is not False:
        raise ValueError("Support width must retain its absolute baseline risk.")
    synthetic = tail_manifest.get("synthetic_modification", {})
    if synthetic.get("cutoff") != 0.12:
        raise ValueError("Expected the saved synthetic-tail cutoff at u=0.12.")

    curves = pd.read_csv(args.curve_csv)
    selected = curves.loc[
        (curves["model_family"] == "glm")
        & (curves["center_statistic"] == "mean")
        & (curves["acceptance_model"] == "linear")
        & (curves["loss_model"] == "linear")
    ].sort_values("u")
    if not np.allclose(selected["u"].to_numpy(float), ACTION_GRID):
        raise ValueError("GLM mean curve must cover the 301-point action grid.")
    mean = gaussian_filter1d(
        selected["center"].to_numpy(float),
        sigma=SMOOTH_SIGMA,
        mode="nearest",
        truncate=SMOOTH_TRUNCATE,
    )

    support = pd.read_csv(args.support_csv).sort_values("u").reset_index(drop=True)
    tail = pd.read_csv(args.tail_csv).sort_values("u").reset_index(drop=True)
    if not np.allclose(support["u"].to_numpy(float), ACTION_GRID):
        raise ValueError("Support cloud must cover the 301-point action grid.")
    if not np.allclose(tail["u"].to_numpy(float), ACTION_GRID):
        raise ValueError("Synthetic tail must cover the 301-point action grid.")
    width = support["smoothed_support_half_width"].to_numpy(float)
    tail_penalty = tail["synthetic_tail_penalty"].to_numpy(float)
    if np.any(tail_penalty[ACTION_GRID <= 0.12] != 0.0):
        raise ValueError("Synthetic tail must be zero through u=0.12.")

    regular = pd.DataFrame(
        {
            "u": ACTION_GRID,
            "mean_profit": mean,
            "support_half_width": width,
            "lower_profit": mean - width,
            "upper_profit": mean + width,
        }
    )
    tailed = regular.copy()
    tailed["synthetic_tail_penalty"] = tail_penalty
    tailed["optimization_support_penalty"] = width + tail_penalty
    tailed["lower_profit"] = mean - width - tail_penalty
    return regular, tailed


def _load_glm_tail_actions(
    actions_path: Path,
    summary_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("models", {}).get("mean_profit_family") != "glm":
        raise ValueError("Tail policy summary must identify the GLM mean-profit family.")
    if summary.get("optimizer", {}).get("entry_point") != (
        "optimization.solvers.run_first_order_minimize"
    ):
        raise ValueError("Tail actions must come from the repository optimizer.")
    if summary.get("optimizer", {}).get("success") is not True:
        raise ValueError("Saved GLM-tail optimizer did not converge successfully.")
    with np.load(actions_path, allow_pickle=False) as saved:
        actions = np.asarray(saved["actions"], dtype=float)
        rows = np.asarray(saved["row_indices"], dtype=int)
    expected = summary["full_application"]
    if actions.shape != rows.shape or actions.size != int(expected["n_customers"]):
        raise ValueError("GLM-tail actions are not aligned to their saved rows.")
    if row_index_sha256(rows) != expected["row_indices_sha256"]:
        raise ValueError("GLM-tail row checksum does not match its summary.")
    if np.any((actions < -0.1) | (actions > 0.2)):
        raise ValueError("GLM-tail actions must lie inside [-0.1, 0.2].")
    return actions, rows, summary


def _load_base_glm_histogram(
    histogram_path: Path,
    config_path: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config.get("estimator") != "first_order" or config.get("range") != [-0.1, 0.2]:
        raise ValueError("Base GLM histogram must replay the saved first-order policy.")
    final = summary.get("estimators", {}).get("first_order", {})
    if not np.isclose(float(final.get("final_value")), -145.7093681631783):
        raise ValueError("Unexpected base GLM optimizer summary.")
    frame = pd.read_csv(histogram_path)
    selected = frame.loc[
        (frame["plot"] == "optimized") & (frame["series"] == "Optimized")
    ].sort_values("bin_left")
    if len(selected) != 30:
        raise ValueError("Expected 30 saved base-GLM histogram bins.")
    return selected.reset_index(drop=True), config, summary


def _histogram_frame(values: np.ndarray, series: str) -> pd.DataFrame:
    counts, _ = np.histogram(values, bins=HISTOGRAM_EDGES)
    density, _ = np.histogram(values, bins=HISTOGRAM_EDGES, density=True)
    return pd.DataFrame(
        {
            "series": series,
            "bin_left": HISTOGRAM_EDGES[:-1],
            "bin_right": HISTOGRAM_EDGES[1:],
            "count": counts,
            "density": density,
        }
    )


def _plot_historical_overlay(
    historical: np.ndarray,
    optimized: np.ndarray,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    ax.hist(
        historical,
        bins=HISTOGRAM_EDGES,
        density=True,
        color=HISTORICAL_COLOR,
        edgecolor=HISTORICAL_COLOR,
        linewidth=0.5,
        alpha=0.55,
        label="Historical",
    )
    ax.hist(
        optimized,
        bins=HISTOGRAM_EDGES,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
        label="Optimized",
    )
    ax.set_xlim(-0.1, 0.2)
    ax.set_title("Historical and Optimized Price Changes", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def _plot_cloud(
    cloud: pd.DataFrame,
    output: Path,
    *,
    lower_only: bool,
    histogram: pd.DataFrame | None = None,
) -> None:
    u = cloud["u"].to_numpy(float)
    mean = cloud["mean_profit"].to_numpy(float)
    lower = cloud["lower_profit"].to_numpy(float)
    upper = cloud["upper_profit"].to_numpy(float)
    fig, profit_ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    cloud_artist = profit_ax.fill_between(
        u,
        lower,
        mean if lower_only else upper,
        color=SUPPORT_COLOR,
        alpha=0.2,
    )
    if lower_only:
        profit_ax.plot(u, lower, color=SUPPORT_COLOR, linewidth=1.0, alpha=0.8)
    mean_line = profit_ax.plot(u, mean, color=SUPPORT_COLOR, linewidth=2.0)[0]
    profit_ax.set_title(
        "Mean Predicted Profit Per Customer vs. Price Change", fontsize=16
    )
    profit_ax.set_xlabel("Price Change", fontsize=12)
    profit_ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    profit_ax.tick_params(labelsize=10)
    profit_ax.set_xlim(-0.1, 0.2)
    profit_ax.grid(True, alpha=0.25)

    if histogram is not None:
        profit_ax.set_ylabel(
            "Mean Predicted Profit Per Customer",
            color=SUPPORT_COLOR,
            fontsize=12,
        )
        profit_ax.tick_params(axis="y", labelcolor=SUPPORT_COLOR, labelsize=10)
        density_ax = profit_ax.twinx()
        bars = density_ax.bar(
            histogram["bin_left"],
            histogram["density"],
            width=histogram["bin_right"] - histogram["bin_left"],
            align="edge",
            color=OPTIMIZED_COLOR,
            edgecolor=OPTIMIZED_COLOR,
            linewidth=0.5,
            alpha=0.60,
        )
        density_ax.set_ylabel(
            "Optimized Price-Change Density",
            color=OPTIMIZED_COLOR,
            fontsize=12,
        )
        density_ax.tick_params(
            axis="y", labelcolor=OPTIMIZED_COLOR, labelsize=10
        )
        density_ax.set_zorder(1)
        profit_ax.set_zorder(2)
        profit_ax.patch.set_visible(False)
        profit_ax.legend(
            [(mean_line, cloud_artist), bars],
            ["Mean Profit", "Optimized Price Changes"],
            loc="upper left",
            fontsize=10,
        )
    fig.canvas.draw()
    fig.savefig(output, format="pdf")
    plt.close(fig)


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    regular, tailed = _load_clouds(args)
    tail_actions, tail_rows, tail_summary = _load_glm_tail_actions(
        args.glm_tail_actions, args.glm_tail_summary
    )
    base_histogram, base_config, base_summary = _load_base_glm_histogram(
        args.base_glm_histogram,
        args.base_glm_histogram_config,
        args.base_glm_summary,
    )
    historical = np.asarray(
        load_observed_u_array("xgb", row_indices=tail_rows), dtype=float
    )
    tail_histogram = _histogram_frame(tail_actions, "Optimized")

    outputs = {
        "historical_vs_optimized_glm_overlay.pdf": lambda path: (
            _plot_historical_overlay(historical, tail_actions, path)
        ),
        "mean_profit_glm_support_cloud_lower_bound_maximization.pdf": lambda path: (
            _plot_cloud(tailed, path, lower_only=True, histogram=tail_histogram)
        ),
        "mean_profit_glm_support_cloud_optimized_price_changes.pdf": lambda path: (
            _plot_cloud(regular, path, lower_only=False, histogram=base_histogram)
        ),
        "glm_xgb_local_support_cloud.pdf": lambda path: (
            _plot_cloud(regular, path, lower_only=False)
        ),
        "glm_xgb_lower_bound_synthetic_tail.pdf": lambda path: (
            _plot_cloud(tailed, path, lower_only=True)
        ),
    }
    output_paths: list[Path] = []
    for name, plotter in outputs.items():
        path = output_dir / name
        plotter(path)
        output_paths.append(path)

    regular_csv = output_dir / "glm_local_support_cloud.csv"
    tail_csv = output_dir / "glm_lower_bound_synthetic_tail.csv"
    histogram_csv = output_dir / "glm_optimizer_histograms.csv"
    regular.to_csv(regular_csv, index=False)
    tailed.to_csv(tail_csv, index=False)
    pd.concat(
        [
            _histogram_frame(historical, "Historical"),
            tail_histogram.assign(policy="glm_synthetic_tail"),
            base_histogram.assign(policy="base_glm"),
        ],
        ignore_index=True,
    ).to_csv(histogram_csv, index=False)
    output_paths.extend([regular_csv, tail_csv, histogram_csv])

    inputs = {
        "curve_csv": args.curve_csv,
        "curve_manifest": args.curve_manifest,
        "support_csv": args.support_csv,
        "support_manifest": args.support_manifest,
        "tail_csv": args.tail_csv,
        "tail_manifest": args.tail_manifest,
        "glm_tail_actions": args.glm_tail_actions,
        "glm_tail_summary": args.glm_tail_summary,
        "base_glm_histogram": args.base_glm_histogram,
        "base_glm_histogram_config": args.base_glm_histogram_config,
        "base_glm_summary": args.base_glm_summary,
    }
    manifest = {
        "analysis": "glm-optimizer-uncertainty-figure-bundle",
        "sample": tail_summary["sample"],
        "mean_profit": {
            "model_family": "glm",
            "acceptance_model": "linear",
            "loss_model": "linear",
            "curve": "20k arithmetic mean of per-customer GLM profit",
            "smoothing": {
                "method": "gaussian_filter1d",
                "sigma": SMOOTH_SIGMA,
                "truncate": SMOOTH_TRUNCATE,
                "mode": "nearest",
            },
        },
        "support": {
            "width_replayed_from": str(args.support_csv.resolve()),
            "synthetic_tail_replayed_from": str(args.tail_csv.resolve()),
            "cutoff": 0.12,
            "interpretation": "illustrative support proxy, not a confidence interval",
        },
        "optimized_actions": {
            "synthetic_tail": {
                "source": str(args.glm_tail_actions.resolve()),
                "optimizer": tail_summary["optimizer"],
            },
            "base_glm": {
                "source": str(args.base_glm_histogram.resolve()),
                "policy_artifact": base_config["policy_artifact"],
                "saved_final_value": base_summary["estimators"]["first_order"][
                    "final_value"
                ],
            },
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            path.name: {"sha256": _sha256(path)} for path in output_paths
        },
        "optimizer": "not rerun; exact saved repository-optimizer outputs replayed",
    }
    manifest_path = output_dir / "glm_optimizer_uncertainty_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths.append(manifest_path)
    for path in output_paths:
        print(path, flush=True)
    return output_paths


def main(argv: Sequence[str] | None = None) -> None:
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
