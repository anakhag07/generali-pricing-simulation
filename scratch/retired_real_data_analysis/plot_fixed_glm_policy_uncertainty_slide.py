#!/usr/bin/env python3
"""Build the figure and two Beamer frames for fixed-GLM rescoring.

The saved original GLM optimizer policy is held fixed.  This script only
replays its full-population scores and overlays the saved synthetic-tail lower
bound; it does not fit a model or select a new optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT.parent / "results"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "optimizer_uncertainty"
DEFAULT_REGULAR_CSV = DEFAULT_OUTPUT_DIR / "glm_local_support_cloud.csv"
DEFAULT_TAIL_CSV = DEFAULT_OUTPUT_DIR / "glm_lower_bound_synthetic_tail.csv"
DEFAULT_COMPARISON_CSV = (
    RESULTS_ROOT
    / "glm-synthetic-tail-140-lower-bound-policy-20k"
    / "glm_policy_comparison.csv"
)
DEFAULT_BASE_GLM_SUMMARY = (
    RESULTS_ROOT
    / "glm-softmax-80-20-first-order"
    / "glm-softmax-80-20-first-order"
    / "summary-seed-8.json"
)
DEFAULT_HISTOGRAM_CSV = (
    RESULTS_ROOT
    / "policy-histogram-analysis"
    / "glm-softmax-80-20-first-order"
    / "decimal-range-minus0.1-to-0.2"
    / "histogram_bin_data_decimal.csv"
)

SUPPORT_COLOR = "C0"
OPTIMIZED_COLOR = "#86002d"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regular-csv", type=Path, default=DEFAULT_REGULAR_CSV)
    parser.add_argument("--tail-csv", type=Path, default=DEFAULT_TAIL_CSV)
    parser.add_argument("--histogram-csv", type=Path, default=DEFAULT_HISTOGRAM_CSV)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument(
        "--base-glm-summary", type=Path, default=DEFAULT_BASE_GLM_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scores(comparison_csv: Path, summary_path: Path) -> dict[str, float | int]:
    comparison = pd.read_csv(comparison_csv)
    full = comparison.loc[comparison["population"] == "full_eligible"].set_index(
        "policy"
    )
    historical = full.loc["historical_actions"]
    glm = full.loc["glm_reference_policy"]
    n_customers = int(glm["n_customers"])
    if n_customers != 715_023:
        raise ValueError(f"Expected 715,023 eligible customers, found {n_customers}.")
    if int(historical["n_customers"]) != n_customers:
        raise ValueError("Historical and GLM policy scores use different populations.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    optimizer = summary["estimators"]["first_order"]
    policy_artifact = summary["policy_artifacts"]["first_order"]
    if summary["config"].get("enabled_estimators") != ["first_order"]:
        raise ValueError("Expected the saved repository first-order optimizer result.")

    raw = float(glm["glm_mean_profit"])
    lower = float(glm["glm_support_adjusted_mean_profit"])
    historical_raw = float(historical["glm_mean_profit"])
    return {
        "n_customers": n_customers,
        "historical_raw_mean_profit": historical_raw,
        "fixed_glm_raw_mean_profit": raw,
        "fixed_glm_lower_bound_mean_profit": lower,
        "envelope_adjustment_per_customer": lower - raw,
        "fixed_glm_raw_uplift_vs_historical": (raw - historical_raw) * n_customers,
        "fixed_glm_lower_bound_vs_historical_raw": (
            lower - historical_raw
        )
        * n_customers,
        "envelope_adjustment_total": (lower - raw) * n_customers,
        "reported_training_objective": -float(optimizer["final_value"]),
        "policy_artifact": policy_artifact,
    }


def _load_original_glm_histogram(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["plot"] == "optimized") & (frame["series"] == "Optimized")
    ].sort_values("bin_left")
    if len(selected) != 30:
        raise ValueError("Expected the saved 30-bin original GLM policy histogram.")
    return selected.reset_index(drop=True)


def _plot(
    regular: pd.DataFrame,
    tail: pd.DataFrame,
    histogram: pd.DataFrame,
    output: Path,
) -> None:
    u = regular["u"].to_numpy(float)
    mean = regular["mean_profit"].to_numpy(float)
    synthetic_lower = tail["lower_profit"].to_numpy(float)

    fig, profit_ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    profit_ax.fill_between(
        u,
        synthetic_lower,
        mean,
        color=SUPPORT_COLOR,
        alpha=0.2,
    )
    mean_line = profit_ax.plot(
        u, mean, color=SUPPORT_COLOR, linewidth=2.0, label="Mean GLM profit"
    )[0]
    lower_line = profit_ax.plot(
        u,
        synthetic_lower,
        color="black",
        linewidth=2.5,
        label="Uncertainty Envelope Lower Bound",
        zorder=5,
    )[0]
    lower_line.set_path_effects(
        [
            path_effects.Stroke(linewidth=4.0, foreground="black"),
            path_effects.Normal(),
        ]
    )

    profit_ax.set_title("Optimized Price Changes", fontsize=16)
    profit_ax.set_xlabel("Price Change", fontsize=12)
    profit_ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12)
    profit_ax.set_xlim(-0.1, 0.2)
    profit_ax.set_ylim(95, 170)
    profit_ax.tick_params(labelsize=10)
    profit_ax.grid(True, alpha=0.25)

    density_ax = profit_ax.twinx()
    density_ax.bar(
        histogram["bin_left"],
        histogram["density"],
        width=histogram["bin_right"] - histogram["bin_left"],
        align="edge",
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.55,
    )
    density_ax.set_ylabel(
        "Optimized Price-Change Density",
        color=OPTIMIZED_COLOR,
        fontsize=12,
    )
    density_ax.tick_params(axis="y", labelcolor=OPTIMIZED_COLOR, labelsize=10)
    density_ax.set_zorder(1)
    profit_ax.set_zorder(2)
    profit_ax.patch.set_visible(False)
    profit_ax.legend(
        [mean_line, lower_line],
        ["Mean Profit", "Uncertainty Envelope Lower Bound"],
        loc="upper left",
        fontsize=9,
    )
    fig.canvas.draw()
    fig.savefig(output, format="pdf")
    plt.close(fig)


def _format_millions(value: float) -> str:
    return f"{value / 1_000_000:+.2f}"


def _write_frame(path: Path, scores: dict[str, float | int]) -> None:
    frame = rf"""% Requires \usepackage{{booktabs}} and \usepackage{{textcomp}}.
% Place these frames immediately before
% mean_profit_glm_support_cloud_lower_bound_maximization.
\begin{{frame}}{{Optimized Price Changes}}
  \begin{{figure}}
    \centering
    \includegraphics[width=0.88\linewidth]{{figures/optimizer_uncertainty/original_glm_policy_with_synthetic_tail_lower_bound.pdf}}
  \end{{figure}}
\end{{frame}}

\begin{{frame}}{{Profit Comparison for the Original GLM Policy}}
  \begin{{table}}
    \centering
    \small
    \caption{{Full-population comparison ($n=715{{,}}023$)}}
    \begin{{tabular}}{{lrr}}
      \toprule
      Scoring rule & Mean profit/customer & Vs. historical \\
      \midrule
      Historical pricing (raw GLM)
        & {scores['historical_raw_mean_profit']:.2f} & -- \\
      Original GLM policy (raw GLM)
        & {scores['fixed_glm_raw_mean_profit']:.2f}
        & \texteuro\,{_format_millions(float(scores['fixed_glm_raw_uplift_vs_historical']))}M \\
      Same policy under uncertainty envelope
        & {scores['fixed_glm_lower_bound_mean_profit']:.2f}
        & \texteuro\,{_format_millions(float(scores['fixed_glm_lower_bound_vs_historical_raw']))}M \\
      \bottomrule
    \end{{tabular}}
  \end{{table}}

  \begin{{center}}
    \footnotesize
    Previously reported original-GLM training score: {scores['reported_training_objective']:.2f}.
  \end{{center}}
\end{{frame}}
"""
    path.write_text(frame, encoding="utf-8")


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    regular = pd.read_csv(args.regular_csv).sort_values("u").reset_index(drop=True)
    tail = pd.read_csv(args.tail_csv).sort_values("u").reset_index(drop=True)
    if not np.allclose(regular["u"], tail["u"]):
        raise ValueError("Regular and synthetic-tail curves use different action grids.")
    histogram = _load_original_glm_histogram(args.histogram_csv)
    scores = _load_scores(args.comparison_csv, args.base_glm_summary)

    plot_path = output_dir / "original_glm_policy_with_synthetic_tail_lower_bound.pdf"
    frame_path = output_dir / "original_glm_policy_uncertainty_frame.tex"
    _plot(regular, tail, histogram, plot_path)
    _write_frame(frame_path, scores)

    manifest_path = output_dir / "original_glm_policy_uncertainty_manifest.json"
    inputs = {
        "regular_csv": args.regular_csv.resolve(),
        "tail_csv": args.tail_csv.resolve(),
        "histogram_csv": args.histogram_csv.resolve(),
        "comparison_csv": args.comparison_csv.resolve(),
        "base_glm_summary": args.base_glm_summary.resolve(),
    }
    outputs = [plot_path, frame_path]
    manifest = {
        "analysis": "fixed-original-glm-policy-rescored-under-synthetic-tail-lower-bound",
        "optimization": "none; exact saved repository-optimizer GLM policy replayed",
        "scores": scores,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            path.name: {"sha256": _sha256(path)} for path in outputs
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs.append(manifest_path)
    for output in outputs:
        print(output)
    return outputs


if __name__ == "__main__":
    run(_parser().parse_args())
