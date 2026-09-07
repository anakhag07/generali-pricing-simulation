#!/usr/bin/env python3
"""Export clean-title plots from exact saved repository-optimizer outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import OBSERVED_U_COL
from data.full_monotone_spline_cache import sha256_file
from data.loader import dataset_csv_path
from experiments.paths import results_root


ACTION_LOW = -0.1
ACTION_HIGH = 0.2
BIN_WIDTH = 0.01
OPTIMIZED_COLOR = "#86002d"
HISTORICAL_COLOR = "#737373"
HISTORICAL_TEXT_COLOR = "#666666"
FULL_POPULATION_SIZE = 715_023
RUN_500 = "glm-vs-full-monotone-spline-xgb-policy-500-steps-seed-8"
RUN_90 = "glm-vs-full-monotone-spline-xgb-policy-acceptance-0p90-seed-8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=results_root())
    return parser


def fixed_bins() -> np.ndarray:
    n_bins = int(round((ACTION_HIGH - ACTION_LOW) / BIN_WIDTH))
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, n_bins + 1, dtype=float)
    edges[-1] = np.nextafter(ACTION_HIGH, np.inf)
    return edges


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def load_saved_actions(
    result_dir: Path, *, expected_floor: float, expected_steps: int
) -> tuple[np.ndarray, Path, Path]:
    policy_path = result_dir / "spline_policy_seed_8.npz"
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not np.allclose(
        summary["comparison_contract"]["action_bounds"],
        [ACTION_LOW, ACTION_HIGH],
    ):
        raise ValueError(f"Unexpected action bounds in {summary_path}.")
    if not np.isclose(
        summary["comparison_contract"]["acceptance_floor"], expected_floor
    ):
        raise ValueError(f"Unexpected acceptance floor in {summary_path}.")
    if int(summary["optimizer"]["max_steps"]) != expected_steps:
        raise ValueError(f"Unexpected optimizer step limit in {summary_path}.")
    if summary["optimizer"]["entry_point"] != (
        "optimization.solvers.run_first_order_minimize"
    ):
        raise ValueError(f"Unexpected optimizer entry point in {summary_path}.")
    if summary["models"]["acceptance"] != (
        "exact full-cache monotone-spline XGBoost"
    ):
        raise ValueError(f"Unexpected acceptance model in {summary_path}.")
    if summary["models"]["loss"] != "XGBoost financial loss":
        raise ValueError(f"Unexpected loss model in {summary_path}.")

    with np.load(policy_path, allow_pickle=False) as payload:
        actions = np.asarray(payload["all_actions"], dtype=float)
        bounds = np.asarray(payload["action_bounds"], dtype=float)
    if actions.shape != (FULL_POPULATION_SIZE,):
        raise ValueError(f"Unexpected full-population shape in {policy_path}.")
    if not np.allclose(bounds, [ACTION_LOW, ACTION_HIGH]):
        raise ValueError(f"Unexpected saved policy bounds in {policy_path}.")
    if not np.isfinite(actions).all():
        raise ValueError(f"Non-finite optimized action in {policy_path}.")
    if float(actions.min()) < ACTION_LOW - 1e-12 or float(actions.max()) > ACTION_HIGH + 1e-12:
        raise ValueError(f"Optimized action outside policy bounds in {policy_path}.")
    return actions, policy_path, summary_path


def plot_optimized(actions: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        actions,
        bins=fixed_bins(),
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.75,
    )
    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_overlay(
    historical: np.ndarray, actions: np.ndarray, output_path: Path
) -> None:
    bins = fixed_bins()
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        historical,
        bins=bins,
        density=True,
        color=HISTORICAL_COLOR,
        edgecolor=HISTORICAL_COLOR,
        linewidth=0.5,
        alpha=0.55,
        label="Historical",
    )
    ax.hist(
        actions,
        bins=bins,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
        label="Optimized",
    )
    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    ax.set_title("Historical and Optimized Price Changes", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    legend = ax.legend(fontsize=10)
    legend_texts = legend.get_texts()
    if len(legend_texts) == 2:
        legend_texts[0].set_color(HISTORICAL_TEXT_COLOR)
        legend_texts[1].set_color(OPTIMIZED_COLOR)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    actions_500, policy_500, summary_500 = load_saved_actions(
        result_root / RUN_500,
        expected_floor=0.8787745289312372,
        expected_steps=500,
    )
    actions_90, policy_90, summary_90 = load_saved_actions(
        result_root / RUN_90,
        expected_floor=0.90,
        expected_steps=100,
    )

    historical_path = dataset_csv_path()
    historical_frame = pd.read_csv(
        historical_path, sep=";", usecols=[OBSERVED_U_COL]
    )
    historical_all = historical_frame[OBSERVED_U_COL].dropna().to_numpy(dtype=float)
    historical = historical_all[
        (historical_all >= ACTION_LOW) & (historical_all <= ACTION_HIGH)
    ]
    if historical.size == 0 or not np.isfinite(historical).all():
        raise ValueError("Historical in-range distribution must be nonempty and finite.")

    overlay_pdf = output_dir / "historical_and_optimized_price_changes.pdf"
    optimized_pdf = output_dir / "optimized_price_changes.pdf"
    optimized_90_pdf = output_dir / "constrained_optimizer_behavior_90_percent.pdf"
    plot_overlay(historical, actions_500, overlay_pdf)
    plot_optimized(actions_500, optimized_pdf, "Optimized Price Changes")
    plot_optimized(
        actions_90,
        optimized_90_pdf,
        "Constrained Optimizer Behavior — 90%",
    )

    manifest_path = output_dir / "provenance.json"
    manifest = {
        "analysis": "clean-title full-population spline policy plots",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action_bounds": [ACTION_LOW, ACTION_HIGH],
        "population": f"all {FULL_POPULATION_SIZE:,} saved policy customers",
        "models": {
            "acceptance": "exact full-cache monotone-spline XGBoost",
            "loss": "XGBoost financial loss",
        },
        "optimizer": {
            "entry_point": "optimization.solvers.run_first_order_minimize",
            "not_rerun_for_title_only_export": True,
            "provenance": "exact saved repository-optimizer outputs replayed",
        },
        "plots": {
            overlay_pdf.name: {
                "title": "Historical and Optimized Price Changes",
                "optimized_run": RUN_500,
                "acceptance_floor": 0.8787745289312372,
                "steps": 500,
            },
            optimized_pdf.name: {
                "title": "Optimized Price Changes",
                "optimized_run": RUN_500,
                "acceptance_floor": 0.8787745289312372,
                "steps": 500,
            },
            optimized_90_pdf.name: {
                "title": "Constrained Optimizer Behavior — 90%",
                "optimized_run": RUN_90,
                "acceptance_floor": 0.90,
                "steps": 100,
            },
        },
        "inputs": {
            "historical_data": _file_record(historical_path),
            "policy_500": _file_record(policy_500),
            "summary_500": _file_record(summary_500),
            "policy_90": _file_record(policy_90),
            "summary_90": _file_record(summary_90),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (overlay_pdf, optimized_pdf, optimized_90_pdf, manifest_path):
        print(path, flush=True)


if __name__ == "__main__":
    main()
