#!/usr/bin/env python3
"""Plot the two saved full-population spline policies and historical pricing.

The optimized distributions replay exact saved outputs from the repository
optimizer. Historical distributions use every non-missing observed price
change inside the displayed range. All densities use 0.01-wide decimal bins.
"""

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


BIN_WIDTH = 0.01
OPTIMIZED_COLOR = "#86002d"
HISTORICAL_COLOR = "#737373"
HISTORICAL_TEXT_COLOR = "#666666"
POLICIES = (
    ("0_to_0p16", 0.0, 0.16, "glm-vs-full-monotone-spline-xgb-policy-u-0-0p16-seed-8"),
    ("minus0p1_to_0p2", -0.1, 0.2, "glm-vs-full-monotone-spline-xgb-policy-seed-8"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=results_root(),
        help="Directory containing the two completed optimizer result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=results_root() / "full-population-spline-policy-histograms-seed-8",
    )
    return parser


def fixed_bins(action_low: float, action_high: float) -> np.ndarray:
    n_bins_float = (float(action_high) - float(action_low)) / BIN_WIDTH
    n_bins = int(round(n_bins_float))
    if n_bins < 1 or not np.isclose(n_bins_float, n_bins):
        raise ValueError("Range must span a positive whole number of 0.01 bins.")
    edges = np.linspace(action_low, action_high, n_bins + 1, dtype=float)
    edges[-1] = np.nextafter(action_high, np.inf)
    return edges


def bounded(values: np.ndarray, action_low: float, action_high: float) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    tolerance = 1e-12
    selected = array[
        (array >= action_low - tolerance) & (array <= action_high + tolerance)
    ]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("Bounded distribution must be nonempty and finite.")
    return np.clip(selected, action_low, action_high)


def load_saved_actions(
    result_dir: Path,
    *,
    expected_bounds: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any], Path]:
    policy_path = result_dir / "spline_policy_seed_8.npz"
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("models", {}).get("acceptance") != (
        "exact full-cache monotone-spline XGBoost"
    ):
        raise ValueError(f"Unexpected acceptance model in {summary_path}.")
    if summary.get("models", {}).get("loss") != "XGBoost financial loss":
        raise ValueError(f"Unexpected loss model in {summary_path}.")
    if not np.allclose(summary["comparison_contract"]["action_bounds"], expected_bounds):
        raise ValueError(f"Unexpected summary action bounds in {summary_path}.")

    with np.load(policy_path, allow_pickle=False) as payload:
        bounds = np.asarray(payload["action_bounds"], dtype=float)
        actions = np.asarray(payload["all_actions"], dtype=float)
        selected_rows = np.asarray(payload["selected_row_indices"], dtype=np.int64)
    if not np.allclose(bounds, expected_bounds):
        raise ValueError(f"Unexpected saved policy bounds in {policy_path}.")
    if actions.shape != selected_rows.shape or actions.size != 715_023:
        raise ValueError("Saved actions must align to the full 715,023-customer cohort.")
    if not np.isfinite(actions).all():
        raise ValueError("Saved actions must be finite.")
    low, high = expected_bounds
    if float(np.min(actions)) < low - 1e-12 or float(np.max(actions)) > high + 1e-12:
        raise ValueError("Saved actions fall outside their policy bounds.")
    return actions, summary, policy_path


def histogram_records(
    *,
    plot: str,
    series: str,
    values: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, Any]]:
    counts, edges = np.histogram(values, bins=bins)
    density = counts / (values.size * np.diff(edges))
    return [
        {
            "plot": plot,
            "series": series,
            "bin_left": float(left),
            "bin_right": float(right),
            "count": int(count),
            "density": float(bin_density),
        }
        for left, right, count, bin_density in zip(
            edges[:-1], edges[1:], counts, density, strict=True
        )
    ]


def plot_optimized(
    values: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    range_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        values,
        bins=fixed_bins(action_low, action_high),
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.75,
    )
    ax.set_xlim(action_low, action_high)
    ax.set_title(f"Optimized Price Changes — Spline/XGBoost {range_label}", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_overlay(
    historical: np.ndarray,
    optimized: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    range_label: str,
    output_path: Path,
) -> None:
    bins = fixed_bins(action_low, action_high)
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
        optimized,
        bins=bins,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
        label="Optimized",
    )
    ax.set_xlim(action_low, action_high)
    ax.set_title(
        f"Historical and Optimized Price Changes — Spline/XGBoost {range_label}",
        fontsize=16,
    )
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


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    source_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_frame = pd.read_csv(
        dataset_csv_path(), sep=";", usecols=[OBSERVED_U_COL]
    )
    historical_all = historical_frame[OBSERVED_U_COL].dropna().to_numpy(dtype=float)
    if historical_all.size == 0 or not np.isfinite(historical_all).all():
        raise ValueError("Historical price changes must be nonempty and finite.")

    loaded: dict[str, tuple[np.ndarray, dict[str, Any], Path]] = {}
    records: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    provenance: dict[str, Any] = {}
    for slug, low, high, directory_name in POLICIES:
        result_dir = source_root / directory_name
        actions, run_summary, policy_path = load_saved_actions(
            result_dir, expected_bounds=(low, high)
        )
        loaded[slug] = (actions, run_summary, policy_path)
        range_label = f"[{low:g}, {high:g}]"
        plot_name = f"optimized_spline_{slug}"
        output_path = output_dir / f"{plot_name}.pdf"
        plot_optimized(
            actions,
            action_low=low,
            action_high=high,
            range_label=range_label,
            output_path=output_path,
        )
        records.extend(
            histogram_records(
                plot=plot_name,
                series="Optimized",
                values=actions,
                bins=fixed_bins(low, high),
            )
        )
        output_paths.append(output_path)
        provenance[slug] = {
            "optimizer_entry_point": run_summary["optimizer"]["entry_point"],
            "optimizer_step_rule": run_summary["optimizer"]["step_rule"],
            "action_bounds": [low, high],
            "all_customers": int(actions.size),
            "policy_output": _file_record(policy_path),
            "run_summary": _file_record(result_dir / "summary.json"),
        }

    historical_range_counts: dict[str, int] = {}
    for slug, low, high, _ in POLICIES:
        actions = loaded[slug][0]
        historical_range = bounded(historical_all, low, high)
        historical_range_counts[slug] = int(historical_range.size)
        overlay_name = f"historical_vs_optimized_spline_{slug}"
        overlay_path = output_dir / f"{overlay_name}.pdf"
        plot_overlay(
            historical_range,
            actions,
            action_low=low,
            action_high=high,
            range_label=f"[{low:g}, {high:g}]",
            output_path=overlay_path,
        )
        overlay_bins = fixed_bins(low, high)
        records.extend(
            histogram_records(
                plot=overlay_name,
                series="Historical",
                values=historical_range,
                bins=overlay_bins,
            )
        )
        records.extend(
            histogram_records(
                plot=overlay_name,
                series="Optimized",
                values=actions,
                bins=overlay_bins,
            )
        )
        output_paths.append(overlay_path)

    histogram_path = output_dir / "histogram_bin_data.csv"
    pd.DataFrame(records).to_csv(histogram_path, index=False)
    output_paths.append(histogram_path)
    summary = {
        "analysis": "full-population-spline-policy-histograms",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "optimized_population": "all 715,023 saved policy customers",
        "historical_population": {
            "source": _file_record(dataset_csv_path()),
            "nonmissing_rows": int(historical_all.size),
            "rows_by_display_range": historical_range_counts,
            "range_semantics": (
                "Historical rows outside each displayed range are excluded before "
                "density normalization."
            ),
        },
        "bin_width": BIN_WIDTH,
        "colors": {
            "optimized": OPTIMIZED_COLOR,
            "historical": HISTORICAL_COLOR,
        },
        "optimizer_output_provenance": provenance,
        "outputs": {path.name: _file_record(path) for path in output_paths},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for path in [*output_paths, summary_path]:
        print(path, flush=True)


if __name__ == "__main__":
    main()
