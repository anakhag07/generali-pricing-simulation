#!/usr/bin/env python3
"""Overlay the saved spline policy on the existing spline profit cloud.

The blue curve and band replay the existing support-cloud CSV. The red
histogram replays the exact full-population ``[-0.1, 0.2]`` spline policy
produced by the repository optimizer; no new optimum is selected here.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.full_monotone_spline_cache import sha256_file
from experiments.paths import results_root


ACTION_LOW = -0.1
ACTION_HIGH = 0.2
BIN_WIDTH = 0.01
OPTIMIZED_COLOR = "#86002d"


def _default_cloud_dir() -> Path:
    return results_root() / "monotone-spline-xgb-support-cloud"


def _default_policy_result() -> Path:
    return results_root() / "glm-vs-full-monotone-spline-xgb-policy-seed-8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud-dir", type=Path, default=_default_cloud_dir())
    parser.add_argument("--policy-result", type=Path, default=_default_policy_result())
    parser.add_argument("--output-dir", type=Path, default=_default_cloud_dir())
    return parser


def fixed_bins() -> np.ndarray:
    n_bins = int(round((ACTION_HIGH - ACTION_LOW) / BIN_WIDTH))
    edges = np.linspace(ACTION_LOW, ACTION_HIGH, n_bins + 1, dtype=float)
    edges[-1] = np.nextafter(ACTION_HIGH, np.inf)
    return edges


def histogram_frame(actions: np.ndarray) -> pd.DataFrame:
    values = np.asarray(actions, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Optimized actions must be nonempty and finite.")
    if float(np.min(values)) < ACTION_LOW - 1e-12 or float(np.max(values)) > ACTION_HIGH + 1e-12:
        raise ValueError("Optimized actions fall outside [-0.1, 0.2].")
    bins = fixed_bins()
    counts, _ = np.histogram(values, bins=bins)
    density = counts / (values.size * np.diff(bins))
    return pd.DataFrame(
        {
            "bin_left": bins[:-1],
            "bin_right": bins[1:],
            "count": counts,
            "density": density,
        }
    )


def _load_cloud(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "u",
        "smoothed_mean_profit",
        "support_cloud_lower_profit",
        "support_cloud_upper_profit",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Support-cloud CSV is missing columns: {sorted(required - set(frame.columns))}")
    values = frame.loc[:, sorted(required)].to_numpy(dtype=float)
    if len(frame) < 2 or not np.isfinite(values).all():
        raise ValueError("Support-cloud data must be nonempty and finite.")
    if not np.allclose([frame["u"].iloc[0], frame["u"].iloc[-1]], [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Support-cloud grid must span [-0.1, 0.2].")
    return frame


def _load_policy(policy_result: Path) -> tuple[np.ndarray, Path, Path, dict[str, Any]]:
    policy_path = policy_result / "spline_policy_seed_8.npz"
    summary_path = policy_result / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["models"]["acceptance"] != "exact full-cache monotone-spline XGBoost":
        raise ValueError("Expected exact full-cache monotone-spline XGBoost acceptance.")
    if summary["models"]["loss"] != "XGBoost financial loss":
        raise ValueError("Expected XGBoost financial loss.")
    if not np.allclose(summary["comparison_contract"]["action_bounds"], [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Expected policy bounds [-0.1, 0.2].")
    with np.load(policy_path, allow_pickle=False) as payload:
        actions = np.asarray(payload["all_actions"], dtype=float)
        bounds = np.asarray(payload["action_bounds"], dtype=float)
        rows = np.asarray(payload["selected_row_indices"], dtype=np.int64)
    if actions.shape != (715_023,) or rows.shape != actions.shape:
        raise ValueError("Expected saved actions for all 715,023 policy customers.")
    if not np.allclose(bounds, [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Saved policy arrays have unexpected action bounds.")
    histogram_frame(actions)
    return actions, policy_path, summary_path, summary


def plot_overlay(cloud: pd.DataFrame, actions: np.ndarray, output_path: Path) -> None:
    u = cloud["u"].to_numpy(dtype=float)
    mean_profit = cloud["smoothed_mean_profit"].to_numpy(dtype=float)
    lower = cloud["support_cloud_lower_profit"].to_numpy(dtype=float)
    upper = cloud["support_cloud_upper_profit"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    ax.fill_between(u, lower, upper, color="C0", alpha=0.2)
    mean_line = ax.plot(u, mean_profit, color="C0", linewidth=2.0, label="Mean Profit")[0]
    density_ax = ax.twinx()
    density_ax.hist(
        actions,
        bins=fixed_bins(),
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
    )

    ax.set_xlim(ACTION_LOW, ACTION_HIGH)
    ax.set_title("Mean Predicted Profit Per Customer vs. Price Change", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Mean Predicted Profit Per Customer", fontsize=12, color="C0")
    density_ax.set_ylabel("Optimized Price Change Density", fontsize=12, color=OPTIMIZED_COLOR)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10, colors="C0")
    density_ax.tick_params(axis="y", labelsize=10, colors=OPTIMIZED_COLOR)
    ax.grid(alpha=0.25)
    optimized_patch = Patch(
        facecolor=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        alpha=0.68,
        label="Optimized Price Changes",
    )
    legend = ax.legend(handles=[mean_line, optimized_patch], fontsize=10, loc="upper left")
    legend_texts = legend.get_texts()
    if len(legend_texts) == 2:
        legend_texts[0].set_color("C0")
        legend_texts[1].set_color(OPTIMIZED_COLOR)
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
    cloud_dir = args.cloud_dir.expanduser().resolve()
    policy_result = args.policy_result.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud_path = cloud_dir / "monotone_spline_xgb_local_support_cloud.csv"
    cloud_manifest_path = cloud_dir / "run_manifest.json"
    cloud = _load_cloud(cloud_path)
    actions, policy_path, optimizer_summary_path, optimizer_summary = _load_policy(policy_result)
    histogram = histogram_frame(actions)

    output_path = output_dir / "mean_profit_support_cloud_with_spline_optimized_price_changes.pdf"
    histogram_path = output_dir / "spline_optimized_price_change_histogram.csv"
    manifest_path = output_dir / "mean_profit_support_cloud_with_spline_optimized_price_changes_manifest.json"
    plot_overlay(cloud, actions, output_path)
    histogram.to_csv(histogram_path, index=False)
    manifest = {
        "analysis": "spline-support-cloud-with-spline-optimized-price-overlay",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "display": {
            "blue": "existing monotone-spline/XGBoost mean-profit and local-support cloud",
            "red": "saved full-population monotone-spline XGBoost optimized price changes",
            "shared_x_axis": "decimal price change on [-0.1, 0.2]",
        },
        "models": {
            "acceptance": "exact full-cache monotone-spline XGBoost",
            "loss": "XGBoost financial loss",
        },
        "optimized_policy": {
            "population": "all 715,023 saved policy customers",
            "action_bounds": [ACTION_LOW, ACTION_HIGH],
            "optimizer_entry_point": optimizer_summary["optimizer"]["entry_point"],
            "optimizer_step_rule": optimizer_summary["optimizer"]["step_rule"],
            "optimizer": "not rerun; exact saved repository-optimizer output replayed",
        },
        "inputs": {
            "support_cloud_csv": _file_record(cloud_path),
            "support_cloud_manifest": _file_record(cloud_manifest_path),
            "spline_policy_output": _file_record(policy_path),
            "spline_optimizer_summary": _file_record(optimizer_summary_path),
        },
        "outputs": {
            output_path.name: _file_record(output_path),
            histogram_path.name: _file_record(histogram_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (output_path, histogram_path, manifest_path):
        print(path, flush=True)


if __name__ == "__main__":
    main()
