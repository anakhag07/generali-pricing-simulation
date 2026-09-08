#!/usr/bin/env python3
"""Plot a saved full-population spline policy for an acceptance constraint."""

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

from data.full_monotone_spline_cache import sha256_file
from experiments.paths import results_root


ACTION_LOW = -0.1
ACTION_HIGH = 0.2
BIN_WIDTH = 0.01
OPTIMIZED_COLOR = "#86002d"


def _default_policy_result() -> Path:
    return results_root() / "glm-vs-full-monotone-spline-xgb-policy-acceptance-0p90-seed-8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-result", type=Path, default=_default_policy_result())
    parser.add_argument("--acceptance-floor", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_policy_result(),
    )
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


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    policy_result = args.policy_result.expanduser().resolve()
    policy_path = policy_result / "spline_policy_seed_8.npz"
    optimizer_summary_path = policy_result / "summary.json"
    optimizer_summary = json.loads(optimizer_summary_path.read_text(encoding="utf-8"))
    expected_floor = float(args.acceptance_floor)
    if not np.isclose(
        optimizer_summary["comparison_contract"]["acceptance_floor"], expected_floor
    ):
        raise ValueError("Saved policy has an unexpected acceptance floor.")
    if not np.allclose(
        optimizer_summary["comparison_contract"]["action_bounds"],
        [ACTION_LOW, ACTION_HIGH],
    ):
        raise ValueError("Saved policy has unexpected action bounds.")
    if optimizer_summary["models"]["acceptance"] != (
        "exact full-cache monotone-spline XGBoost"
    ):
        raise ValueError("Saved policy must use exact spline acceptance.")
    if optimizer_summary["models"]["loss"] != "XGBoost financial loss":
        raise ValueError("Saved policy must use XGBoost financial loss.")
    with np.load(policy_path, allow_pickle=False) as payload:
        actions = np.asarray(payload["all_actions"], dtype=float)
        bounds = np.asarray(payload["action_bounds"], dtype=float)
    if actions.shape != (715_023,) or not np.allclose(bounds, [ACTION_LOW, ACTION_HIGH]):
        raise ValueError("Saved arrays do not match the full-population policy contract.")
    histogram = histogram_frame(actions)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    floor_percent = 100.0 * expected_floor
    floor_slug = f"{floor_percent:g}".replace(".", "p")
    pdf_path = output_dir / f"constrained_optimizer_behavior_{floor_slug}_spline.pdf"
    csv_path = output_dir / f"constrained_optimizer_behavior_{floor_slug}_spline_histogram.csv"
    manifest_path = output_dir / f"constrained_optimizer_behavior_{floor_slug}_spline_manifest.json"

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
    ax.set_title(
        f"Constrained Optimizer Behavior — {floor_percent:g}% (Spline/XGBoost)",
        fontsize=16,
    )
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    histogram.to_csv(csv_path, index=False)

    manifest = {
        "analysis": "full-population-spline-policy-acceptance-constraint-histogram",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acceptance_floor": expected_floor,
        "action_bounds": [ACTION_LOW, ACTION_HIGH],
        "population": "all 715,023 saved policy customers",
        "models": {
            "acceptance": "exact full-cache monotone-spline XGBoost",
            "loss": "XGBoost financial loss",
        },
        "optimizer": {
            "entry_point": optimizer_summary["optimizer"]["entry_point"],
            "step_rule": optimizer_summary["optimizer"]["step_rule"],
            "not_rerun_for_plot": True,
            "provenance": "exact saved repository-optimizer output replayed",
        },
        "optimized_metrics": optimizer_summary["spline"]["optimized_all"],
        "inputs": {
            "policy_output": _file_record(policy_path),
            "optimizer_summary": _file_record(optimizer_summary_path),
        },
        "outputs": {
            pdf_path.name: _file_record(pdf_path),
            csv_path.name: _file_record(csv_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in (pdf_path, csv_path, manifest_path):
        print(path, flush=True)


if __name__ == "__main__":
    main()
