"""Plot saved homoskedastic-noise sweep theta distances to first-order truth."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.paths import results_root


def _default_sweep_dir() -> Path:
    return results_root() / "homoskedastic-noise-sweep"


def _default_truth_summary() -> Path:
    return (
        results_root()
        / "planted_logistic_base"
        / "first_order_truth_20260701_174139"
        / "summary.json"
    )


ESTIMATORS = ("finite_difference", "stein_difference")


@dataclass(frozen=True)
class DistanceRow:
    noise_std: float
    estimator: str
    distance_l2_to_truth: float
    optimizer_success: bool | None
    final_value: float
    summary_path: Path


def main() -> None:
    args = _parse_args()
    output_path = args.output or args.sweep_dir / "theta_distance_to_first_order_truth_by_noise.png"
    csv_path = args.csv_output or args.sweep_dir / "theta_distance_to_first_order_truth_by_noise.csv"

    truth_theta = _truth_theta(args.truth_summary)
    rows = _collect_rows(args.sweep_dir, truth_theta)
    _write_rows(csv_path, rows)
    _plot_rows(output_path, rows)

    print(f"Wrote theta-distance CSV to {csv_path}")
    print(f"Wrote theta-distance plot to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=_default_sweep_dir(),
        help="Saved homoskedastic-noise sweep directory.",
    )
    parser.add_argument(
        "--truth-summary",
        type=Path,
        default=_default_truth_summary(),
        help="summary.json containing the first_order truth theta.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults inside --sweep-dir.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Defaults inside --sweep-dir.",
    )
    return parser.parse_args()


def _collect_rows(sweep_dir: Path, truth_theta: np.ndarray) -> list[DistanceRow]:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    rows: list[DistanceRow] = []
    for variant_dir in sorted(sweep_dir.iterdir(), key=_variant_sort_key):
        if not variant_dir.is_dir():
            continue
        for summary_path in _summary_paths(variant_dir):
            summary = _load_json(summary_path)
            noise_std = _noise_std(summary, variant_dir.name)
            for estimator in ESTIMATORS:
                theta = _estimator_theta(summary, estimator, summary_path)
                rows.append(
                    DistanceRow(
                        noise_std=noise_std,
                        estimator=estimator,
                        distance_l2_to_truth=float(np.linalg.norm(theta - truth_theta)),
                        optimizer_success=_optimizer_success(summary, estimator),
                        final_value=float(summary["estimators"][estimator]["final_value"]),
                        summary_path=summary_path,
                    )
                )
    if not rows:
        raise ValueError(f"No homoskedastic-noise summary rows found under {sweep_dir}")
    return rows


def _summary_paths(variant_dir: Path) -> list[Path]:
    # Current saved sweep uses one timestamped run dir per noise std. Keep support for
    # seed-aware variant-level summaries if this scratch plot is reused later.
    seed_summaries = sorted(variant_dir.glob("summary-seed-*.json"))
    if seed_summaries:
        return seed_summaries
    direct_summary = variant_dir / "summary.json"
    if direct_summary.exists():
        return [direct_summary]
    return sorted(variant_dir.glob("*/summary.json"))


def _truth_theta(truth_summary: Path) -> np.ndarray:
    summary = _load_json(truth_summary)
    return _estimator_theta(summary, "first_order", truth_summary)


def _estimator_theta(summary: dict[str, Any], estimator: str, summary_path: Path) -> np.ndarray:
    try:
        theta = summary["estimators"][estimator]["theta"]
    except KeyError as exc:
        raise KeyError(f"Missing estimator '{estimator}' theta in {summary_path}") from exc
    return np.asarray(theta, dtype=float)


def _optimizer_success(summary: dict[str, Any], estimator: str) -> bool | None:
    value = summary["estimators"][estimator].get("optimizer_success")
    return None if value is None else bool(value)


def _noise_std(summary: dict[str, Any], variant_name: str) -> float:
    objective = summary.get("config", {}).get("objective", {})
    noise = objective.get("noise", {}) if isinstance(objective, dict) else {}
    if "std" in noise:
        return float(noise["std"])
    if variant_name.startswith("noise-std-"):
        return float(variant_name.removeprefix("noise-std-"))
    raise KeyError(f"Could not resolve noise std for variant '{variant_name}'")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_rows(path: Path, rows: list[DistanceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "noise_std",
        "estimator",
        "distance_l2_to_truth",
        "optimizer_success",
        "final_value",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        sorted_rows = sorted(
            rows,
            key=lambda item: (item.noise_std, item.estimator, str(item.summary_path)),
        )
        for row in sorted_rows:
            writer.writerow(
                {
                    "noise_std": row.noise_std,
                    "estimator": row.estimator,
                    "distance_l2_to_truth": row.distance_l2_to_truth,
                    "optimizer_success": "" if row.optimizer_success is None else row.optimizer_success,
                    "final_value": row.final_value,
                    "summary_path": str(row.summary_path),
                }
            )


def _plot_rows(path: Path, rows: list[DistanceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))

    styles = {
        "finite_difference": {"label": "Finite difference", "color": "tab:blue", "marker": "o"},
        "stein_difference": {"label": "Stein difference", "color": "tab:orange", "marker": "s"},
    }
    all_noise_stds = sorted({row.noise_std for row in rows})

    for estimator in ESTIMATORS:
        selected = [row for row in rows if row.estimator == estimator]
        if not selected:
            continue
        noise_stds = sorted({row.noise_std for row in selected})
        means = [_mean_distance(selected, noise_std) for noise_std in noise_stds]
        stds = [_std_distance(selected, noise_std) for noise_std in noise_stds]
        style = styles[estimator]
        yerr = stds if any(std > 0.0 for std in stds) else None
        ax.errorbar(
            noise_stds,
            means,
            yerr=yerr,
            label=str(style["label"]),
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=1.8,
            markersize=5.5,
            capsize=3.0,
        )
        failed_noise_stds = [noise_std for noise_std in noise_stds if _has_failed(selected, noise_std)]
        if failed_noise_stds:
            failed_means = [_mean_distance(selected, noise_std) for noise_std in failed_noise_stds]
            ax.scatter(
                failed_noise_stds,
                failed_means,
                label=f"{style['label']} optimizer_success=False",
                color=str(style["color"]),
                marker="x",
                s=60,
                linewidths=1.5,
                zorder=4,
            )

    _set_noise_axis(ax, all_noise_stds)
    _set_distance_axis(ax, [row.distance_l2_to_truth for row in rows])
    ax.set_xlabel("Homoskedastic noise std")
    ax.set_ylabel(r"$||\theta_{estimator} - \theta_{first\ order\ truth}||_2$")
    ax.set_title("Final theta distance to first-order truth by noise")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _mean_distance(rows: list[DistanceRow], noise_std: float) -> float:
    values = [row.distance_l2_to_truth for row in rows if row.noise_std == noise_std]
    return float(np.mean(values))


def _std_distance(rows: list[DistanceRow], noise_std: float) -> float:
    values = [row.distance_l2_to_truth for row in rows if row.noise_std == noise_std]
    return float(np.std(values, ddof=0))


def _has_failed(rows: list[DistanceRow], noise_std: float) -> bool:
    return any(row.noise_std == noise_std and row.optimizer_success is False for row in rows)


def _set_noise_axis(ax: plt.Axes, noise_stds: list[float]) -> None:
    nonzero = [abs(noise_std) for noise_std in noise_stds if noise_std != 0.0]
    if nonzero:
        ax.set_xscale("symlog", linthresh=min(nonzero))
    ax.set_xticks(noise_stds)
    ax.set_xticklabels([f"{noise_std:g}" for noise_std in noise_stds])


def _set_distance_axis(ax: plt.Axes, distances: list[float]) -> None:
    if all(distance > 0.0 for distance in distances):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-8)


def _variant_sort_key(path: Path) -> tuple[int, float | str]:
    if path.name.startswith("noise-std-"):
        return (0, float(path.name.removeprefix("noise-std-")))
    return (1, path.name)


if __name__ == "__main__":
    main()
