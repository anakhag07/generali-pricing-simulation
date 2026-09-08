#!/usr/bin/env python3
"""Analyze a saved full-population spline policy by sensitivity tertile.

This replays an exact saved repository-optimizer output. Sensitivity follows
the existing acceptance-grid scaffold: mean absolute acceptance derivative
across the policy's action grid, split deterministically into low, medium, and
high tertiles.
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.full_monotone_spline_cache import ShardedMonotoneSplineCache, sha256_file
from experiments.paths import results_root
from experiments.sensitivity_buckets import SENSITIVITY_BUCKETS, split_sensitivity_tertiles
from experiments.slurm import submit_to_slurm_if_needed


ACTION_LOW = -0.1
ACTION_HIGH = 0.2
BIN_WIDTH = 0.01
EXPECTED_ROWS = 715_023


def _default_policy_result() -> Path:
    return results_root() / "glm-vs-full-monotone-spline-xgb-policy-seed-8"


def _default_cache() -> Path:
    return (
        results_root()
        / "cache"
        / "monotone-spline-xgb-full-v1"
        / "sweeps"
        / "full-715023-v1-20260812"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-result", type=Path, default=_default_policy_result())
    parser.add_argument("--spline-cache", type=Path, default=_default_cache())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            results_root()
            / "full-population-spline-policy-sensitivity-minus0p1-to-0p2-seed-8"
        ),
    )
    parser.add_argument("--grid-count", type=int, default=31)
    parser.add_argument("--chunk-rows", type=int, default=25_000)
    parser.add_argument("--launch", choices=("local", "slurm"), default="local")
    parser.add_argument("--no-sbatch", action="store_true")
    return parser


def mean_abs_sensitivity_scores(
    cache: Any,
    row_indices: np.ndarray,
    u_grid: np.ndarray,
    *,
    chunk_rows: int,
) -> np.ndarray:
    """Match the acceptance-grid sensitivity score without a full dense matrix."""
    rows = np.asarray(row_indices, dtype=np.int64)
    grid = np.asarray(u_grid, dtype=float)
    if rows.ndim != 1 or grid.ndim != 1 or rows.size < 1 or grid.size < 1:
        raise ValueError("row_indices and u_grid must be nonempty 1D arrays.")
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be positive.")
    scores = np.empty(rows.size, dtype=float)
    for start in range(0, rows.size, int(chunk_rows)):
        stop = min(start + int(chunk_rows), rows.size)
        derivative = np.asarray(cache.derivative(rows[start:stop], grid), dtype=float)
        if derivative.shape != (stop - start, grid.size):
            raise ValueError("Spline derivative matrix has an unexpected shape.")
        scores[start:stop] = np.mean(np.abs(derivative), axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("Sensitivity scores must be finite.")
    return scores


def _value_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def action_region_table(
    actions: np.ndarray,
    bucket_rank: np.ndarray,
    *,
    action_low: float = ACTION_LOW,
    action_high: float = ACTION_HIGH,
) -> pd.DataFrame:
    """Return bucket composition of the lower bin, interior, and upper bin."""
    values = np.asarray(actions, dtype=float)
    ranks = np.asarray(bucket_rank, dtype=int)
    if values.shape != ranks.shape:
        raise ValueError("actions and bucket_rank must have matching shapes.")
    regions = (
        ("lower_endpoint_bin", values < action_low + BIN_WIDTH),
        (
            "interior",
            (values >= action_low + BIN_WIDTH) & (values < action_high - BIN_WIDTH),
        ),
        ("upper_endpoint_bin", values >= action_high - BIN_WIDTH),
    )
    rows: list[dict[str, Any]] = []
    for region_name, region_mask in regions:
        region_count = int(np.sum(region_mask))
        for rank, bucket_name in enumerate(SENSITIVITY_BUCKETS):
            bucket_mask = ranks == rank
            count = int(np.sum(region_mask & bucket_mask))
            rows.append(
                {
                    "action_region": region_name,
                    "sensitivity_bucket": bucket_name,
                    "count": count,
                    "share_within_action_region": (
                        float(count / region_count) if region_count else 0.0
                    ),
                    "share_within_sensitivity_bucket": float(count / np.sum(bucket_mask)),
                }
            )
    return pd.DataFrame(rows)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _plot_action_distributions(
    actions: np.ndarray,
    bucket_rank: np.ndarray,
    output_path: Path,
) -> None:
    bins = np.linspace(ACTION_LOW, ACTION_HIGH, 31, dtype=float)
    bins[-1] = np.nextafter(ACTION_HIGH, np.inf)
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True, constrained_layout=True
    )
    for rank, (ax, bucket_name) in enumerate(zip(axes, SENSITIVITY_BUCKETS, strict=True)):
        bucket_actions = actions[bucket_rank == rank]
        ax.hist(bucket_actions, bins=bins, density=True, alpha=0.75, linewidth=0.5)
        ax.set_title(f"{bucket_name.capitalize()} sensitivity", fontsize=14)
        ax.set_xlabel("Price Change", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Density", fontsize=12)
    fig.suptitle(
        "Optimized Price Changes by Spline Acceptance Sensitivity [-0.1, 0.2]",
        fontsize=16,
    )
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _plot_bucket_metrics(bucket_summary: pd.DataFrame, output_path: Path) -> None:
    labels = bucket_summary["sensitivity_bucket"].tolist()
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].bar(x, bucket_summary["action_mean"])
    axes[0].set_ylabel("Mean Price Change", fontsize=12)
    axes[0].set_xlabel("Spline Acceptance Sensitivity", fontsize=12)
    axes[0].set_title("Mean Optimized Price Change", fontsize=14)
    axes[0].set_xticks(x, labels)
    axes[0].tick_params(labelsize=10)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, 100.0 * bucket_summary["upper_endpoint_bin_share"])
    axes[1].set_ylabel("Customers in [0.19, 0.2] (%)", fontsize=12)
    axes[1].set_xlabel("Spline Acceptance Sensitivity", fontsize=12)
    axes[1].set_title("Upper-Endpoint Decisions", fontsize=14)
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(labelsize=10)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Saved Spline Policy by Sensitivity Tertile", fontsize=16)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> list[Path]:
    if int(args.grid_count) < 2:
        raise ValueError("grid_count must be at least two.")
    policy_result = args.policy_result.expanduser().resolve()
    policy_path = policy_result / "spline_policy_seed_8.npz"
    optimizer_summary_path = policy_result / "summary.json"
    optimizer_summary = json.loads(optimizer_summary_path.read_text(encoding="utf-8"))
    if optimizer_summary["models"]["acceptance"] != (
        "exact full-cache monotone-spline XGBoost"
    ):
        raise ValueError("The saved policy must use exact spline acceptance.")
    if not np.allclose(
        optimizer_summary["comparison_contract"]["action_bounds"],
        [ACTION_LOW, ACTION_HIGH],
    ):
        raise ValueError("The saved policy must use action bounds [-0.1, 0.2].")
    with np.load(policy_path, allow_pickle=False) as payload:
        row_indices = np.asarray(payload["selected_row_indices"], dtype=np.int64)
        actions = np.asarray(payload["all_actions"], dtype=float)
        saved_bounds = np.asarray(payload["action_bounds"], dtype=float)
    if (
        row_indices.shape != (EXPECTED_ROWS,)
        or actions.shape != (EXPECTED_ROWS,)
        or not np.allclose(saved_bounds, [ACTION_LOW, ACTION_HIGH])
    ):
        raise ValueError("Saved policy arrays do not match the full-cohort contract.")

    cache_path = args.spline_cache.expanduser().resolve()
    cache = ShardedMonotoneSplineCache(cache_path, verify_checksums=False)
    if not np.array_equal(np.asarray(cache.eligible_row_indices), row_indices):
        raise ValueError("Spline cache rows do not match the saved policy rows.")
    u_grid = np.linspace(ACTION_LOW, ACTION_HIGH, int(args.grid_count), dtype=float)
    print(
        f"Scoring mean absolute spline sensitivity for {row_indices.size:,} customers...",
        flush=True,
    )
    sensitivity = mean_abs_sensitivity_scores(
        cache,
        row_indices,
        u_grid,
        chunk_rows=int(args.chunk_rows),
    )
    acceptance, local_derivative = cache.pairwise_acceptance_and_derivative(
        row_indices, actions
    )
    local_sensitivity = np.abs(local_derivative)

    buckets = split_sensitivity_tertiles(row_indices, sensitivity)
    bucket_rank = np.empty(row_indices.size, dtype=np.int8)
    summary_rows: list[dict[str, Any]] = []
    for rank, bucket in enumerate(buckets):
        positions = np.searchsorted(row_indices, bucket.row_indices)
        if not np.array_equal(row_indices[positions], bucket.row_indices):
            raise ValueError("Sensitivity bucket rows are not aligned to saved policy rows.")
        bucket_rank[positions] = rank
        bucket_actions = actions[positions]
        action_summary = _value_summary(bucket_actions)
        score_summary = _value_summary(sensitivity[positions])
        summary_rows.append(
            {
                "sensitivity_bucket": bucket.name,
                "bucket_rank": rank,
                "n_customers": int(positions.size),
                **{f"sensitivity_{key}": value for key, value in score_summary.items()},
                **{f"action_{key}": value for key, value in action_summary.items()},
                "upper_endpoint_bin_share": float(np.mean(bucket_actions >= 0.19)),
                "lower_endpoint_bin_share": float(np.mean(bucket_actions < -0.09)),
                "mean_local_abs_sensitivity_at_policy_action": float(
                    np.mean(local_sensitivity[positions])
                ),
                "mean_acceptance_at_policy_action": float(np.mean(acceptance[positions])),
            }
        )
    bucket_summary = pd.DataFrame(summary_rows)
    region_table = action_region_table(actions, bucket_rank)
    pearson = float(np.corrcoef(sensitivity, actions)[0, 1])
    spearman = float(spearmanr(sensitivity, actions).statistic)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_csv = output_dir / "sensitivity_bucket_summary.csv"
    region_csv = output_dir / "action_region_by_sensitivity.csv"
    arrays_path = output_dir / "customer_sensitivity_analysis.npz"
    distribution_pdf = output_dir / "optimized_actions_by_sensitivity.pdf"
    metrics_pdf = output_dir / "policy_metrics_by_sensitivity.pdf"
    bucket_summary.to_csv(bucket_csv, index=False)
    region_table.to_csv(region_csv, index=False)
    np.savez_compressed(
        arrays_path,
        selected_row_indices=row_indices,
        optimized_actions=actions,
        mean_abs_sensitivity=sensitivity,
        local_abs_sensitivity_at_policy_action=local_sensitivity,
        acceptance_at_policy_action=acceptance,
        sensitivity_bucket_rank=bucket_rank,
        sensitivity_grid=u_grid,
        action_bounds=np.asarray([ACTION_LOW, ACTION_HIGH]),
    )
    _plot_action_distributions(actions, bucket_rank, distribution_pdf)
    _plot_bucket_metrics(bucket_summary, metrics_pdf)

    upper_rows = region_table[region_table["action_region"] == "upper_endpoint_bin"]
    outputs = [bucket_csv, region_csv, arrays_path, distribution_pdf, metrics_pdf]
    summary = {
        "analysis": "saved-full-population-spline-policy-by-sensitivity-tertile",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "population_rows": int(row_indices.size),
        "action_bounds": [ACTION_LOW, ACTION_HIGH],
        "sensitivity_definition": (
            "mean absolute exact-spline acceptance derivative across an evenly spaced "
            "grid over [-0.1, 0.2]"
        ),
        "sensitivity_grid_count": int(u_grid.size),
        "bucket_definition": (
            "deterministic equal-sized tertiles via "
            "experiments.sensitivity_buckets.split_sensitivity_tertiles"
        ),
        "action_sensitivity_association": {
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
        },
        "bucket_summary": bucket_summary.to_dict(orient="records"),
        "upper_endpoint_bin": {
            "range": [0.19, 0.2],
            "n_customers": int(np.sum(actions >= 0.19)),
            "sensitivity_composition": {
                str(row["sensitivity_bucket"]): float(row["share_within_action_region"])
                for _, row in upper_rows.iterrows()
            },
        },
        "provenance": {
            "reported_solution": "exact replay of saved repository optimizer output",
            "optimizer_entry_point": optimizer_summary["optimizer"]["entry_point"],
            "optimizer_step_rule": optimizer_summary["optimizer"]["step_rule"],
            "policy_output": _file_record(policy_path),
            "optimizer_summary": _file_record(optimizer_summary_path),
            "spline_cache_manifest": _file_record(cache_path / "manifest.json"),
        },
        "outputs": {path.name: _file_record(path) for path in outputs},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(bucket_summary.to_string(index=False), flush=True)
    print(region_table.to_string(index=False), flush=True)
    for path in [*outputs, summary_path]:
        print(path, flush=True)
    return [*outputs, summary_path]


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    if args.launch == "slurm":
        submission = submit_to_slurm_if_needed(
            requires_jax=False,
            no_sbatch=bool(args.no_sbatch),
            argv=original_argv,
            cwd=ROOT,
        )
        if submission is not None:
            print(
                f"Submitted {submission.profile.name} Slurm job {submission.job_id}; "
                f"logs: {submission.profile.output}",
                flush=True,
            )
            return
    run_analysis(args)


if __name__ == "__main__":
    main()
