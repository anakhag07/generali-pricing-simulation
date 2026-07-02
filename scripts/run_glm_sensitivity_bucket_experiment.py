"""Run GLM pricing experiments on low/medium/high sensitivity buckets."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.results import ExperimentResult
from experiments.sensitivity_buckets import (
    SensitivityBucket,
    build_glm_sensitivity_buckets,
    median_observed_u,
)
from reporting.visualization import _estimator_style

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-sensitivity-buckets"
RUN_OVERRIDES = {
    "policy_kind": "softmax",
    "policy_preprocessing": "no_pca",
    "feature_order": "linear",
    "constraint_mode": "trust_constr",
    "n_grad_samples": 8,
    "t_steps": 100,
    "enabled_estimators": ("first_order", "finite_difference", "stein_difference"),
    "plot": True,
    "wandb_enabled": False,
}

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_BUCKET_RANK = {"low": 0, "medium": 1, "high": 2}


def _run_bucket(bucket: SensitivityBucket) -> ExperimentResult:
    overrides = {
        **RUN_OVERRIDES,
        "n_samples": int(bucket.row_indices.size),
        "row_indices": bucket.row_indices,
    }
    config = get_config(BASE_PRESET, overrides=overrides)
    executed = execute_experiment_run(
        f"{bucket.name}_sensitivity",
        config,
        runs_root=str(Path("outputs") / PROJECT_NAME),
    )
    return executed.result


def _policy_u_values(result: ExperimentResult, estimator: str) -> np.ndarray:
    objective = result.config.objective
    theta = result.results[estimator].theta
    policy_value = getattr(objective, "policy_value", None)
    if not callable(policy_value):
        return np.asarray([], dtype=float)
    u_values = np.asarray(policy_value(theta, result.x_samples), dtype=float).reshape(-1)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float).reshape(-1)
    return u_values


def _acceptance_values(result: ExperimentResult, u_values: np.ndarray) -> np.ndarray:
    acceptance_fn = getattr(result.config.objective, "_acceptance_proba", None)
    if not callable(acceptance_fn) or u_values.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(acceptance_fn(result.x_samples, u_values), dtype=float).reshape(-1)


def _summary(prefix: str, values: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_q05": "",
            f"{prefix}_q25": "",
            f"{prefix}_q50": "",
            f"{prefix}_q75": "",
            f"{prefix}_q95": "",
        }
    quantiles = np.quantile(finite, _QUANTILES)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
    }


def _collect_rows(
    bucket_results: Sequence[tuple[SensitivityBucket, ExperimentResult]],
    *,
    u_ref: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket, result in bucket_results:
        score_summary = _summary("sensitivity", bucket.scores)
        for estimator, estimator_result in result.results.items():
            u_values = _policy_u_values(result, estimator)
            acceptance_values = _acceptance_values(result, u_values)
            row: dict[str, object] = {
                "bucket": bucket.name,
                "bucket_rank": _BUCKET_RANK[bucket.name],
                "u_ref": float(u_ref),
                "n_rows": int(bucket.row_indices.size),
                "row_index_min": int(np.min(bucket.row_indices)),
                "row_index_max": int(np.max(bucket.row_indices)),
                "estimator": estimator,
                "u": float(estimator_result.u),
                "mean_acceptance": (
                    float(estimator_result.mean_acceptance)
                    if estimator_result.mean_acceptance is not None
                    else ""
                ),
                "value": float(estimator_result.value),
                "runtime_sec": float(estimator_result.time),
                "constraint_violation": (
                    float(estimator_result.constraint_violation)
                    if estimator_result.constraint_violation is not None
                    else ""
                ),
            }
            row.update(score_summary)
            row.update(_summary("u", u_values))
            row.update(_summary("acceptance", acceptance_values))
            rows.append(row)
    return rows


def _write_rows(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "glm_sensitivity_bucket_experiment.csv"
    fieldnames = [
        "bucket",
        "bucket_rank",
        "u_ref",
        "n_rows",
        "row_index_min",
        "row_index_max",
        "estimator",
        "u",
        "mean_acceptance",
        "value",
        "runtime_sec",
        "constraint_violation",
        "sensitivity_mean",
        "sensitivity_q05",
        "sensitivity_q25",
        "sensitivity_q50",
        "sensitivity_q75",
        "sensitivity_q95",
        "u_mean",
        "u_q05",
        "u_q25",
        "u_q50",
        "u_q75",
        "u_q95",
        "acceptance_mean",
        "acceptance_q05",
        "acceptance_q25",
        "acceptance_q50",
        "acceptance_q75",
        "acceptance_q95",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_by_estimator(rows: Sequence[Mapping[str, object]]) -> list[tuple[str, list[Mapping[str, object]]]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["estimator"]), []).append(row)
    ordered_names = ["first_order", "finite_difference", "stein_difference"]
    ordered = [name for name in ordered_names if name in grouped]
    ordered.extend(sorted(name for name in grouped if name not in ordered))
    return [
        (name, sorted(grouped[name], key=lambda row: int(row["bucket_rank"])))
        for name in ordered
    ]


def _plot_bucket_tradeoffs(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_u, ax_acceptance = axes
    bucket_labels = ["low", "medium", "high"]
    x_values = np.arange(len(bucket_labels), dtype=float)

    for estimator, estimator_rows in _rows_by_estimator(rows):
        style = _estimator_style(estimator)
        ranks = [int(row["bucket_rank"]) for row in estimator_rows]
        plot_kwargs = {
            "label": str(style["label"]),
            "color": style["color"],
            "linewidth": 1.8,
            "marker": style["marker"],
            "markersize": float(style.get("marker_size", 6.0)),
            "alpha": 0.9,
        }
        ax_u.plot(ranks, [float(row["u"]) for row in estimator_rows], **plot_kwargs)
        ax_acceptance.plot(
            ranks,
            [float(row["mean_acceptance"]) for row in estimator_rows],
            **plot_kwargs,
        )

    ax_u.set_ylabel("Final mean u")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)
    ax_acceptance.set_ylabel("Mean acceptance")
    ax_acceptance.set_xlabel("Sensitivity bucket")
    ax_acceptance.set_xticks(x_values)
    ax_acceptance.set_xticklabels(bucket_labels)
    ax_acceptance.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "bucket_vs_u_acceptance.png", dpi=200)
    plt.close(fig)


def _plot_bucket_pareto(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    y_key: str,
    y_label: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for estimator, estimator_rows in _rows_by_estimator(rows):
        style = _estimator_style(estimator)
        x_values = [float(row["mean_acceptance"]) for row in estimator_rows]
        y_values = [float(row[y_key]) for row in estimator_rows]
        ax.plot(x_values, y_values, color=style["color"], alpha=0.35, linewidth=1.0)
        ax.scatter(
            x_values,
            y_values,
            label=str(style["label"]),
            color=style["color"],
            marker=style["marker"],
            s=float(style.get("scatter_size", 45.0)),
            alpha=0.9,
        )
        for row, x_val, y_val in zip(estimator_rows, x_values, y_values):
            ax.annotate(str(row["bucket"]), (x_val, y_val), fontsize=8, alpha=0.75)
    ax.set_xlabel("Mean acceptance")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _plot_score_histograms(buckets: Sequence[SensitivityBucket], output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for bucket in buckets:
        ax.hist(
            bucket.scores,
            bins=50,
            alpha=0.35,
            density=True,
            label=bucket.name,
        )
    ax.set_xlabel("Sensitivity score |d p_accept / du|")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sensitivity_score_histograms.png", dpi=200)
    plt.close(fig)


def _write_plots(
    rows: Sequence[Mapping[str, object]],
    buckets: Sequence[SensitivityBucket],
    output_dir: Path,
) -> None:
    _plot_bucket_tradeoffs(rows, output_dir)
    _plot_bucket_pareto(
        rows,
        output_dir,
        y_key="value",
        y_label="Final objective value",
        filename="bucket_objective_acceptance.png",
    )
    _plot_bucket_pareto(
        rows,
        output_dir,
        y_key="u",
        y_label="Final mean u",
        filename="bucket_u_acceptance.png",
    )
    _plot_score_histograms(buckets, output_dir)


def main() -> None:
    u_ref = median_observed_u("glm")
    buckets = build_glm_sensitivity_buckets(u_ref=u_ref)
    bucket_results: list[tuple[SensitivityBucket, ExperimentResult]] = []
    for bucket in buckets:
        print(
            f"[sensitivity-buckets] start bucket={bucket.name} "
            f"n_rows={bucket.row_indices.size} "
            f"score_median={float(np.median(bucket.scores)):.6f}",
            flush=True,
        )
        result = _run_bucket(bucket)
        bucket_results.append((bucket, result))

    rows = _collect_rows(bucket_results, u_ref=u_ref)
    if not rows:
        raise ValueError("No sensitivity bucket rows were produced.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"sensitivity_bucket_summary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir)
    _write_plots(rows, buckets, output_dir)

    print(f"Completed {len(bucket_results)} sensitivity bucket runs for preset '{BASE_PRESET}'.")
    print(f"Wrote bucket summary and aggregate plots to {output_dir}.")


if __name__ == "__main__":
    main()
