"""Run first-order GLM experiments across reference elasticity buckets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan, task_payloads
from experiments.results import ExperimentResult
from experiments.sensitivity_buckets import SensitivityBucket, build_glm_sensitivity_buckets
from reporting.visualization import _estimator_style

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-reference-elasticity-buckets"
REFERENCE_U_VALUES = (-0.1, 0.1, 0.2, 0.3)
RUN_OVERRIDES = {
    "policy_kind": "softmax",
    "policy_preprocessing": "no_pca",
    "feature_order": "linear",
    "constraint_mode": "trust_constr",
    "n_grad_samples": 8,
    "t_steps": 100,
    "enabled_estimators": ("first_order",),
    "plot": True,
    "wandb_enabled": False,
    "train_fraction": 0.8,
    "test_fraction": 0.2,
    "softmax_action_bounds": (-0.1, 0.2),
}

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_BUCKET_RANK = {"low": 0, "medium": 1, "high": 2}


def _u_label(u_ref: float) -> str:
    return f"u_ref_{u_ref:+.1f}".replace("+", "plus").replace("-", "minus").replace(".", "p")


def _run_bucket(bucket: SensitivityBucket, *, u_ref: float) -> ExperimentResult:
    overrides = {
        **RUN_OVERRIDES,
        "n_samples": int(bucket.row_indices.size),
        "row_indices": bucket.row_indices,
    }
    config = get_config(BASE_PRESET, overrides=overrides)
    executed = execute_experiment_run(
        f"{_u_label(u_ref)}_{bucket.name}_elasticity",
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
    bucket_results: Sequence[tuple[float, SensitivityBucket, ExperimentResult]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for u_ref, bucket, result in bucket_results:
        score_summary = _summary("elasticity_abs", bucket.scores)
        for estimator, estimator_result in result.results.items():
            u_values = _policy_u_values(result, estimator)
            acceptance_values = _acceptance_values(result, u_values)
            row: dict[str, object] = {
                "u_ref": float(u_ref),
                "bucket": bucket.name,
                "bucket_rank": _BUCKET_RANK[bucket.name],
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


def _fieldnames() -> list[str]:
    return [
        "u_ref",
        "bucket",
        "bucket_rank",
        "n_rows",
        "row_index_min",
        "row_index_max",
        "estimator",
        "u",
        "mean_acceptance",
        "value",
        "runtime_sec",
        "constraint_violation",
        "elasticity_abs_mean",
        "elasticity_abs_q05",
        "elasticity_abs_q25",
        "elasticity_abs_q50",
        "elasticity_abs_q75",
        "elasticity_abs_q95",
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


def _write_rows(rows: Sequence[Mapping[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows_for_u_ref(rows: Sequence[Mapping[str, object]], u_ref: float) -> list[Mapping[str, object]]:
    selected = [row for row in rows if float(row["u_ref"]) == float(u_ref)]
    return sorted(selected, key=lambda row: int(row["bucket_rank"]))


def _annotate_bucket_elasticity(ax: plt.Axes, row: Mapping[str, object], xy: tuple[float, float]) -> None:
    ax.annotate(
        f"{row['bucket']}\navg |elast|={float(row['elasticity_abs_mean']):.3f}",
        xy,
        fontsize=8,
        alpha=0.8,
        textcoords="offset points",
        xytext=(4, 4),
    )


def _plot_bucket_tradeoffs(rows: Sequence[Mapping[str, object]], output_dir: Path, *, u_ref: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax_u, ax_acceptance = axes
    bucket_labels = ["low", "medium", "high"]
    x_values = np.arange(len(bucket_labels), dtype=float)
    style = _estimator_style("first_order")
    ranks = [int(row["bucket_rank"]) for row in rows]
    u_values = [float(row["u"]) for row in rows]
    acceptance_values = [float(row["mean_acceptance"]) for row in rows]
    plot_kwargs = {
        "label": str(style["label"]),
        "color": style["color"],
        "linewidth": 1.8,
        "marker": style["marker"],
        "markersize": float(style.get("marker_size", 6.0)),
        "alpha": 0.9,
    }
    ax_u.plot(ranks, u_values, **plot_kwargs)
    ax_acceptance.plot(ranks, acceptance_values, **plot_kwargs)
    for row, rank, u_val, acceptance in zip(rows, ranks, u_values, acceptance_values):
        _annotate_bucket_elasticity(ax_u, row, (rank, u_val))
        _annotate_bucket_elasticity(ax_acceptance, row, (rank, acceptance))

    ax_u.set_title(f"Reference u={u_ref:.1f}")
    ax_u.set_ylabel("Final mean u")
    ax_u.legend()
    ax_u.grid(True, alpha=0.3)
    ax_acceptance.set_ylabel("Mean acceptance")
    ax_acceptance.set_xlabel("Magnitude-ranked elasticity bucket")
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
    u_ref: float,
    y_key: str,
    y_label: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    style = _estimator_style("first_order")
    x_values = [float(row["mean_acceptance"]) for row in rows]
    y_values = [float(row[y_key]) for row in rows]
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
    for row, x_val, y_val in zip(rows, x_values, y_values):
        _annotate_bucket_elasticity(ax, row, (x_val, y_val))
    ax.set_title(f"Reference u={u_ref:.1f}")
    ax.set_xlabel("Mean acceptance")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _plot_score_histograms(buckets: Sequence[SensitivityBucket], output_dir: Path, *, u_ref: float) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for bucket in buckets:
        ax.hist(
            bucket.scores,
            bins=50,
            alpha=0.35,
            density=True,
            label=f"{bucket.name} (mean={float(np.mean(bucket.scores)):.3f})",
        )
    ax.set_title(f"Reference u={u_ref:.1f}")
    ax.set_xlabel("Elasticity magnitude |d p_accept / du|")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "elasticity_score_histograms.png", dpi=200)
    plt.close(fig)


def _write_plots(
    rows: Sequence[Mapping[str, object]],
    buckets: Sequence[SensitivityBucket],
    output_dir: Path,
    *,
    u_ref: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_bucket_tradeoffs(rows, output_dir, u_ref=u_ref)
    _plot_bucket_pareto(
        rows,
        output_dir,
        u_ref=u_ref,
        y_key="value",
        y_label="Final objective value",
        filename="bucket_objective_acceptance.png",
    )
    _plot_bucket_pareto(
        rows,
        output_dir,
        u_ref=u_ref,
        y_key="u",
        y_label="Final mean u",
        filename="bucket_u_acceptance.png",
    )
    _plot_score_histograms(buckets, output_dir, u_ref=u_ref)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_args(parser, default_launch="local", default_array=False)
    return parser.parse_args(argv)


def _task_ref_bucket(index: int) -> tuple[float, SensitivityBucket]:
    u_ref = float(REFERENCE_U_VALUES[index // 3])
    bucket_index = index % 3
    buckets = build_glm_sensitivity_buckets(u_ref=u_ref)
    return u_ref, buckets[bucket_index]


def _run_reference_bucket_task(index: int, context: LaunchContext) -> dict[str, object]:
    del context
    u_ref, bucket = _task_ref_bucket(index)
    print(
        f"[reference-elasticity-buckets] start u_ref={u_ref:.1f} "
        f"bucket={bucket.name} n_rows={bucket.row_indices.size} "
        f"mean_abs_elasticity={float(np.mean(bucket.scores)):.6f}",
        flush=True,
    )
    result = _run_bucket(bucket, u_ref=u_ref)
    rows = _collect_rows([(u_ref, bucket, result)])
    return {"u_ref": u_ref, "bucket": bucket.name, "rows": rows}


def _run_reference_buckets_serial(context: LaunchContext) -> None:
    del context
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"reference_elasticity_bucket_summary_{timestamp}"
    all_bucket_results: list[tuple[float, SensitivityBucket, ExperimentResult]] = []
    buckets_by_u: dict[float, tuple[SensitivityBucket, SensitivityBucket, SensitivityBucket]] = {}

    for u_ref in REFERENCE_U_VALUES:
        buckets = build_glm_sensitivity_buckets(u_ref=float(u_ref))
        buckets_by_u[float(u_ref)] = buckets
        for bucket in buckets:
            print(
                f"[reference-elasticity-buckets] start u_ref={u_ref:.1f} "
                f"bucket={bucket.name} n_rows={bucket.row_indices.size} "
                f"mean_abs_elasticity={float(np.mean(bucket.scores)):.6f}",
                flush=True,
            )
            result = _run_bucket(bucket, u_ref=float(u_ref))
            all_bucket_results.append((float(u_ref), bucket, result))

    rows = _collect_rows(all_bucket_results)
    if not rows:
        raise ValueError("No reference elasticity bucket rows were produced.")
    _write_rows(rows, output_dir / "glm_reference_elasticity_bucket_experiment.csv")
    for u_ref, buckets in buckets_by_u.items():
        rows_for_ref = _rows_for_u_ref(rows, u_ref)
        _write_rows(rows_for_ref, output_dir / _u_label(u_ref) / "glm_reference_elasticity_bucket_experiment.csv")
        _write_plots(rows_for_ref, buckets, output_dir / _u_label(u_ref), u_ref=u_ref)

    print(f"Completed {len(all_bucket_results)} reference elasticity bucket runs.")
    print(f"Wrote bucket summary and aggregate plots to {output_dir}.")


def _collect_reference_bucket_tasks(context: LaunchContext) -> None:
    payloads = task_payloads(context)
    rows = [row for payload in payloads for row in payload.get("rows", [])]
    if not rows:
        raise ValueError("No reference elasticity bucket rows were produced.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"reference_elasticity_bucket_summary_{timestamp}"
    _write_rows(rows, output_dir / "glm_reference_elasticity_bucket_experiment.csv")
    for u_ref in REFERENCE_U_VALUES:
        buckets = build_glm_sensitivity_buckets(u_ref=float(u_ref))
        rows_for_ref = _rows_for_u_ref(rows, float(u_ref))
        _write_rows(rows_for_ref, output_dir / _u_label(float(u_ref)) / "glm_reference_elasticity_bucket_experiment.csv")
        _write_plots(rows_for_ref, buckets, output_dir / _u_label(float(u_ref)), u_ref=float(u_ref))
    print(f"Collected {len(payloads)} reference elasticity bucket array tasks.")
    print(f"Wrote bucket summary and aggregate plots to {output_dir}.")


def _build_launch_plan() -> LaunchPlan:
    return LaunchPlan(
        name=PROJECT_NAME,
        task_count=len(REFERENCE_U_VALUES) * 3,
        requires_jax=RUN_OVERRIDES.get("compute_backend") == "jax",
        run_task=_run_reference_bucket_task,
        run_all=_run_reference_buckets_serial,
        collect=_collect_reference_bucket_tasks,
        runs_root="outputs",
        default_launch="local",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
