"""Aggregate tables and plots for continuous-GP regret decomposition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.policy_lcb.continuous_gp_core import (
    DecomposedGPLandscape,
    FourierGPDraw,
    analytic_uniform_certificate,
    true_value,
)
from experiments.sweep_reporting import write_rows_csv


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: tuple[str, ...],
    numeric: tuple[str, ...],
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[field] for field in groups), []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(buckets, key=lambda values: tuple(str(value) for value in values)):
        bucket = buckets[key]
        record: dict[str, object] = dict(zip(groups, key))
        record["n"] = len(bucket)
        for field in numeric:
            values = np.asarray([float(row[field]) for row in bucket], dtype=float)
            record[f"{field}_mean"] = float(np.mean(values))
            record[f"{field}_median"] = float(np.median(values))
            record[f"{field}_q05"] = float(np.quantile(values, 0.05))
            record[f"{field}_q95"] = float(np.quantile(values, 0.95))
        output.append(record)
    return output


def _membership_rows(
    rows: Sequence[Mapping[str, Any]], memberships: set[str] | None = None
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        for membership in str(row["memberships"]).split("|"):
            if memberships is None or membership in memberships:
                output.append({**row, "sweep": membership})
    return output


def _cross_validated_predictions(
    rows: Sequence[Mapping[str, Any]], features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray([float(row["true_regret"]) for row in rows], dtype=float)
    x = np.asarray([[float(row[field]) for field in features] for row in rows], dtype=float)
    seeds = np.asarray([int(row["run_seed"]) for row in rows], dtype=int)
    unique = np.asarray(sorted(set(seeds)))
    folds = {int(seed): index % min(5, len(unique)) for index, seed in enumerate(unique)}
    predictions = np.full_like(y, np.nan)
    for fold in sorted(set(folds.values())):
        test = np.asarray([folds[int(seed)] == fold for seed in seeds])
        train = ~test
        train_x = np.column_stack([np.ones(np.sum(train)), x[train]])
        coefficients, *_ = np.linalg.lstsq(train_x, y[train], rcond=None)
        predictions[test] = np.column_stack([np.ones(np.sum(test)), x[test]]) @ coefficients
    return y, predictions, seeds


def _model_summary(
    rows: Sequence[Mapping[str, Any]], reporting_seed: int
) -> list[dict[str, object]]:
    eligible = [
        row
        for row in rows
        if bool(row["certificate_eligible"])
        and np.isfinite(float(row["true_regret"]))
        and ("combined_factorial" in str(row["memberships"]) or "shape_robustness" in str(row["memberships"]))
    ]
    models = {
        "envelope_term": ("envelope_term",),
        "optimizer_error": ("optimization_error_lower",),
        "additive_decomposition": ("envelope_term", "optimization_error_lower"),
        "add_shape_geometry": (
            "envelope_term",
            "optimization_error_lower",
            "shape_mismatch",
            "envelope_distance_to_optimum",
        ),
    }
    if len({int(row["run_seed"]) for row in eligible}) < 2:
        return []
    rng = np.random.default_rng(reporting_seed)
    output: list[dict[str, object]] = []
    for name, features in models.items():
        y, predictions, seeds = _cross_validated_predictions(eligible, features)
        residual = y - predictions
        r2 = 1.0 - float(np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2))
        mae = float(np.mean(np.abs(residual)))
        unique = np.asarray(sorted(set(seeds)))
        boot_r2: list[float] = []
        boot_mae: list[float] = []
        for _ in range(500):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in chosen])
            sample_y = y[indices]
            sample_residual = residual[indices]
            denominator = float(np.sum((sample_y - np.mean(sample_y)) ** 2))
            boot_r2.append(
                1.0 - float(np.sum(sample_residual**2)) / denominator
                if denominator > 0.0
                else float("nan")
            )
            boot_mae.append(float(np.mean(np.abs(sample_residual))))
        output.append(
            {
                "model": name,
                "features": "|".join(features),
                "n_rows": len(y),
                "n_seeds": len(unique),
                "grouped_cv_r2": r2,
                "grouped_cv_mae": mae,
                "bootstrap_r2_lower": float(np.nanquantile(boot_r2, 0.025)),
                "bootstrap_r2_upper": float(np.nanquantile(boot_r2, 0.975)),
                "bootstrap_mae_lower": float(np.quantile(boot_mae, 0.025)),
                "bootstrap_mae_upper": float(np.quantile(boot_mae, 0.975)),
                "reporting_seed": reporting_seed,
            }
        )
    return output


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save(fig: Any, path: Path) -> None:
    plt = _load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _one_axis_plot(
    rows: Sequence[Mapping[str, Any]], plot_dir: Path
) -> None:
    plt = _load_pyplot()
    definitions = (
        ("axis_1_surrogate_scale", "surrogate_scale", r"Surrogate scale $c_f$"),
        ("axis_2_envelope_scale", "envelope_scale", r"Envelope scale $c_E$"),
        ("axis_2_envelope_shape", "envelope_center", r"Envelope center $m_E$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), constrained_layout=True)
    for column, (membership, x_field, label) in enumerate(definitions):
        selected = [row for row in rows if membership in str(row["memberships"]).split("|")]
        for mf, color in ((0.25, "tab:orange"), (0.5, "tab:blue")):
            center_rows = [row for row in selected if float(row["surrogate_center"]) == mf]
            x_values = sorted({float(row[x_field]) for row in center_rows})
            regrets = []
            lower = []
            upper = []
            certified = []
            realized = []
            for x in x_values:
                bucket = [row for row in center_rows if float(row[x_field]) == x]
                values = np.asarray([float(row["true_regret"]) for row in bucket])
                regrets.append(float(np.median(values)))
                lower.append(float(np.quantile(values, 0.05)))
                upper.append(float(np.quantile(values, 0.95)))
                certified.append(float(bucket[0]["certified_coverage_probability"]))
                realized.append(float(np.mean([row["coverage_status"] == "covered" for row in bucket])))
            axes[0, column].plot(x_values, regrets, marker="o", color=color, label=f"m_f={mf:g}")
            axes[0, column].fill_between(x_values, lower, upper, color=color, alpha=0.15)
            axes[1, column].plot(x_values, certified, marker="o", color=color, label=f"certified; m_f={mf:g}")
            axes[1, column].plot(x_values, realized, marker="x", linestyle="--", color=color, label=f"realized; m_f={mf:g}")
        axes[0, column].set_xlabel(label)
        axes[0, column].set_ylabel("Exact-LCB true regret")
        axes[1, column].set_xlabel(label)
        axes[1, column].set_ylabel("Coverage probability/rate")
        axes[1, column].set_ylim(-0.03, 1.03)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Independent surrogate and envelope sweeps")
    _save(fig, plot_dir / "one_at_a_time_sweeps.png")


def _optimizer_error_plot(best_rows: Sequence[Mapping[str, Any]], plot_dir: Path) -> None:
    plt = _load_pyplot()
    selected = [row for row in best_rows if "axis_3_optimizer" in str(row["memberships"]).split("|")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    for axis, estimator in zip(axes, ("finite_difference", "stein_difference")):
        for mf, color in ((0.25, "tab:orange"), (0.5, "tab:blue")):
            bucket = [row for row in selected if row["estimator"] == estimator and float(row["surrogate_center"]) == mf]
            axis.scatter(
                [max(float(row["optimization_error_lower"]), 1e-12) for row in bucket],
                [float(row["true_regret"]) for row in bucket],
                s=10,
                alpha=0.25,
                color=color,
                label=f"m_f={mf:g}",
            )
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_xlabel(r"Measured optimizer error $\varepsilon$")
        axis.set_title(estimator.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("True regret")
    axes[1].legend()
    fig.suptitle("Optimizer-error axis uses measured global gap, not iteration count")
    _save(fig, plot_dir / "optimizer_error_vs_regret.png")


def _decomposition_plot(best_rows: Sequence[Mapping[str, Any]], plot_dir: Path) -> None:
    plt = _load_pyplot()
    if len(best_rows) > 30000:
        stride = int(np.ceil(len(best_rows) / 30000))
        rows = best_rows[::stride]
    else:
        rows = best_rows
    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    eligible = [row for row in rows if bool(row["certificate_eligible"])]
    invalid = [row for row in rows if not bool(row["certificate_eligible"])]
    axis.scatter(
        [float(row["decomposition_rhs_upper"]) for row in invalid],
        [float(row["true_regret"]) for row in invalid],
        s=9,
        alpha=0.2,
        color="tab:red",
        label="realized invalid/undecided",
    )
    axis.scatter(
        [float(row["decomposition_rhs_upper"]) for row in eligible],
        [float(row["true_regret"]) for row in eligible],
        s=9,
        alpha=0.25,
        color="tab:blue",
        label="realized envelope certified",
    )
    finite = [
        max(float(row["decomposition_rhs_upper"]), float(row["true_regret"]))
        for row in rows
        if np.isfinite(float(row["decomposition_rhs_upper"]))
    ]
    limit = max(finite, default=1.0)
    axis.plot([0.0, limit], [0.0, limit], color="black", linestyle="--", label="R = T + epsilon")
    axis.set(xlabel=r"Certified RHS $T+\varepsilon_{upper}$", ylabel="True regret", title="Regret decomposition certificate check")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    _save(fig, plot_dir / "decomposition_bound_check.png")


def _factorial_plot(best_rows: Sequence[Mapping[str, Any]], plot_dir: Path) -> None:
    plt = _load_pyplot()
    requested = (10, 100, 500)
    available = sorted({int(row["step"]) for row in best_rows})
    steps = [step for step in requested if step in available] or available[-min(3, len(available)) :]
    estimators = ("finite_difference", "stein_difference")
    centers = (0.25, 0.5)
    fig, axes = plt.subplots(
        len(estimators) * len(centers),
        len(steps),
        figsize=(4.0 * len(steps), 3.4 * len(estimators) * len(centers)),
        squeeze=False,
        constrained_layout=True,
    )
    scales = sorted(
        {
            float(row["surrogate_scale"])
            for row in best_rows
            if "combined_factorial" in str(row["memberships"]).split("|")
            and float(row["surrogate_scale"]) > 0.0
        }
    )
    for row_index, (estimator, mf) in enumerate(
        (item for estimator in estimators for item in ((estimator, 0.25), (estimator, 0.5)))
    ):
        for column, step in enumerate(steps):
            axis = axes[row_index, column]
            matrix = np.full((len(scales), len(scales)), np.nan)
            annotations = np.full_like(matrix, np.nan)
            for i, ce in enumerate(scales):
                for j, cf in enumerate(scales):
                    bucket = [
                        row
                        for row in best_rows
                        if "combined_factorial" in str(row["memberships"]).split("|")
                        and row["estimator"] == estimator
                        and int(row["step"]) == step
                        and float(row["surrogate_center"]) == mf
                        and float(row["surrogate_scale"]) == cf
                        and float(row["envelope_scale"]) == ce
                    ]
                    if bucket:
                        matrix[i, j] = float(np.median([float(row["true_regret"]) for row in bucket]))
                        annotations[i, j] = float(np.median([float(row["optimization_error_lower"]) for row in bucket]))
            image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="magma")
            for i in range(len(scales)):
                for j in range(len(scales)):
                    if np.isfinite(annotations[i, j]):
                        axis.text(j, i, f"e={annotations[i, j]:.1g}", ha="center", va="center", fontsize=6, color="white")
            axis.set_xticks(range(len(scales)), [f"{value:g}" for value in scales])
            axis.set_yticks(range(len(scales)), [f"{value:g}" for value in scales])
            axis.set_xlabel("c_f")
            axis.set_ylabel("c_E")
            axis.set_title(f"{estimator}; m_f={mf:g}; step={step}")
            fig.colorbar(image, ax=axis, label="median true regret")
    fig.suptitle("Factorial regret; cells annotate median optimizer error")
    _save(fig, plot_dir / "factorial_tradeoff_heatmaps.png")


def _representative_plot(
    manifest: Any,
    results: Sequence[Any],
    condition_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
    plot_dir: Path,
) -> None:
    plt = _load_pyplot()
    by_seed = {result.run_seed: result for result in results}
    seeds = [seed for seed in manifest.spec.diagnostic_run_seeds if seed in by_seed]
    if not seeds:
        seeds = [results[0].run_seed]
    regimes = ((1.0, 1.0, "matched"), (2.0, 1.0, "under-corrected"), (1.0, 2.0, "over-corrected"))
    fig, axes = plt.subplots(len(seeds), 3, figsize=(13.5, 3.7 * len(seeds)), squeeze=False, constrained_layout=True)
    x = np.linspace(0.0, 1.0, 1001)
    quantile = analytic_uniform_certificate(manifest.spec.gp, manifest.spec.confidence).quantile
    for row_index, seed in enumerate(seeds):
        result = by_seed[seed]
        draw = FourierGPDraw(manifest.spec.gp, result.a_coefficients, result.b_coefficients)
        for column, (cf, ce, label) in enumerate(regimes):
            axis = axes[row_index, column]
            landscape = DecomposedGPLandscape(draw, manifest.spec.uncertainty, 0.5, cf, 0.5, ce, quantile)
            surrogate = DecomposedGPLandscape(draw, manifest.spec.uncertainty, 0.5, cf, 0.5, ce, quantile, target="surrogate")
            condition = next(
                row
                for row in condition_rows
                if int(row["run_seed"]) == seed
                and float(row["surrogate_center"]) == 0.5
                and float(row["envelope_center"]) == 0.5
                and float(row["surrogate_scale"]) == cf
                and float(row["envelope_scale"]) == ce
            )
            axis.plot(x, true_value(x), color="black", label="f")
            axis.plot(x, surrogate.evaluate(x), color="tab:orange", label="f-hat")
            axis.plot(x, landscape.evaluate(x), color="tab:blue", label="lower")
            axis.axvline(float(condition["global_x"]), color="tab:green", linestyle="--", label="global lower max")
            finals = [
                row
                for row in best_rows
                if int(row["run_seed"]) == seed
                and row["condition_id"] == condition["condition_id"]
                and row["estimator"] == "stein_difference"
                and int(row["step"]) == manifest.spec.optimizer.max_steps
            ]
            if finals:
                axis.scatter([float(finals[0]["x"])], [float(finals[0]["selected_lower_value"])], color="tab:red", marker="x", s=60, label="Stein final")
            axis.set_title(f"seed={seed}; {label}")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle("Predeclared analytic Fourier paths; curves are rendered evaluations", y=1.02)
    _save(fig, plot_dir / "representative_landscapes.png")


def write_decomposition_reports(
    manifest: Any,
    *,
    results: Sequence[Any],
    condition_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
    project_dir: Path,
) -> None:
    """Write aggregate tables, explanatory models, and five direct plots."""
    axis_rows = _membership_rows(
        condition_rows,
        {"axis_1_surrogate_scale", "axis_2_envelope_scale", "axis_2_envelope_shape"},
    )
    axis_summary = _aggregate(
        axis_rows,
        groups=("sweep", "surrogate_center", "surrogate_scale", "envelope_center", "envelope_scale"),
        numeric=("true_regret", "surrogate_error_lower", "envelope_term", "certified_coverage_probability"),
    )
    optimizer_summary = _aggregate(
        best_rows,
        groups=("surrogate_center", "surrogate_scale", "envelope_center", "envelope_scale", "estimator", "step"),
        numeric=("true_regret", "optimization_error_lower", "optimization_error_upper", "envelope_term", "decomposition_rhs_upper"),
    )
    coverage_rows = []
    for condition_id in sorted({str(row["condition_id"]) for row in condition_rows}):
        bucket = [row for row in condition_rows if row["condition_id"] == condition_id]
        coverage_rows.append(
            {
                "condition_id": condition_id,
                "n_seeds": len(bucket),
                "certified_coverage_probability": float(bucket[0]["certified_coverage_probability"]),
                "realized_covered_rate": float(np.mean([row["coverage_status"] == "covered" for row in bucket])),
                "realized_violated_rate": float(np.mean([row["coverage_status"] == "violated" for row in bucket])),
                "realized_undecided_rate": float(np.mean([row["coverage_status"] == "undecided" for row in bucket])),
            }
        )
    model_rows = _model_summary(best_rows, manifest.spec.reporting_seed)
    for filename, rows in (
        ("axis_summary.csv", axis_summary),
        ("optimizer_summary.csv", optimizer_summary),
        ("coverage_summary.csv", coverage_rows),
        ("explanatory_model_summary.csv", model_rows),
    ):
        if rows:
            write_rows_csv(project_dir / filename, rows, tuple(rows[0]))
    plot_dir = project_dir / "plots"
    _one_axis_plot(condition_rows, plot_dir)
    _optimizer_error_plot(best_rows, plot_dir)
    _decomposition_plot(best_rows, plot_dir)
    _factorial_plot(best_rows, plot_dir)
    _representative_plot(manifest, results, condition_rows, best_rows, plot_dir)


__all__ = ["write_decomposition_reports"]
