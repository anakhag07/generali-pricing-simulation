"""Run noisy all-data GLM theta-distance and variance sweeps on GPU/JAX."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.execution import default_reporter_stack, execute_experiment_run
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan, task_payloads
from experiments.paths import results_root
from experiments.reporting.context import create_run_context
from experiments.reporting.json_summary import JsonReporter
from experiments.seeds import replicate_seed_setup
from objective.noise import HeteroskedasticGaussianNoise, HomoskedasticGaussianNoise, NoisyObjective


BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "noisy-glm-theta-variance-sweep"
DEFAULT_TRUTH_SUMMARY = Path(
    "/home/anakhag/projects/generali-pricing/results/real_data_glm_base__20260706_124627/summary.json"
)
REQUIRED_ESTIMATORS = ("finite_difference", "stein_difference")
RUN_SEEDS = (8, 9, 10)
VARY = ("optimizer", "noise")
THETA_DISTANCE_FRACTIONS = (0.0, 0.1, 0.35, 1.0)
NOISE_VARIANCES = (0.0, 0.25, 1.0, 4.0)
THETA_SWEEP_NOISE_VARIANCE = 1.0

GridName = Literal["homoskedastic", "heteroskedastic", "all"]
NoiseKind = Literal["homoskedastic", "heteroskedastic"]
AxisKind = Literal["theta_distance", "noise_variance"]

FINAL_FIELDNAMES = (
    "project",
    "variant",
    "run_seed",
    "estimator",
    "noise_kind",
    "axis",
    "axis_value",
    "theta_fraction",
    "theta_start_distance_to_truth",
    "theta_final_distance_to_truth",
    "theta_distance_improvement",
    "noise_variance",
    "noise_std",
    "noise_growth",
    "u_center",
    "final_value",
    "truth_final_value",
    "objective_gap_to_truth",
    "final_u",
    "truth_final_u",
    "mean_acceptance",
    "truth_mean_acceptance",
    "mean_acceptance_gap_to_truth",
    "runtime_sec",
    "optimizer_success",
    "optimizer_status",
    "summary_path",
    "run_dir",
)

SUMMARY_METRICS = (
    "theta_final_distance_to_truth",
    "theta_distance_improvement",
    "objective_gap_to_truth",
    "final_value",
    "mean_acceptance_gap_to_truth",
    "final_u",
    "runtime_sec",
)
SUMMARY_FIELDNAMES = (
    "project",
    "variant",
    "estimator",
    "noise_kind",
    "axis",
    "axis_value",
    "n_seeds",
    *(f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in ("mean", "std", "min", "max")),
)


@dataclass(frozen=True)
class TruthReference:
    summary_path: Path
    theta_initial: np.ndarray
    theta_truth: np.ndarray
    final_u: float
    final_value: float
    mean_acceptance: float | None
    run_seed: int
    preset_overrides: dict[str, Any]

    @property
    def initial_distance_to_truth(self) -> float:
        return float(np.linalg.norm(self.theta_initial - self.theta_truth))


@dataclass(frozen=True)
class SweepVariant:
    project_name: str
    name: str
    noise_kind: NoiseKind
    axis: AxisKind
    axis_value: float
    theta_fraction: float
    theta_start: np.ndarray
    theta_start_distance_to_truth: float
    noise_variance: float
    noise_std: float
    noise_growth: float
    u_center: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth-summary",
        default=str(DEFAULT_TRUTH_SUMMARY),
        help="All-data no-noise first-order summary.json used as theta/u truth.",
    )
    parser.add_argument(
        "--grids",
        choices=("all", "homoskedastic", "heteroskedastic"),
        default="all",
    )
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(RUN_SEEDS))
    parser.add_argument("--anchor-seed", type=int, default=None)
    parser.add_argument("--project-prefix", default="")
    parser.add_argument("--t-steps", type=int, default=None)
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


def _load_truth_reference(path: str | Path) -> TruthReference:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    first_order = summary["estimators"]["first_order"]
    preset = summary.get("preset", {})
    return TruthReference(
        summary_path=summary_path,
        theta_initial=np.asarray(summary["config"]["theta0"], dtype=float),
        theta_truth=np.asarray(first_order["theta"], dtype=float),
        final_u=float(first_order["final_u"]),
        final_value=float(first_order["final_value"]),
        mean_acceptance=_optional_float(first_order.get("mean_acceptance")),
        run_seed=int(summary["config"].get("seed", 8)),
        preset_overrides=dict(preset.get("overrides", {})),
    )


def _base_overrides(reference: TruthReference, args: argparse.Namespace) -> dict[str, Any]:
    overrides = dict(reference.preset_overrides)
    overrides.update(
        {
            "compute_backend": "jax",
            "constraint_mode": "trust_constr",
            "n_samples": None,
            "train_fraction": 1.0,
            "test_fraction": 0.0,
            "enabled_estimators": REQUIRED_ESTIMATORS,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        }
    )
    if args.t_steps is not None:
        overrides["t_steps"] = int(args.t_steps)
    return overrides


def _theta_start(reference: TruthReference, fraction: float) -> tuple[np.ndarray, float]:
    theta = reference.theta_truth + float(fraction) * (reference.theta_initial - reference.theta_truth)
    distance = float(np.linalg.norm(theta - reference.theta_truth))
    return theta, distance


def _noise_from_variant(variant: SweepVariant):
    if variant.noise_kind == "homoskedastic":
        return HomoskedasticGaussianNoise(std=variant.noise_std)
    return HeteroskedasticGaussianNoise(
        base_std=0.0,
        growth=variant.noise_growth,
        u_center=variant.u_center,
    )


def _variant_sets(reference: TruthReference, project_prefix: str = "") -> dict[str, tuple[SweepVariant, ...]]:
    prefix = f"{_path_part(project_prefix)}-" if project_prefix else ""
    return {
        "homoskedastic": (
            *_theta_variants(
                reference,
                project_name=f"{prefix}noisy-glm-homoskedastic-theta-distance-sweep",
                noise_kind="homoskedastic",
            ),
            *_variance_variants(
                reference,
                project_name=f"{prefix}noisy-glm-homoskedastic-variance-sweep",
                noise_kind="homoskedastic",
            ),
        ),
        "heteroskedastic": (
            *_theta_variants(
                reference,
                project_name=f"{prefix}noisy-glm-heteroskedastic-theta-distance-sweep",
                noise_kind="heteroskedastic",
            ),
            *_variance_variants(
                reference,
                project_name=f"{prefix}noisy-glm-heteroskedastic-variance-sweep",
                noise_kind="heteroskedastic",
            ),
        ),
    }


def _theta_variants(
    reference: TruthReference,
    *,
    project_name: str,
    noise_kind: NoiseKind,
) -> tuple[SweepVariant, ...]:
    variance = float(THETA_SWEEP_NOISE_VARIANCE)
    std_or_growth = float(np.sqrt(variance))
    variants: list[SweepVariant] = []
    for fraction in THETA_DISTANCE_FRACTIONS:
        theta, distance = _theta_start(reference, float(fraction))
        variants.append(
            SweepVariant(
                project_name=project_name,
                name=f"theta-frac-{_value_label(fraction)}",
                noise_kind=noise_kind,
                axis="theta_distance",
                axis_value=distance,
                theta_fraction=float(fraction),
                theta_start=theta,
                theta_start_distance_to_truth=distance,
                noise_variance=variance,
                noise_std=std_or_growth if noise_kind == "homoskedastic" else 0.0,
                noise_growth=std_or_growth if noise_kind == "heteroskedastic" else 0.0,
                u_center=reference.final_u,
            )
        )
    return tuple(variants)


def _variance_variants(
    reference: TruthReference,
    *,
    project_name: str,
    noise_kind: NoiseKind,
) -> tuple[SweepVariant, ...]:
    theta, distance = _theta_start(reference, 1.0)
    variants: list[SweepVariant] = []
    for variance in NOISE_VARIANCES:
        variance_value = float(variance)
        std_or_growth = float(np.sqrt(variance_value))
        variants.append(
            SweepVariant(
                project_name=project_name,
                name=f"variance-{_value_label(variance_value)}",
                noise_kind=noise_kind,
                axis="noise_variance",
                axis_value=variance_value,
                theta_fraction=1.0,
                theta_start=theta.copy(),
                theta_start_distance_to_truth=distance,
                noise_variance=variance_value,
                noise_std=std_or_growth if noise_kind == "homoskedastic" else 0.0,
                noise_growth=std_or_growth if noise_kind == "heteroskedastic" else 0.0,
                u_center=reference.final_u,
            )
        )
    return tuple(variants)


def _selected_variants(reference: TruthReference, args: argparse.Namespace) -> tuple[SweepVariant, ...]:
    variant_sets = _variant_sets(reference, project_prefix=str(args.project_prefix or ""))
    if args.grids == "all":
        return (*variant_sets["homoskedastic"], *variant_sets["heteroskedastic"])
    return variant_sets[args.grids]


def _task_specs(args: argparse.Namespace) -> list[tuple[SweepVariant, int]]:
    reference = _load_truth_reference(args.truth_summary)
    return [
        (variant, int(seed))
        for variant in _selected_variants(reference, args)
        for seed in args.run_seeds
    ]


def _config_for_variant(variant: SweepVariant, seed: int, reference: TruthReference, args: argparse.Namespace):
    anchor_seed = int(args.anchor_seed) if args.anchor_seed is not None else int(reference.run_seed)
    seed_setup = replicate_seed_setup(seed, anchor_seed, vary=VARY)
    overrides = {
        **_base_overrides(reference, args),
        "theta0": variant.theta_start.copy(),
        "seed_setup": seed_setup,
    }
    config = get_config(BASE_PRESET, overrides=overrides)
    return replace(
        config,
        objective=NoisyObjective(config.objective, _noise_from_variant(variant)),
        correctness=CorrectnessSpec(gradient_source="denoised_exact"),
    )


def _run_sweep_task(index: int, context: LaunchContext, *, args: argparse.Namespace) -> dict[str, object]:
    del context
    reference = _load_truth_reference(args.truth_summary)
    variant, seed = _task_specs(args)[index]
    variant_dir = _variant_dir(variant.project_name, variant.name)
    seed_summary = variant_dir / f"summary-seed-{seed}.json"
    payload = _task_payload(variant, seed, seed_summary)
    if _summary_has_estimators(seed_summary, REQUIRED_ESTIMATORS):
        print(f"Skipping completed task '{variant.name}' seed {seed} in '{variant.project_name}'.")
        return payload

    config = _config_for_variant(variant, seed, reference, args)
    run_context = create_run_context(
        variant.name,
        run_dir=variant_dir / "seeds" / f"seed-{seed}",
        run_metadata={
            "preset_name": BASE_PRESET,
            "variant_name": variant.name,
            "run_seed": seed,
            "truth_summary": str(reference.summary_path),
            "noise_kind": variant.noise_kind,
            "axis": variant.axis,
            "axis_value": variant.axis_value,
            "noise_variance": variant.noise_variance,
            "theta_start_distance_to_truth": variant.theta_start_distance_to_truth,
        },
    )
    executed = execute_experiment_run(
        variant.name,
        config,
        run_context=run_context,
        reporter_stack_factory=_seed_reporter_stack_factory(variant_dir, seed),
    )
    return {**payload, "run_dir": str(executed.run_context.run_dir)}


def _task_payload(variant: SweepVariant, seed: int, seed_summary: Path) -> dict[str, object]:
    return {
        "project": variant.project_name,
        "variant": variant.name,
        "run_seed": int(seed),
        "run_dir": str(_variant_dir(variant.project_name, variant.name) / "seeds" / f"seed-{seed}"),
        "summary_json": str(seed_summary),
        "noise_kind": variant.noise_kind,
        "axis": variant.axis,
        "axis_value": float(variant.axis_value),
        "theta_fraction": float(variant.theta_fraction),
        "theta_start_distance_to_truth": float(variant.theta_start_distance_to_truth),
        "noise_variance": float(variant.noise_variance),
        "noise_std": float(variant.noise_std),
        "noise_growth": float(variant.noise_growth),
        "u_center": float(variant.u_center),
    }


def _run_sweep_serial(context: LaunchContext, *, args: argparse.Namespace) -> None:
    payloads = [
        _run_sweep_task(index, context, args=args)
        for index in range(len(_task_specs(args)))
    ]
    _write_outputs_from_payloads(payloads, args)
    print(f"Completed {len(payloads)} noisy GLM sweep tasks.")


def _collect_sweep_tasks(context: LaunchContext, *, args: argparse.Namespace) -> None:
    payloads = task_payloads(context)
    _write_outputs_from_payloads(payloads, args)
    print(f"Collected {len(payloads)} noisy GLM array tasks.")


def _write_outputs_from_payloads(payloads: Sequence[Mapping[str, object]], args: argparse.Namespace) -> None:
    reference = _load_truth_reference(args.truth_summary)
    rows = _final_rows_from_payloads(payloads, reference)
    if not rows:
        raise ValueError("No noisy GLM final rows were produced.")
    for project in sorted({str(row["project"]) for row in rows}):
        project_rows = [row for row in rows if str(row["project"]) == project]
        project_dir = _project_dir(project)
        summary_rows = _aggregate_rows(project_rows)
        _write_rows(project_dir / "noisy_glm_theta_variance_finals.csv", project_rows, FINAL_FIELDNAMES)
        _write_rows(project_dir / "noisy_glm_theta_variance_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
        _write_project_plots(project_dir, project_rows)


def _final_rows_from_payloads(
    payloads: Sequence[Mapping[str, object]],
    reference: TruthReference,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        summary_path = Path(str(payload["summary_json"]))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.extend(_summary_final_rows(summary, payload, reference))
    return rows


def _summary_final_rows(
    summary: Mapping[str, Any],
    payload: Mapping[str, object],
    reference: TruthReference,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    theta_start_distance = float(payload["theta_start_distance_to_truth"])
    for estimator, estimator_payload in summary.get("estimators", {}).items():
        theta = np.asarray(estimator_payload.get("theta", []), dtype=float)
        theta_final_distance = float(np.linalg.norm(theta - reference.theta_truth)) if theta.size else float("nan")
        final_value = float(estimator_payload["final_value"])
        mean_acceptance = _optional_float(estimator_payload.get("mean_acceptance"))
        row = {
            "project": str(payload["project"]),
            "variant": str(payload["variant"]),
            "run_seed": int(payload["run_seed"]),
            "estimator": str(estimator),
            "noise_kind": str(payload["noise_kind"]),
            "axis": str(payload["axis"]),
            "axis_value": float(payload["axis_value"]),
            "theta_fraction": float(payload["theta_fraction"]),
            "theta_start_distance_to_truth": theta_start_distance,
            "theta_final_distance_to_truth": theta_final_distance,
            "theta_distance_improvement": theta_start_distance - theta_final_distance,
            "noise_variance": float(payload["noise_variance"]),
            "noise_std": float(payload["noise_std"]),
            "noise_growth": float(payload["noise_growth"]),
            "u_center": float(payload["u_center"]),
            "final_value": final_value,
            "truth_final_value": reference.final_value,
            "objective_gap_to_truth": final_value - reference.final_value,
            "final_u": float(estimator_payload["final_u"]),
            "truth_final_u": reference.final_u,
            "mean_acceptance": "" if mean_acceptance is None else mean_acceptance,
            "truth_mean_acceptance": "" if reference.mean_acceptance is None else reference.mean_acceptance,
            "mean_acceptance_gap_to_truth": _acceptance_gap(mean_acceptance, reference.mean_acceptance),
            "runtime_sec": float(estimator_payload["runtime_sec"]),
            "optimizer_success": estimator_payload.get("optimizer_success", ""),
            "optimizer_status": estimator_payload.get("optimizer_status", ""),
            "summary_path": str(payload["summary_json"]),
            "run_dir": str(payload["run_dir"]),
        }
        rows.append(row)
    return rows


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(row["project"]), str(row["variant"]), str(row["estimator"])) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for project, variant, estimator in keys:
        group = [
            row for row in rows
            if str(row["project"]) == project and str(row["variant"]) == variant and str(row["estimator"]) == estimator
        ]
        first = group[0]
        summary: dict[str, object] = {
            "project": project,
            "variant": variant,
            "estimator": estimator,
            "noise_kind": first["noise_kind"],
            "axis": first["axis"],
            "axis_value": first["axis_value"],
            "n_seeds": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group if _has_float(row.get(metric))], dtype=float)
            if values.size == 0:
                for stat in ("mean", "std", "min", "max"):
                    summary[f"{metric}_{stat}"] = ""
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(summary)
    return summary_rows


def _write_project_plots(project_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    plot_dir = project_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    axis = str(rows[0]["axis"])
    x_label = "Initial theta distance to first-order truth" if axis == "theta_distance" else "Noise variance"
    x_key = "theta_start_distance_to_truth" if axis == "theta_distance" else "noise_variance"
    _plot_metric_by_axis(
        rows,
        plot_dir / "theta_distance_to_truth.png",
        x_key=x_key,
        y_key="theta_final_distance_to_truth",
        x_label=x_label,
        y_label="Final theta distance to first-order truth",
    )
    _plot_metric_by_axis(
        rows,
        plot_dir / "objective_gap_to_truth.png",
        x_key=x_key,
        y_key="objective_gap_to_truth",
        x_label=x_label,
        y_label="Clean objective gap to first-order truth",
    )


def _plot_metric_by_axis(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.2))
    x_values = sorted({float(row[x_key]) for row in rows if _has_float(row.get(y_key))})
    for estimator in sorted({str(row["estimator"]) for row in rows}):
        selected = [row for row in rows if str(row["estimator"]) == estimator and _has_float(row.get(y_key))]
        if not selected:
            continue
        xs = sorted({float(row[x_key]) for row in selected})
        means = [_metric_mean(selected, x_key, y_key, x_value) for x_value in xs]
        stds = [_metric_std(selected, x_key, y_key, x_value) for x_value in xs]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=1.8, capsize=3.0, label=estimator)
    _configure_axis_scale(ax, x_values)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _metric_mean(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str, x_value: float) -> float:
    values = [float(row[y_key]) for row in rows if float(row[x_key]) == x_value and _has_float(row.get(y_key))]
    return float(np.mean(values))


def _metric_std(rows: Sequence[Mapping[str, object]], x_key: str, y_key: str, x_value: float) -> float:
    values = [float(row[y_key]) for row in rows if float(row[x_key]) == x_value and _has_float(row.get(y_key))]
    return float(np.std(values, ddof=0))


def _configure_axis_scale(ax: object, values: Sequence[float]) -> None:
    nonzero = [abs(value) for value in values if value != 0.0]
    if nonzero:
        ax.set_xscale("symlog", linthresh=min(nonzero))
    ax.set_xticks(list(values))
    ax.set_xticklabels([f"{value:g}" for value in values], rotation=45, ha="right")


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _seed_reporter_stack_factory(variant_dir: Path, seed: int):
    def factory(config):
        return default_reporter_stack(
            config,
            json_reporter=JsonReporter(
                summary_name=f"summary-seed-{seed}.json",
                summary_dir=variant_dir,
            ),
            include_plots=False,
        )

    return factory


def _summary_has_estimators(summary_path: Path, estimators: Sequence[str]) -> bool:
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    estimator_payload = payload.get("estimators", {})
    return all(name in estimator_payload for name in estimators)


def _project_dir(project_name: str) -> Path:
    return results_root() / _path_part(project_name)


def _variant_dir(project_name: str, variant_name: str) -> Path:
    return _project_dir(project_name) / _path_part(variant_name)


def _value_label(value: object) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-")


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _acceptance_gap(value: float | None, truth: float | None) -> float | str:
    if value is None or truth is None:
        return ""
    return float(value - truth)


def _has_float(value: object) -> bool:
    if value is None or value == "":
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(out))


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    return LaunchPlan(
        name=PROJECT_NAME,
        task_count=len(_task_specs(args)),
        requires_jax=True,
        run_task=lambda index, context: _run_sweep_task(index, context, args=args),
        run_all=lambda context: _run_sweep_serial(context, args=args),
        collect=lambda context: _collect_sweep_tasks(context, args=args),
        default_launch="auto",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
