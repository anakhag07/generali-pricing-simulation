"""Run dense planted-logistic homoskedastic-noise sweeps.

Two dense fill-in grids (theta-offset and noise-std) over the planted-logistic
homoskedastic-noise objective, driven through the shared ``LaunchPlan`` launch
framework. Serial runs skip already-complete variant folders and, after the runs
finish, both the serial path and the array collector regenerate the
theta-distance-to-first-order-truth CSV/PNG plots for each grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.execution import default_reporter_stack, execute_experiment_run
from experiments.launch import (
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
    task_payloads,
)
from experiments.paths import results_root
from experiments.reporting.context import create_run_context
from experiments.reporting.json_summary import JsonReporter
from experiments.seeds import replicate_seed_setup
from experiments.sweep_reporting import (
    DEFAULT_SEED_METRIC_BARS,
    aggregate_seed_grid_rows,
    write_seed_grid_csvs,
)
from experiments.sweep_utils import expand_sweep_overrides, run_sweep
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective
from reporting.visualization import plot_seed_grid_frontier, plot_seed_grid_metric_bars


# =============================================================================
# Experiment-specific sweep definitions
# Everything in this section may mention planted-logistic objectives, noise,
# concrete estimator choices, fixed seeds, theta offsets, or output project names.
# =============================================================================

BASE_PRESET = "planted_logistic_base"
LAUNCH_PLAN_NAME = "homoskedastic-fill-in-sweeps"
THETA_PROJECT_NAME = "homoskedastic-theta-offset-sweep"
NOISE_PROJECT_NAME = "homoskedastic-noise-sweep"
# Backward-compatible aliases used by tests and older ad-hoc imports.
PROJECT_NAME = THETA_PROJECT_NAME
DISPLAY_KEYS: tuple[str, ...] = ()
REQUIRED_ESTIMATORS = ("finite_difference", "stein_difference")
NOISE_STD = 0.5

# These dense fill-in sweeps match the existing saved single-seed runs. Existing
# completed variant folders are skipped before dispatching to run_sweep().
RUN_SEEDS: tuple[int, ...] = (7,)
ANCHOR_SEED = 7
VARY: tuple[str, ...] = ("optimizer",)
FIXED_SEEDS: dict[str, int | None] = {"noise": 101}
BASE_THETA = np.asarray(
    [
        0.4054882808450241,
        0.00012799868045781167,
        -4.657524122982136e-05,
        6.221922280809605e-05,
    ],
    dtype=float,
)
THETA_OFFSETS = (
    0.0,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
)
NOISE_STDS = (
    0.0,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
)

_PLANTED_BASE = get_config(BASE_PRESET)


def _first_order_truth_summary() -> Path:
    """Path to the saved first-order truth run used as the distance reference."""
    return (
        results_root()
        / "planted_logistic_base"
        / "first_order_truth_20260701_174139"
        / "summary.json"
    )


def _theta0(offset: float) -> np.ndarray:
    return BASE_THETA + float(offset)


def _noisy_objective(noise_std: float) -> NoisyObjective:
    return NoisyObjective(
        base_objective=_PLANTED_BASE.objective,
        noise=HomoskedasticGaussianNoise(std=float(noise_std)),
    )


COMMON_OVERRIDES: dict[str, object] = {
    "enabled_estimators": REQUIRED_ESTIMATORS,
    "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
    "perturbation_space": "u",
    "step_rule": "l-bfgs-b",
    "t_steps": 1000,
    "step_size": 0.001,
    "n_samples": 1000,
    "sigma": 0.05,
    "n_grad_samples": 8,
    "plot": True,
    "verbose": False,
    "wandb_enabled": False,
}


def _build_theta_override_list() -> list[dict[str, object]]:
    return [
        {
            "_run_name": _axis_run_name("theta-offset", offset),
            **COMMON_OVERRIDES,
            "objective": _noisy_objective(NOISE_STD),
            "theta0": _theta0(offset),
        }
        for offset in THETA_OFFSETS
    ]


def _build_noise_override_list() -> list[dict[str, object]]:
    return [
        {
            "_run_name": _axis_run_name("noise-std", noise_std),
            **COMMON_OVERRIDES,
            "objective": _noisy_objective(noise_std),
            "theta0": np.zeros_like(BASE_THETA),
        }
        for noise_std in NOISE_STDS
    ]


# =============================================================================
# Reusable sweep-script helpers
# Helpers in this section must receive run-specific values as parameters. If a
# helper needs BASE_THETA, NOISE_STD, THETA_OFFSETS, NOISE_STDS, NoisyObjective,
# or a concrete estimator list, keep it in the experiment-specific section.
# =============================================================================


def _axis_run_name(axis: str, value: object) -> str:
    return f"{axis}-{_format_sweep_value(value)}"


def _format_sweep_value(value: object) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        text = str(value)
        return text.replace(" ", "").replace("/", "-")


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-")


def _project_dir(project_name: str) -> Path:
    return results_root() / _path_part(project_name)


def _variant_dir(project_name: str, variant_name: str) -> Path:
    return _project_dir(project_name) / _path_part(variant_name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured preset sweeps.")
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


def _sweeps_require_jax() -> bool:
    return any(
        overrides.get("compute_backend") == "jax"
        for _, override_list in SWEEPS
        for overrides in override_list
    )


def _missing_overrides(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    required_estimators: Sequence[str],
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    project_dir = _project_dir(project_name)
    for overrides in override_list:
        run_name = overrides.get("_run_name")
        if run_name is None:
            raise ValueError("Resume/skipping requires each override to include '_run_name'.")
        if not _variant_is_completed(project_dir / _path_part(run_name), required_estimators):
            missing.append(dict(overrides))
    return missing


def _variant_is_completed(variant_dir: Path, required_estimators: Sequence[str]) -> bool:
    if not variant_dir.is_dir():
        return False
    for summary_path in _summary_paths(variant_dir):
        if _summary_has_estimators(summary_path, required_estimators):
            return True
    return False


def _summary_paths(variant_dir: Path) -> list[Path]:
    paths = sorted(variant_dir.glob("summary-seed-*.json"))
    direct_summary = variant_dir / "summary.json"
    if direct_summary.exists():
        paths.append(direct_summary)
    paths.extend(sorted(variant_dir.glob("seeds/seed-*/summary.json")))
    paths.extend(sorted(variant_dir.glob("*/summary.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _summary_has_estimators(summary_path: Path, estimators: Sequence[str]) -> bool:
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    estimator_payload = payload.get("estimators", {})
    return all(name in estimator_payload for name in estimators)


def _run_missing_sweep(
    *,
    base_preset: str,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    run_seeds: Sequence[int],
    vary: tuple[str, ...],
    anchor_seed: int,
    fixed: Mapping[str, int | None],
    display_keys: Sequence[str],
    required_estimators: Sequence[str],
) -> int:
    missing = _missing_overrides(
        project_name=project_name,
        override_list=override_list,
        required_estimators=required_estimators,
    )
    skipped = len(override_list) - len(missing)
    if not missing:
        print(f"No missing variants for '{project_name}' ({skipped} already complete).")
        return 0

    sweep = run_sweep(
        base_preset=base_preset,
        run_seeds=run_seeds,
        override_list=missing,
        vary=vary,
        anchor_seed=anchor_seed,
        fixed=fixed,
        project_name=project_name,
        display_keys=display_keys,
    )
    print(
        f"Completed {len(sweep.run_results)} missing runs for '{project_name}' "
        f"({len(missing)} variants x {len(run_seeds)} seeds; skipped {skipped})."
    )
    return len(sweep.run_results)


# =============================================================================
# Concrete sweep instances
# This section binds the experiment-specific definitions to the reusable helpers.
# =============================================================================


THETA_OVERRIDE_LIST = _build_theta_override_list()
NOISE_OVERRIDE_LIST = _build_noise_override_list()
OVERRIDE_LIST = THETA_OVERRIDE_LIST
SWEEPS: tuple[tuple[str, list[dict[str, object]]], ...] = (
    (THETA_PROJECT_NAME, THETA_OVERRIDE_LIST),
    (NOISE_PROJECT_NAME, NOISE_OVERRIDE_LIST),
)


# =============================================================================
# Launch wiring
# Expands the two grids into launch tasks and delegates local/Slurm orchestration
# to the shared LaunchPlan framework. Array tasks run one (grid, variant, seed);
# the serial path fills in missing variants and the collector aggregates records.
# =============================================================================


def _task_specs() -> list[tuple[str, str, dict[str, Any], int]]:
    specs: list[tuple[str, str, dict[str, Any], int]] = []
    for project_name, override_list in SWEEPS:
        variants = expand_sweep_overrides(
            base_preset=BASE_PRESET,
            override_list=override_list,
            display_keys=DISPLAY_KEYS,
        )
        for variant_name, overrides in variants:
            for seed in RUN_SEEDS:
                specs.append((project_name, variant_name, dict(overrides), int(seed)))
    return specs


def _run_sweep_task(index: int, context: LaunchContext) -> dict[str, object]:
    del context
    project_name, variant_name, overrides, seed = _task_specs()[index]
    variant_dir = _variant_dir(project_name, variant_name)
    seed_setup = replicate_seed_setup(
        seed,
        ANCHOR_SEED,
        vary=VARY,
        fixed=FIXED_SEEDS,
    )
    merged_overrides = {**overrides, "seed_setup": seed_setup}
    config = get_config(BASE_PRESET, overrides=merged_overrides)
    run_context = create_run_context(
        variant_name,
        run_dir=variant_dir / "seeds" / f"seed-{seed}",
    )
    executed = execute_experiment_run(
        variant_name,
        config,
        run_context=run_context,
        reporter_stack_factory=_seed_reporter_stack_factory(variant_dir, seed),
    )
    return {
        "project": project_name,
        "variant": variant_name,
        "run_seed": seed,
        "run_dir": str(executed.run_context.run_dir),
        "summary_json": str(variant_dir / f"summary-seed-{seed}.json"),
    }


def _run_sweep_serial(context: LaunchContext) -> None:
    del context
    n_runs = 0
    for project_name, override_list in SWEEPS:
        n_runs += _run_missing_sweep(
            base_preset=BASE_PRESET,
            project_name=project_name,
            override_list=override_list,
            run_seeds=RUN_SEEDS,
            vary=VARY,
            anchor_seed=ANCHOR_SEED,
            fixed=FIXED_SEEDS,
            display_keys=DISPLAY_KEYS,
            required_estimators=REQUIRED_ESTIMATORS,
        )
    _regenerate_distance_plots()
    print(f"Completed {n_runs} total missing sweep runs for preset '{BASE_PRESET}'.")


def _collect_sweep_tasks(context: LaunchContext) -> None:
    records = read_task_records(context)
    if len(records) != len(_task_specs()):
        raise RuntimeError(
            f"Expected {len(_task_specs())} task records under {context.tasks_dir}, "
            f"found {len(records)}."
        )
    payloads = task_payloads(context)
    final_rows = _final_rows_from_payloads(payloads)
    if not final_rows:
        raise ValueError("No sweep task final rows were produced.")

    rows_by_variant: dict[tuple[str, str], list[dict[str, object]]] = {}
    rows_by_project: dict[str, list[dict[str, object]]] = {}
    for row in final_rows:
        project = str(row["project"])
        variant = str(row["variant"])
        rows_by_variant.setdefault((project, variant), []).append(row)
        rows_by_project.setdefault(project, []).append(row)

    for (project, variant), rows in rows_by_variant.items():
        _write_seed_outputs_from_rows(_variant_dir(project, variant), rows)
    for project, rows in rows_by_project.items():
        _write_seed_outputs_from_rows(_project_dir(project), rows)
    _regenerate_distance_plots()
    print(f"Collected {len(payloads)} sweep array tasks under {results_root()}.")


def _final_rows_from_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        summary_path = Path(str(payload["summary_json"]))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.extend(
            _summary_final_rows(
                summary,
                project=str(payload["project"]),
                variant=str(payload["variant"]),
                run_seed=int(payload["run_seed"]),
                run_dir=str(payload["run_dir"]),
            )
        )
    return rows


def _summary_final_rows(
    summary: dict[str, Any],
    *,
    project: str,
    variant: str,
    run_seed: int,
    run_dir: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for estimator, estimator_payload in summary.get("estimators", {}).items():
        row: dict[str, object] = {
            "project": project,
            "variant": variant,
            "run_seed": int(run_seed),
            "run_dir": run_dir,
            "estimator": estimator,
            "final_u": estimator_payload.get("final_u", ""),
            "final_value": estimator_payload.get("final_value", ""),
            "runtime_sec": estimator_payload.get("runtime_sec", ""),
            "mean_acceptance": estimator_payload.get("mean_acceptance", ""),
            "constraint_violation": estimator_payload.get("constraint_violation", ""),
        }
        row.update(_evaluation_fields("train", estimator_payload.get("train")))
        row.update(_evaluation_fields("test", estimator_payload.get("test")))
        rows.append(row)
    return rows


def _evaluation_fields(prefix: str, evaluation: dict[str, Any] | None) -> dict[str, object]:
    if not evaluation:
        return {
            f"{prefix}_objective_value": "",
            f"{prefix}_objective_sum": "",
            f"{prefix}_mean_u": "",
            f"{prefix}_mean_acceptance": "",
        }
    return {
        f"{prefix}_objective_value": evaluation.get("objective_value", ""),
        f"{prefix}_objective_sum": evaluation.get("objective_sum", ""),
        f"{prefix}_mean_u": evaluation.get("mean_u", ""),
        f"{prefix}_mean_acceptance": evaluation.get("mean_acceptance", ""),
    }


def _write_seed_outputs_from_rows(output_dir: Path, final_rows: list[dict[str, object]]) -> None:
    summary_rows = aggregate_seed_grid_rows(final_rows)
    write_seed_grid_csvs(output_dir, final_rows, summary_rows)
    if not summary_rows:
        return
    plot_dir = str(output_dir / "plots")
    for metric, y_label, filename in DEFAULT_SEED_METRIC_BARS:
        plot_seed_grid_metric_bars(summary_rows, plot_dir, metric=metric, y_label=y_label, filename=filename)
    plot_seed_grid_frontier(summary_rows, plot_dir)


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


def _build_launch_plan() -> LaunchPlan:
    return LaunchPlan(
        name=LAUNCH_PLAN_NAME,
        task_count=len(_task_specs()),
        requires_jax=_sweeps_require_jax(),
        run_task=_run_sweep_task,
        run_all=_run_sweep_serial,
        collect=_collect_sweep_tasks,
        default_launch="auto",
        default_array=False,
    )


# =============================================================================
# Experiment-specific distance-reporting helpers
# These helpers are intentionally tied to the two dense homoskedastic sweeps.
# =============================================================================


def _regenerate_distance_plots() -> None:
    truth_summary = _first_order_truth_summary()
    if not truth_summary.exists():
        print(f"Skipping distance plots; missing truth summary: {truth_summary}")
        return
    truth_theta = _theta_from_summary(truth_summary, "first_order")
    _write_distance_plot(
        project_name=THETA_PROJECT_NAME,
        truth_theta=truth_theta,
        axis_key="theta_offset",
        x_label="Theta offset added to first-order truth theta",
        title="Final theta distance to first-order truth by offset",
        csv_name="theta_distance_to_first_order_truth_by_offset.csv",
        plot_name="theta_distance_to_first_order_truth_by_offset.png",
    )
    _write_distance_plot(
        project_name=NOISE_PROJECT_NAME,
        truth_theta=truth_theta,
        axis_key="noise_std",
        x_label="Homoskedastic noise std",
        title="Final theta distance to first-order truth by noise",
        csv_name="theta_distance_to_first_order_truth_by_noise.csv",
        plot_name="theta_distance_to_first_order_truth_by_noise.png",
    )


def _write_distance_plot(
    *,
    project_name: str,
    truth_theta: np.ndarray,
    axis_key: str,
    x_label: str,
    title: str,
    csv_name: str,
    plot_name: str,
) -> None:
    project_dir = _project_dir(project_name)
    rows = _collect_distance_rows(project_dir, truth_theta, axis_key)
    if not rows:
        print(f"Skipping distance plot for '{project_name}'; no summary rows found.")
        return
    _write_distance_csv(project_dir / csv_name, rows, axis_key)
    _plot_distance_rows(project_dir / plot_name, rows, axis_key, x_label, title)
    print(f"Wrote distance plot for '{project_name}' to {project_dir / plot_name}.")


def _collect_distance_rows(
    project_dir: Path,
    truth_theta: np.ndarray,
    axis_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not project_dir.is_dir():
        return rows
    for variant_dir in sorted(project_dir.iterdir(), key=_variant_sort_key):
        if not variant_dir.is_dir():
            continue
        for summary_path in _summary_paths(variant_dir):
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            axis_value = _distance_axis_value(axis_key, variant_dir.name, summary)
            if axis_value is None:
                continue
            for estimator in REQUIRED_ESTIMATORS:
                estimator_payload = summary.get("estimators", {}).get(estimator)
                if estimator_payload is None or "theta" not in estimator_payload:
                    continue
                theta = np.asarray(estimator_payload["theta"], dtype=float)
                rows.append(
                    {
                        axis_key: axis_value,
                        "estimator": estimator,
                        "distance_l2_to_truth": float(np.linalg.norm(theta - truth_theta)),
                        "optimizer_success": estimator_payload.get("optimizer_success", ""),
                        "final_value": float(estimator_payload["final_value"]),
                        "summary_path": str(summary_path),
                    }
                )
    return rows


def _distance_axis_value(
    axis_key: str,
    variant_name: str,
    summary: dict,
) -> float | None:
    if axis_key == "theta_offset" and variant_name.startswith("theta-offset-"):
        return float(variant_name.removeprefix("theta-offset-"))
    if axis_key == "noise_std":
        objective = summary.get("config", {}).get("objective", {})
        noise = objective.get("noise", {}) if isinstance(objective, dict) else {}
        if "std" in noise:
            return float(noise["std"])
        if variant_name.startswith("noise-std-"):
            return float(variant_name.removeprefix("noise-std-"))
    return None


def _theta_from_summary(summary_path: Path, estimator: str) -> np.ndarray:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return np.asarray(summary["estimators"][estimator]["theta"], dtype=float)


def _write_distance_csv(path: Path, rows: list[dict[str, object]], axis_key: str) -> None:
    fieldnames = [
        axis_key,
        "estimator",
        "distance_l2_to_truth",
        "optimizer_success",
        "final_value",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (float(item[axis_key]), str(item["estimator"]))):
            writer.writerow(row)


def _plot_distance_rows(
    path: Path,
    rows: list[dict[str, object]],
    axis_key: str,
    x_label: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "finite_difference": {"label": "Finite difference", "color": "tab:blue", "marker": "o"},
        "stein_difference": {"label": "Stein difference", "color": "tab:orange", "marker": "s"},
    }
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.6))
    x_values = sorted({float(row[axis_key]) for row in rows})
    for estimator in REQUIRED_ESTIMATORS:
        selected = [row for row in rows if row["estimator"] == estimator]
        if not selected:
            continue
        xs = sorted({float(row[axis_key]) for row in selected})
        means = [_distance_mean(selected, axis_key, x_value) for x_value in xs]
        stds = [_distance_std(selected, axis_key, x_value) for x_value in xs]
        style = styles[estimator]
        ax.errorbar(
            xs,
            means,
            yerr=stds if any(std > 0.0 for std in stds) else None,
            label=str(style["label"]),
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=1.8,
            markersize=5.5,
            capsize=3.0,
        )
        failed_xs = [x_value for x_value in xs if _distance_has_failure(selected, axis_key, x_value)]
        if failed_xs:
            ax.scatter(
                failed_xs,
                [_distance_mean(selected, axis_key, x_value) for x_value in failed_xs],
                label=f"{style['label']} optimizer_success=False",
                color=str(style["color"]),
                marker="x",
                s=60,
                linewidths=1.5,
                zorder=4,
            )
    _set_symlog_ticks(ax, x_values)
    distances = [float(row["distance_l2_to_truth"]) for row in rows]
    if all(distance > 0.0 for distance in distances):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$||\theta_{estimator} - \theta_{first\ order\ truth}||_2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _distance_mean(rows: list[dict[str, object]], axis_key: str, x_value: float) -> float:
    values = [float(row["distance_l2_to_truth"]) for row in rows if float(row[axis_key]) == x_value]
    return float(np.mean(values))


def _distance_std(rows: list[dict[str, object]], axis_key: str, x_value: float) -> float:
    values = [float(row["distance_l2_to_truth"]) for row in rows if float(row[axis_key]) == x_value]
    return float(np.std(values, ddof=0))


def _distance_has_failure(rows: list[dict[str, object]], axis_key: str, x_value: float) -> bool:
    return any(
        float(row[axis_key]) == x_value and row.get("optimizer_success") is False
        for row in rows
    )


def _set_symlog_ticks(ax: object, values: list[float]) -> None:
    nonzero = [abs(value) for value in values if value != 0.0]
    if nonzero:
        ax.set_xscale("symlog", linthresh=min(nonzero))
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:g}" for value in values], rotation=45, ha="right")


def _variant_sort_key(path: Path) -> tuple[int, float | str]:
    for prefix in ("theta-offset-", "noise-std-"):
        if path.name.startswith(prefix):
            return (0, float(path.name.removeprefix(prefix)))
    return (1, path.name)


# =============================================================================
# Entry point
# This section wires the concrete sweeps into the reusable launch helpers.
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
