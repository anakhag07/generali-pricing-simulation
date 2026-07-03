"""Run a planted-logistic homoskedastic-noise sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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

BASE_PRESET = "planted_logistic_base"
PROJECT_NAME = "homoskedastic-theta-offset-sweep"
DISPLAY_KEYS: tuple[str, ...] = ()
NOISE_STD = 0.5

# Seed replication: each theta-offset variant is repeated across these run seeds so
# every plot gets error bars. The stein-difference estimator draws its perturbations
# from optimizer_seed, so vary="optimizer" here yields non-degenerate error bars.
# Non-varied streams stay pinned to ANCHOR_SEED (data/split/theta identical across
# replicates); FIXED_SEEDS keeps the same homoskedastic noise realization as before.
RUN_SEEDS: tuple[int, ...] = (7, 8, 9)
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
THETA_OFFSETS = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

_PLANTED_BASE = get_config(BASE_PRESET)


def _theta_offset_label(offset: float) -> str:
    return f"theta-offset-{float(offset):g}"


def _theta0(offset: float) -> np.ndarray:
    return BASE_THETA + float(offset)


def _noisy_objective() -> NoisyObjective:
    return NoisyObjective(
        base_objective=_PLANTED_BASE.objective,
        noise=HomoskedasticGaussianNoise(std=NOISE_STD),
    )


OVERRIDE_LIST = [
    {
        "_run_name": _theta_offset_label(offset),
        "objective": _noisy_objective(),
        "theta0": _theta0(offset),
        "enabled_estimators": ("finite_difference", "stein_difference"),
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
    for offset in THETA_OFFSETS
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured preset sweep.")
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


def _override_list_requires_jax() -> bool:
    return any(overrides.get("compute_backend") == "jax" for overrides in OVERRIDE_LIST)


def _task_specs() -> list[tuple[str, dict[str, Any], int]]:
    variants = expand_sweep_overrides(
        base_preset=BASE_PRESET,
        override_list=OVERRIDE_LIST,
        display_keys=DISPLAY_KEYS,
    )
    return [
        (variant_name, dict(overrides), int(seed))
        for variant_name, overrides in variants
        for seed in RUN_SEEDS
    ]


def _project_dir() -> Path:
    return results_root() / _path_part(PROJECT_NAME)


def _variant_dir(variant_name: str) -> Path:
    return _project_dir() / _path_part(variant_name)


def _run_sweep_task(index: int, context: LaunchContext) -> dict[str, object]:
    del context
    variant_name, overrides, seed = _task_specs()[index]
    variant_dir = _variant_dir(variant_name)
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
        "variant": variant_name,
        "run_seed": seed,
        "run_dir": str(executed.run_context.run_dir),
        "summary_json": str(variant_dir / f"summary-seed-{seed}.json"),
    }


def _run_sweep_serial(context: LaunchContext) -> None:
    del context
    sweep = run_sweep(
        base_preset=BASE_PRESET,
        run_seeds=RUN_SEEDS,
        override_list=OVERRIDE_LIST,
        vary=VARY,
        anchor_seed=ANCHOR_SEED,
        fixed=FIXED_SEEDS,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    print(
        f"Completed {len(sweep.run_results)} sweep runs "
        f"({len(OVERRIDE_LIST)} variants x {len(RUN_SEEDS)} seeds) for preset '{BASE_PRESET}'."
    )


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

    rows_by_variant: dict[str, list[dict[str, object]]] = {}
    for row in final_rows:
        rows_by_variant.setdefault(str(row["variant"]), []).append(row)

    for variant_name, rows in rows_by_variant.items():
        _write_seed_outputs_from_rows(_variant_dir(variant_name), rows)
    if len(rows_by_variant) > 1:
        _write_seed_outputs_from_rows(_project_dir(), final_rows)
    print(f"Collected {len(payloads)} sweep array tasks into {_project_dir()}.")


def _final_rows_from_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        summary_path = Path(str(payload["summary_json"]))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.extend(
            _summary_final_rows(
                summary,
                variant=str(payload["variant"]),
                run_seed=int(payload["run_seed"]),
                run_dir=str(payload["run_dir"]),
            )
        )
    return rows


def _summary_final_rows(
    summary: dict[str, Any],
    *,
    variant: str,
    run_seed: int,
    run_dir: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for estimator, estimator_payload in summary.get("estimators", {}).items():
        row: dict[str, object] = {
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
        name=PROJECT_NAME,
        task_count=len(_task_specs()),
        requires_jax=_override_list_requires_jax(),
        run_task=_run_sweep_task,
        run_all=_run_sweep_serial,
        collect=_collect_sweep_tasks,
        default_launch="auto",
        default_array=False,
    )


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
