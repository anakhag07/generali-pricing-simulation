"""Utilities for running preset-based parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.config import ExperimentConfig
from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.reporting.context import RunContext, create_run_context
from experiments.results import ExperimentResult
from experiments.seeds import SeedStream, replicate_seed_setup

_RUN_NAME_KEY = "_run_name"


@dataclass(frozen=True)
class SweepRunResult:
    """Completed sweep variant with config, overrides, result, and output context.

    ``run_name`` identifies the variant (shared across seed replicates); ``run_seed``
    is the replicate seed (``None`` for single-seed sweeps).
    """

    run_name: str
    config: ExperimentConfig
    overrides: dict[str, Any]
    result: ExperimentResult
    run_context: RunContext
    run_seed: int | None = None


@dataclass(frozen=True)
class SweepResult:
    """Aggregate outcome of a canonical seed-aware sweep.

    ``run_results`` holds one record per (variant, seed); ``summary_rows`` are the
    cross-seed aggregate rows (per variant x estimator mean/std) written alongside
    ``project_dir``.
    """

    project_dir: Path
    run_results: list[SweepRunResult]
    summary_rows: list[dict[str, Any]]


def expand_override_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Build cartesian-product override dictionaries from a field-value grid."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    value_lists = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def apply_config_overrides(config: ExperimentConfig, overrides: Mapping[str, Any]) -> ExperimentConfig:
    """Return a config copy with top-level ExperimentConfig fields overridden."""
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(key for key in overrides.keys() if key not in valid_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown config override fields: {unknown_text}.")
    return replace(config, **dict(overrides))


def make_sweep_name(base_name: str, index: int, overrides: Mapping[str, Any]) -> str:
    """Build a readable, deterministic run name for one sweep variant."""
    if not overrides:
        return f"{base_name}__sweep_{index:03d}"
    parts = [f"{key}-{_stringify_override_value(overrides[key])}" for key in sorted(overrides.keys())]
    suffix = "__".join(parts)
    return f"{base_name}__sweep_{index:03d}__{suffix}"


def make_display_name(
    base_name: str,
    index: int,
    overrides: Mapping[str, Any],
    *,
    display_keys: Sequence[str] | None = None,
) -> str:
    """Build a compact display name using only selected override keys."""
    if display_keys is None:
        keys = sorted(overrides.keys())
    else:
        keys = [key for key in display_keys if key in overrides]
    if not keys:
        return f"{base_name}__run_{index:03d}"
    parts = [f"{_display_key_label(key)}-{_stringify_override_value(overrides[key])}" for key in keys]
    return "__".join(parts)


def expand_sweep_overrides(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    display_keys: Sequence[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Expand a base preset into ``(variant_name, override_dict)`` pairs.

    Pure name/override resolution with no ``get_config`` call, so callers can merge
    additional overrides (e.g. a per-seed ``seed_setup``) before the config is built.
    """
    if override_grid is not None and override_list is not None:
        raise ValueError("Specify either override_grid or override_list, not both.")

    if override_list is not None:
        overrides = [dict(item) for item in override_list] if override_list else [{}]
    elif override_grid is not None:
        overrides = expand_override_grid(override_grid)
    else:
        overrides = [{}]

    variants: list[tuple[str, dict[str, Any]]] = []
    for index, override in enumerate(overrides, start=1):
        override_payload = dict(override)
        explicit_run_name = override_payload.pop(_RUN_NAME_KEY, None)
        run_name = (
            str(explicit_run_name)
            if explicit_run_name is not None
            else make_display_name(
                base_preset,
                index=index,
                overrides=override_payload,
                display_keys=display_keys,
            )
        )
        variants.append((run_name, override_payload))
    return variants


def generate_sweep_runs(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    display_keys: Sequence[str] | None = None,
) -> list[tuple[str, ExperimentConfig, dict[str, Any]]]:
    """Generate named configs by applying overrides to a base preset."""
    runs: list[tuple[str, ExperimentConfig, dict[str, Any]]] = []
    for run_name, override_payload in expand_sweep_overrides(
        base_preset=base_preset,
        override_grid=override_grid,
        override_list=override_list,
        display_keys=display_keys,
    ):
        config = get_config(base_preset, overrides=override_payload)
        runs.append((run_name, config, override_payload))
    return runs


def run_preset_sweep(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    runs_root: str = "outputs",
    project_name: str | None = None,
    display_keys: Sequence[str] | None = None,
) -> list[SweepRunResult]:
    """Execute a preset sweep and return rich per-run result records."""
    sweep_runs = generate_sweep_runs(
        base_preset=base_preset,
        override_grid=override_grid,
        override_list=override_list,
        display_keys=display_keys,
    )
    results: list[SweepRunResult] = []
    runs_root_path = _project_runs_root(runs_root, project_name)

    for run_name, config, overrides in sweep_runs:
        executed = execute_experiment_run(run_name, config, runs_root=runs_root_path)
        results.append(
            SweepRunResult(
                run_name=run_name,
                config=config,
                overrides=overrides,
                result=executed.result,
                run_context=executed.run_context,
            )
        )

    return results


def run_sweep(
    *,
    base_preset: str,
    run_seeds: Sequence[int],
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    vary: tuple[SeedStream, ...] = ("theta",),
    anchor_seed: int | None = None,
    fixed: Mapping[str, int | None] | None = None,
    per_seed_plots: bool = False,
    runs_root: str = "outputs",
    project_name: str | None = None,
    display_keys: Sequence[str] | None = None,
) -> SweepResult:
    """Run a canonical seed-aware sweep: every variant is replicated across seeds.

    Each ``(variant, run_seed)`` pair runs with a ``replicate_seed_setup`` that keeps
    non-``vary`` streams pinned to ``anchor_seed`` (default ``run_seeds[0]``), so by
    default data/split/noise stay identical and only ``theta`` init changes. Per-seed
    runs share one variant folder (``summary-seed-<seed>.json`` at the variant root,
    heavy artifacts under ``seeds/seed-<seed>/``); aggregate error-bar plots and
    ``seed_grid_summary.csv`` are written per variant and, for multi-variant sweeps,
    across variants at the project root. A plain seed sweep is the no-axis case.
    """
    from experiments.sweep_reporting import (
        objective_traces_by_estimator,
        write_seed_grid_outputs,
    )
    from reporting.visualization import plot_seed_loss_bands

    if not run_seeds:
        raise ValueError("run_seeds must contain at least one seed.")
    seeds = tuple(int(seed) for seed in run_seeds)
    anchor = int(anchor_seed) if anchor_seed is not None else seeds[0]

    variants = expand_sweep_overrides(
        base_preset=base_preset,
        override_grid=override_grid,
        override_list=override_list,
        display_keys=display_keys,
    )
    project_dir = Path(_project_runs_root(runs_root, project_name))
    all_run_results: list[SweepRunResult] = []

    for variant_name, overrides in variants:
        variant_dir = project_dir / _stringify_override_value(variant_name)
        variant_results: list[SweepRunResult] = []
        for seed in seeds:
            seed_setup = replicate_seed_setup(seed, anchor, vary=vary, fixed=fixed)
            merged_overrides = {**overrides, "seed_setup": seed_setup}
            config = get_config(base_preset, overrides=merged_overrides)
            run_context = create_run_context(
                variant_name, run_dir=variant_dir / "seeds" / f"seed-{seed}"
            )
            executed = execute_experiment_run(
                variant_name,
                config,
                run_context=run_context,
                reporter_stack_factory=_seed_reporter_stack_factory(
                    variant_dir, seed, per_seed_plots=per_seed_plots
                ),
            )
            record = SweepRunResult(
                run_name=variant_name,
                config=config,
                overrides=dict(overrides),
                result=executed.result,
                run_context=executed.run_context,
                run_seed=seed,
            )
            variant_results.append(record)
            all_run_results.append(record)

        write_seed_grid_outputs(variant_dir, variant_results)
        plot_seed_loss_bands(
            objective_traces_by_estimator(variant_results),
            str(variant_dir / "plots"),
        )

    if len(variants) > 1:
        summary_rows = write_seed_grid_outputs(project_dir, all_run_results)
    else:
        from experiments.sweep_reporting import (
            aggregate_seed_grid_rows,
            collect_seed_grid_final_rows,
        )

        summary_rows = aggregate_seed_grid_rows(collect_seed_grid_final_rows(all_run_results))

    return SweepResult(
        project_dir=project_dir,
        run_results=all_run_results,
        summary_rows=summary_rows,
    )


def _seed_reporter_stack_factory(variant_dir: Path, seed: int, *, per_seed_plots: bool):
    def factory(config: ExperimentConfig):
        from experiments.reporting.artifacts import PolicyArtifactReporter
        from experiments.reporting.base import ReporterStack
        from experiments.reporting.console import ConsoleReporter
        from experiments.reporting.json_summary import JsonReporter
        from experiments.reporting.plots import PlotReporter
        from experiments.reporting.step_logger import FileStepLogger
        from experiments.reporting.wandb import WandbReporter

        reporters = [
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            PolicyArtifactReporter(),
            JsonReporter(summary_name=f"summary-seed-{seed}.json", summary_dir=variant_dir),
        ]
        if per_seed_plots:
            reporters.append(PlotReporter())
        if config.wandb_enabled:
            reporters.append(WandbReporter())
        return ReporterStack(reporters)

    return factory


def _stringify_override_value(value: Any) -> str:
    text = str(value)
    return text.replace(" ", "").replace("/", "-")


def _display_key_label(key: str) -> str:
    aliases = {
        "n_grad_samples": "ngrad",
        "n_samples": "nsamp",
    }
    return aliases.get(key, key)


def _project_runs_root(runs_root: str, project_name: str | None) -> str:
    if not project_name:
        return runs_root
    return str(Path(runs_root) / _stringify_override_value(project_name))
