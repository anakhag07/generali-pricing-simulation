"""Utilities for running preset-based parameter sweeps."""

from __future__ import annotations

from dataclasses import fields, replace
from itertools import product
from typing import Any, Mapping, Sequence

from experiments.config import ExperimentConfig
from experiments.configs import get_config
from experiments.reporters import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    ReporterStack,
    WandbReporter,
    create_run_context,
)
from experiments.results import ExperimentResult
from experiments.run import run_experiment


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


def generate_sweep_runs(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[str, ExperimentConfig, dict[str, Any]]]:
    """Generate named configs by applying overrides to a base preset."""
    if override_grid is not None and override_list is not None:
        raise ValueError("Specify either override_grid or override_list, not both.")

    if override_list is not None:
        overrides = [dict(item) for item in override_list] if override_list else [{}]
    elif override_grid is not None:
        overrides = expand_override_grid(override_grid)
    else:
        overrides = [{}]

    base_config = get_config(base_preset)
    runs: list[tuple[str, ExperimentConfig, dict[str, Any]]] = []
    for index, override in enumerate(overrides, start=1):
        config = apply_config_overrides(base_config, override)
        run_name = make_sweep_name(base_preset, index=index, overrides=override)
        runs.append((run_name, config, override))
    return runs


def run_preset_sweep(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    runs_root: str = "outputs",
) -> list[tuple[str, ExperimentResult]]:
    """Execute a preset sweep and return `(run_name, result)` pairs."""
    sweep_runs = generate_sweep_runs(
        base_preset=base_preset,
        override_grid=override_grid,
        override_list=override_list,
    )
    results: list[tuple[str, ExperimentResult]] = []

    for run_name, config, _ in sweep_runs:
        run_context = create_run_context(run_name, runs_root=runs_root)
        reporter_list = [
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            JsonReporter(),
            PlotReporter(),
        ]
        if config.wandb_enabled:
            reporter_list.append(WandbReporter())
        reporters = ReporterStack(reporter_list)
        reporters.on_start(run_context, config)
        result = run_experiment(config, step_reporter=reporters)
        reporters.on_end(run_context, result)
        results.append((run_name, result))

    return results


def _stringify_override_value(value: Any) -> str:
    text = str(value)
    return text.replace(" ", "").replace("/", "-")
