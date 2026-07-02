"""Utilities for running preset-based parameter sweeps."""

from __future__ import annotations

from dataclasses import fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.config import ExperimentConfig
from experiments.configs import get_config
from experiments.reporters import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    PolicyArtifactReporter,
    ReporterStack,
    WandbReporter,
    create_run_context,
)
from experiments.results import ExperimentResult
from experiments.run import run_experiment

_RUN_NAME_KEY = "_run_name"


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


def generate_sweep_runs(
    *,
    base_preset: str,
    override_grid: Mapping[str, Sequence[Any]] | None = None,
    override_list: Sequence[Mapping[str, Any]] | None = None,
    display_keys: Sequence[str] | None = None,
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

    runs: list[tuple[str, ExperimentConfig, dict[str, Any]]] = []
    for index, override in enumerate(overrides, start=1):
        override_payload = dict(override)
        explicit_run_name = override_payload.pop(_RUN_NAME_KEY, None)
        config = get_config(base_preset, overrides=override_payload)
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
) -> list[tuple[str, ExperimentResult]]:
    """Execute a preset sweep and return `(run_name, result)` pairs."""
    sweep_runs = generate_sweep_runs(
        base_preset=base_preset,
        override_grid=override_grid,
        override_list=override_list,
        display_keys=display_keys,
    )
    results: list[tuple[str, ExperimentResult]] = []
    runs_root_path = _project_runs_root(runs_root, project_name)

    for run_name, config, _ in sweep_runs:
        run_context = create_run_context(run_name, runs_root=runs_root_path)
        reporter_list = [
            ConsoleReporter(verbose=config.verbose),
            FileStepLogger(),
            PolicyArtifactReporter(),
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
