"""JSON manifest orchestration for seed-aware experiment sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.paths import results_root
from experiments.sweep_utils import SweepResult, run_sweep
from objective.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)
from objective.objectives import BiasedObjective, LinearActionBias, UpperSupportHingeBias


_SPECIAL_AXES = {"noise", "bias"}


@dataclass(frozen=True)
class ManifestVariant:
    """One expanded manifest variant before seed replication."""

    name: str
    axes: dict[str, Any]
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ManifestSweepResult:
    """Execution summary for one sweep block in a manifest."""

    name: str
    base_preset: str
    project_dir: Path
    variants: list[ManifestVariant]
    skipped_variants: list[str]
    sweep_results: list[SweepResult]
    dry_run: bool = False

    @property
    def executed_runs(self) -> int:
        return sum(len(result.run_results) for result in self.sweep_results)


@dataclass(frozen=True)
class ExperimentManifestResult:
    """Execution summary for a full JSON experiment manifest."""

    path: Path | None
    sweeps: list[ManifestSweepResult]

    @property
    def executed_runs(self) -> int:
        return sum(result.executed_runs for result in self.sweeps)


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment manifest from disk."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Experiment manifest must be a JSON object.")
    return dict(payload)


def run_experiment_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    dry_run: bool = False,
    runs_root: str | Path | None = None,
) -> ExperimentManifestResult:
    """Run every sweep in a JSON experiment manifest."""
    manifest_path: Path | None = None
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        payload = load_experiment_manifest(manifest_path)
    else:
        payload = dict(manifest)

    results = [
        run_manifest_sweep(sweep, dry_run=dry_run, runs_root=runs_root)
        for sweep in _sweep_payloads(payload)
    ]
    return ExperimentManifestResult(path=manifest_path, sweeps=results)


def plan_manifest_sweep(payload: Mapping[str, Any]) -> tuple[list[ManifestVariant], dict[str, Any]]:
    """Expand one sweep block without executing it."""
    sweep = dict(payload)
    base_preset = _required_str(sweep, "base_preset")
    defaults = _defaults(sweep)
    matrix = _matrix(sweep)
    variants: list[ManifestVariant] = []

    for axes in _axis_products(matrix):
        variant_name = _variant_name(sweep, base_preset=base_preset, axes=axes)
        direct_overrides = _direct_overrides(defaults, axes)
        objective = _wrapped_objective(base_preset, direct_overrides, axes, sweep)
        overrides = dict(direct_overrides)
        if objective is not None:
            overrides["objective"] = objective
        overrides["_run_name"] = variant_name
        variants.append(
            ManifestVariant(
                name=variant_name,
                axes=dict(axes),
                overrides=overrides,
            )
        )

    return variants, sweep


def run_manifest_sweep(
    payload: Mapping[str, Any],
    *,
    dry_run: bool = False,
    runs_root: str | Path | None = None,
) -> ManifestSweepResult:
    """Run one manifest sweep block through the existing seed-aware runner."""
    variants, sweep = plan_manifest_sweep(payload)
    base_preset = _required_str(sweep, "base_preset")
    name = str(sweep.get("name") or sweep.get("project_name") or f"{base_preset}-manifest")
    project_name = str(sweep.get("project_name") or name)
    run_seeds = _int_tuple(sweep.get("run_seeds", [7]), key="run_seeds")
    vary = tuple(str(item) for item in sweep.get("vary", ["theta"]))
    anchor_seed = sweep.get("anchor_seed")
    fixed = _fixed_seeds(sweep.get("fixed"))
    display_keys = tuple(str(item) for item in sweep.get("display_keys", ()))
    per_seed_plots = bool(sweep.get("per_seed_plots", False))
    skip_completed = bool(sweep.get("skip_completed", True))
    completion = _completion_spec(sweep)
    root = _runs_root(runs_root)
    project_dir = root / _path_part(project_name)
    runnable_variants = variants
    skipped: list[str] = []

    if skip_completed:
        runnable_variants = []
        for variant in variants:
            if _variant_all_seeds_completed(
                project_dir / _path_part(variant.name),
                run_seeds=run_seeds,
                required_estimators=completion["required_estimators"],
            ):
                skipped.append(variant.name)
            else:
                runnable_variants.append(variant)

    if dry_run or not runnable_variants:
        return ManifestSweepResult(
            name=name,
            base_preset=base_preset,
            project_dir=project_dir,
            variants=variants,
            skipped_variants=skipped,
            sweep_results=[],
            dry_run=dry_run,
        )

    sweep_results: list[SweepResult] = []
    if skip_completed and len(runnable_variants) < len(variants):
        # Avoid overwriting a project-level aggregate with only the missing
        # variants. Per-variant summaries are still regenerated by run_sweep.
        for variant in runnable_variants:
            sweep_results.append(
                run_sweep(
                    base_preset=base_preset,
                    run_seeds=run_seeds,
                    override_list=[dict(variant.overrides)],
                    vary=vary,
                    anchor_seed=None if anchor_seed is None else int(anchor_seed),
                    fixed=fixed,
                    per_seed_plots=per_seed_plots,
                    runs_root=root,
                    project_name=project_name,
                    display_keys=display_keys,
                )
            )
    else:
        sweep_results.append(
            run_sweep(
                base_preset=base_preset,
                run_seeds=run_seeds,
                override_list=[dict(variant.overrides) for variant in runnable_variants],
                vary=vary,
                anchor_seed=None if anchor_seed is None else int(anchor_seed),
                fixed=fixed,
                per_seed_plots=per_seed_plots,
                runs_root=root,
                project_name=project_name,
                display_keys=display_keys,
            )
        )

    return ManifestSweepResult(
        name=name,
        base_preset=base_preset,
        project_dir=project_dir,
        variants=variants,
        skipped_variants=skipped,
        sweep_results=sweep_results,
        dry_run=False,
    )


def manifest_requires_jax(manifest: Mapping[str, Any]) -> bool:
    """Return whether any sweep block explicitly requests a JAX compute backend."""
    for sweep in _sweep_payloads(manifest):
        defaults = _defaults(sweep)
        if defaults.get("compute_backend") == "jax":
            return True
        for axes in _axis_products(_matrix(sweep)):
            if any(key == "compute_backend" and value == "jax" for key, value in axes.items()):
                return True
            for value in axes.values():
                if isinstance(value, Mapping):
                    overrides = value.get("overrides")
                    if isinstance(overrides, Mapping) and overrides.get("compute_backend") == "jax":
                        return True
    return False


def completed_variant_names(
    *,
    project_dir: str | Path,
    variants: Sequence[ManifestVariant],
    run_seeds: Sequence[int],
    required_estimators: Sequence[str] = (),
) -> list[str]:
    """Return variant names whose expected seed summaries are already complete."""
    project_path = Path(project_dir)
    return [
        variant.name
        for variant in variants
        if _variant_all_seeds_completed(
            project_path / _path_part(variant.name),
            run_seeds=run_seeds,
            required_estimators=tuple(required_estimators),
        )
    ]


def _sweep_payloads(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "sweeps" not in payload:
        return [dict(payload)]
    sweeps = payload["sweeps"]
    if not isinstance(sweeps, Sequence) or isinstance(sweeps, (str, bytes)):
        raise ValueError("manifest.sweeps must be a list of sweep objects.")
    out: list[dict[str, Any]] = []
    for index, sweep in enumerate(sweeps, start=1):
        if not isinstance(sweep, Mapping):
            raise ValueError(f"manifest.sweeps[{index - 1}] must be a JSON object.")
        inherited = {key: value for key, value in payload.items() if key != "sweeps"}
        inherited.update(dict(sweep))
        out.append(inherited)
    if not out:
        raise ValueError("manifest.sweeps must contain at least one sweep.")
    return out


def _defaults(sweep: Mapping[str, Any]) -> dict[str, Any]:
    if "defaults" in sweep and "overrides" in sweep:
        raise ValueError("Use either 'defaults' or 'overrides' in a manifest sweep, not both.")
    defaults = sweep.get("defaults", sweep.get("overrides", {}))
    if defaults is None:
        return {}
    if not isinstance(defaults, Mapping):
        raise ValueError("manifest defaults must be a JSON object.")
    return {str(key): _coerce_override_value(str(key), value) for key, value in defaults.items()}


def _matrix(sweep: Mapping[str, Any]) -> dict[str, list[Any]]:
    raw = sweep.get("matrix", sweep.get("axes", {}))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("manifest matrix must be a JSON object.")
    matrix: dict[str, list[Any]] = {}
    for key, values in raw.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, Mapping)):
            raise ValueError(f"matrix axis '{key}' must be a JSON list.")
        if not values:
            raise ValueError(f"matrix axis '{key}' must contain at least one value.")
        matrix[str(key)] = [_coerce_override_value(str(key), value) for value in values]
    return matrix


def _axis_products(matrix: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    return [dict(zip(keys, values)) for values in product(*(matrix[key] for key in keys))]


def _direct_overrides(defaults: Mapping[str, Any], axes: Mapping[str, Any]) -> dict[str, Any]:
    overrides = dict(defaults)
    for key, value in axes.items():
        if key in _SPECIAL_AXES:
            continue
        if isinstance(value, Mapping) and "overrides" in value:
            axis_overrides = value["overrides"]
            if not isinstance(axis_overrides, Mapping):
                raise ValueError(f"axis '{key}' overrides must be a JSON object.")
            overrides.update(_coerced_overrides(axis_overrides))
        else:
            overrides[key] = value
    return overrides


def _wrapped_objective(
    base_preset: str,
    direct_overrides: Mapping[str, Any],
    axes: Mapping[str, Any],
    sweep: Mapping[str, Any],
) -> object | None:
    if not any(key in axes for key in _SPECIAL_AXES):
        return None
    base_config = get_config(base_preset, overrides=direct_overrides)
    objective = base_config.objective
    for wrapper_name in _wrapper_order(sweep, axes):
        spec = axes.get(wrapper_name)
        if spec is None:
            continue
        if wrapper_name == "bias":
            objective = _apply_bias(objective, spec)
        elif wrapper_name == "noise":
            objective = _apply_noise(objective, spec)
    return objective


def _wrapper_order(sweep: Mapping[str, Any], axes: Mapping[str, Any]) -> tuple[str, ...]:
    raw = sweep.get("objective_wrapper_order")
    if raw is None:
        return tuple(key for key in ("bias", "noise") if key in axes)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("objective_wrapper_order must be a JSON list.")
    order = tuple(str(item) for item in raw)
    unknown = sorted(set(order) - _SPECIAL_AXES)
    if unknown:
        raise ValueError(f"Unknown objective wrapper(s): {', '.join(unknown)}.")
    missing = tuple(key for key in _SPECIAL_AXES if key in axes and key not in order)
    return (*order, *missing)


def _apply_noise(objective: object, spec: Any) -> object:
    if isinstance(spec, str):
        spec = {"kind": spec}
    if not isinstance(spec, Mapping):
        raise ValueError("noise axis values must be strings or JSON objects.")
    kind = _kind(spec)
    if kind in {"none", "no_noise", "disabled"}:
        return objective
    if kind in {"homoskedastic", "homoskedastic_gaussian"}:
        noise = HomoskedasticGaussianNoise(std=float(spec.get("std", spec.get("sigma", 0.0))))
    elif kind in {"heteroskedastic", "heteroskedastic_gaussian"}:
        noise = HeteroskedasticGaussianNoise(
            base_std=float(spec.get("base_std", 0.0)),
            growth=float(spec.get("growth", spec.get("gamma", 1.0))),
            u_center=float(_resolve_reference(spec.get("u_center", "objective.optimal_u"), objective)),
        )
    else:
        raise ValueError(f"Unknown noise kind '{kind}'.")
    return NoisyObjective(base_objective=objective, noise=noise)


def _apply_bias(objective: object, spec: Any) -> object:
    if isinstance(spec, str):
        spec = {"kind": spec}
    if not isinstance(spec, Mapping):
        raise ValueError("bias axis values must be strings or JSON objects.")
    kind = _kind(spec)
    if kind in {"none", "no_bias", "disabled"}:
        return objective
    if kind in {"linear", "linear_action"}:
        bias = LinearActionBias(lambda_bias=float(spec.get("lambda_bias", spec.get("lambda", 0.0))))
    elif kind in {"upper_support_hinge", "support_hinge"}:
        bias = UpperSupportHingeBias(
            lambda_bias=float(spec.get("lambda_bias", spec.get("lambda", 0.0))),
            support_center=float(
                _resolve_reference(spec.get("support_center", "objective.optimal_u"), objective)
            ),
            support_radius=float(spec.get("support_radius", 0.0)),
            smooth_tau=(
                None
                if spec.get("smooth_tau") is None
                else float(spec.get("smooth_tau"))
            ),
        )
    else:
        raise ValueError(f"Unknown bias kind '{kind}'.")
    return BiasedObjective(base_objective=objective, bias=bias)


def _resolve_reference(value: Any, objective: object) -> Any:
    if not isinstance(value, str):
        return value
    if value in {"objective.optimal_u", "objective.u_star"}:
        optimal_u = getattr(objective, "optimal_u", None)
        if callable(optimal_u):
            return float(optimal_u())
        u_star = getattr(objective, "u_star", None)
        if u_star is not None:
            return float(u_star)
        return 0.0
    return value


def _kind(spec: Mapping[str, Any]) -> str:
    return str(spec.get("kind", spec.get("type", "none"))).lower().replace("-", "_")


def _coerce_override_value(key: str, value: Any) -> Any:
    if key == "correctness" and isinstance(value, Mapping):
        return CorrectnessSpec(**dict(value))
    if key in {"enabled_estimators", "constant_u_baselines", "wandb_tags", "wandb_estimator_allowlist"}:
        return tuple(value) if value is not None else None
    if key == "theta0" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return np.asarray(value, dtype=float)
    return value


def _coerced_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _coerce_override_value(str(key), value)
        for key, value in values.items()
    }


def _variant_name(
    sweep: Mapping[str, Any],
    *,
    base_preset: str,
    axes: Mapping[str, Any],
) -> str:
    template = sweep.get("run_name_template", sweep.get("variant_name_template"))
    if template is not None:
        return str(template).format_map(_format_context(axes))
    if not axes:
        return f"{base_preset}__run_001"
    return "__".join(_axis_label(key, value) for key, value in axes.items())


def _axis_label(key: str, value: Any) -> str:
    if isinstance(value, Mapping):
        if "label" in value:
            return _path_part(str(value["label"]))
        label_items = [
            (item_key, item_value)
            for item_key, item_value in value.items()
            if item_key not in {"overrides"}
        ]
        if not label_items:
            return key
        parts = [key]
        for item_key, item_value in label_items:
            parts.append(f"{item_key}-{_value_label(item_value)}")
        return "-".join(parts)
    return f"{key}-{_value_label(value)}"


def _value_label(value: Any) -> str:
    if value is None:
        return "all"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, Mapping):
        if "label" in value:
            return str(value["label"])
        return "-".join(f"{key}-{_value_label(item)}" for key, item in value.items())
    return str(value)


def _format_context(axes: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _namespace(value) for key, value in axes.items()}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _completion_spec(sweep: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = sweep.get("completion", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("completion must be a JSON object.")
    required = raw.get("required_estimators", sweep.get("required_estimators", ()))
    return {"required_estimators": tuple(str(item) for item in (required or ()))}


def _variant_all_seeds_completed(
    variant_dir: Path,
    *,
    run_seeds: Sequence[int],
    required_estimators: Sequence[str],
) -> bool:
    if not variant_dir.is_dir():
        return False
    for seed in run_seeds:
        if not _summary_has_estimators(
            variant_dir / f"summary-seed-{int(seed)}.json",
            required_estimators,
        ):
            return False
    return True


def _summary_has_estimators(path: Path, required_estimators: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    if not required_estimators:
        return True
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    estimators = payload.get("estimators", {})
    return all(name in estimators for name in required_estimators)


def _runs_root(runs_root: str | Path | None) -> Path:
    return results_root() if runs_root is None else Path(runs_root)


def _fixed_seeds(value: Any) -> dict[str, int | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("fixed must be a JSON object.")
    return {str(key): None if item is None else int(item) for key, item in value.items()}


def _int_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a JSON list.")
    if not value:
        raise ValueError(f"{key} must contain at least one integer.")
    return tuple(int(item) for item in value)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"manifest sweep requires '{key}'.")
    return str(value)


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-")


__all__ = [
    "ExperimentManifestResult",
    "ManifestSweepResult",
    "ManifestVariant",
    "completed_variant_names",
    "load_experiment_manifest",
    "manifest_requires_jax",
    "plan_manifest_sweep",
    "run_experiment_manifest",
    "run_manifest_sweep",
]
