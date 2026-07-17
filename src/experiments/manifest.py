"""Manifest-driven experiment orchestration.

The manifest layer is intentionally thin: it validates a human-readable JSON
contract, expands variants, delegates execution to ``run_sweep(...)``, and
rebuilds project-level outputs from saved per-seed summaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.configs import get_config
from experiments.paths import results_root
from experiments.sweep_reporting import (
    DEFAULT_SEED_METRIC_BARS,
    aggregate_seed_grid_rows,
    write_rows_csv,
    write_seed_grid_csvs,
)
from experiments.sweep_utils import run_sweep as run_seed_sweep
from objective.modifications import coerce_objective_modification, modification_to_dict

TruthSource = Literal["clean_base_objective", "summary_json"]
LaunchArray = Literal["none", "variant"]
LaunchMode = Literal["auto", "local", "slurm"]


@dataclass(frozen=True)
class TruthSpec:
    """Explicit source of truth for derived metrics."""

    source: TruthSource
    path: Path | None = None
    estimator: str | None = None


@dataclass(frozen=True)
class SeedSpec:
    """Explicit seed policy for a manifest sweep."""

    run_seeds: tuple[int, ...]
    anchor_seed: int
    vary: tuple[str, ...]
    fixed: dict[str, int | None] | None = None


@dataclass(frozen=True)
class LaunchSpec:
    """Explicit launch policy for a manifest sweep."""

    mode: LaunchMode
    array: LaunchArray
    array_max_parallel: int | None = None
    requires_jax: bool | None = None


@dataclass(frozen=True)
class ManifestVariant:
    """One expanded manifest variant."""

    name: str
    overrides: dict[str, Any]
    axes: dict[str, Any]


@dataclass(frozen=True)
class ExperimentManifest:
    """Resolved manifest ready to execute."""

    name: str
    base_preset: str
    objective: dict[str, Any]
    objective_modifications: tuple[dict[str, Any], ...]
    optimizer: dict[str, Any]
    seeds: SeedSpec
    truth: TruthSpec
    launch: LaunchSpec
    variants: tuple[ManifestVariant, ...]
    defaults: dict[str, Any]
    per_seed_plots: bool
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / _path_part(self.name)

    def variant_dir(self, variant: ManifestVariant, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / _path_part(variant.name)

    def requires_jax(self) -> bool:
        if self.launch.requires_jax is not None:
            return bool(self.launch.requires_jax)
        return any(variant.overrides.get("compute_backend") == "jax" for variant in self.variants)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate an experiment manifest from JSON."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_experiment_manifest(payload, source_path=manifest_path)


def parse_experiment_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ExperimentManifest:
    """Validate a manifest payload and expand its variants."""
    if not isinstance(payload, Mapping):
        raise ValueError("Experiment manifest must be a JSON object.")

    source = None if source_path is None else Path(source_path)
    name = str(payload.get("name") or payload.get("project_name") or "").strip()
    if not name:
        raise ValueError("Manifest must specify name or project_name.")

    objective = _objective_payload(payload)
    base_preset = str(_mapping_value(objective, "preset")).strip()
    if not base_preset:
        raise ValueError("Manifest objective.preset must be non-empty.")

    if "objective_modifications" not in payload:
        raise ValueError("Manifest must explicitly specify objective_modifications, even if [].")
    objective_modifications = _objective_modifications(payload["objective_modifications"])

    optimizer = _required_mapping(payload, "optimizer")
    if not optimizer.get("step_rule"):
        raise ValueError("Manifest optimizer.step_rule is required.")

    seeds = _seed_spec(_required_mapping(payload, "seeds"))
    truth = _truth_spec(_required_mapping(payload, "truth"), source)
    launch = _launch_spec(_required_mapping(payload, "launch"))
    defaults = dict(_optional_mapping(payload.get("defaults"), field="defaults"))

    objective_overrides = dict(_optional_mapping(objective.get("overrides"), field="objective.overrides"))
    base_overrides: dict[str, Any] = {
        **objective_overrides,
        **defaults,
        **dict(optimizer),
        "objective_modifications": list(objective_modifications),
    }
    variants = _manifest_variants(payload, base_overrides)

    return ExperimentManifest(
        name=name,
        base_preset=base_preset,
        objective=dict(objective),
        objective_modifications=tuple(objective_modifications),
        optimizer=dict(optimizer),
        seeds=seeds,
        truth=truth,
        launch=launch,
        variants=tuple(variants),
        defaults=defaults,
        per_seed_plots=bool(payload.get("per_seed_plots", False)),
        source_path=source,
    )


def write_experiment_readme(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write the human-readable project descriptor for a manifest run."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    variant_lines = [
        "| Variant | Axes | Overrides |",
        "|---|---|---|",
    ]
    for variant in manifest.variants:
        variant_lines.append(
            "| "
            + " | ".join(
                [
                    _md_code(variant.name),
                    _md_code(json.dumps(_jsonable(variant.axes), sort_keys=True)),
                    _md_code(json.dumps(_jsonable(variant.overrides), sort_keys=True)),
                ]
            )
            + " |"
        )
    source_text = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    array_text = (
        "one task per variant; each task runs all seeds for that variant"
        if manifest.launch.array == "variant"
        else "one task; all variants and seeds run serially"
    )
    content = "\n".join(
        [
            f"# {manifest.name}",
            "",
            f"- Manifest source: `{source_text}`",
            f"- Objective preset: `{manifest.base_preset}`",
            f"- Truth source: `{_truth_description(manifest.truth)}`",
            f"- Launch mode: `{manifest.launch.mode}`",
            f"- Launch array structure: `{manifest.launch.array}` ({array_text})",
            f"- Requires JAX/GPU launch profile: `{manifest.requires_jax()}`",
            f"- Per-seed plots: `{manifest.per_seed_plots}`",
            "",
            "## Optimizer",
            "",
            "```json",
            json.dumps(_jsonable(manifest.optimizer), indent=2, sort_keys=True),
            "```",
            "",
            "## Seeds",
            "",
            "```json",
            json.dumps(
                {
                    "run_seeds": list(manifest.seeds.run_seeds),
                    "anchor_seed": manifest.seeds.anchor_seed,
                    "vary": list(manifest.seeds.vary),
                    "fixed": manifest.seeds.fixed,
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Objective Modifications",
            "",
            "```json",
            json.dumps(_jsonable(list(manifest.objective_modifications)), indent=2, sort_keys=True),
            "```",
            "",
            "## Variants",
            "",
            *variant_lines,
            "",
            "## Default Outputs",
            "",
            "- Every seed writes `summary-seed-<seed>.json` at the variant root.",
            "- Every seed writes heavy logs/plots/artifacts under `seeds/seed-<seed>/`.",
            "- Every variant writes `seed_grid_finals.csv`, `seed_grid_summary.csv`, "
            "seed metric bar plots, a seed frontier plot, and seed loss bands.",
            "- Multi-variant manifests write project-level seed-grid CSVs/plots and "
            "`derived_metrics.csv` from saved seed summaries.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def variant_complete(
    manifest: ExperimentManifest,
    variant: ManifestVariant,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return true when all requested seed summaries exist for ``variant``."""
    variant_dir = manifest.variant_dir(variant, runs_root)
    return all((variant_dir / f"summary-seed-{seed}.json").exists() for seed in manifest.seeds.run_seeds)


def run_manifest_variant(
    manifest: ExperimentManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one manifest variant across all manifest seeds."""
    variant = manifest.variants[index]
    write_experiment_readme(manifest, runs_root=runs_root)
    if not force and variant_complete(manifest, variant, runs_root=runs_root):
        return {
            "project_dir": str(manifest.project_dir(runs_root)),
            "variant": variant.name,
            "skipped": True,
            "n_runs": 0,
        }

    sweep = run_seed_sweep(
        base_preset=manifest.base_preset,
        run_seeds=manifest.seeds.run_seeds,
        override_list=[{"_run_name": variant.name, **variant.overrides}],
        vary=manifest.seeds.vary,
        anchor_seed=manifest.seeds.anchor_seed,
        fixed=manifest.seeds.fixed,
        per_seed_plots=manifest.per_seed_plots,
        runs_root=runs_root,
        project_name=manifest.name,
    )
    return {
        "project_dir": str(sweep.project_dir),
        "variant": variant.name,
        "skipped": False,
        "n_runs": len(sweep.run_results),
    }


def run_manifest_serial(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run all manifest variants serially and collect project outputs."""
    payloads = [
        run_manifest_variant(manifest, index, runs_root=runs_root, force=force)
        for index in range(len(manifest.variants))
    ]
    collect_manifest_outputs(manifest, runs_root=runs_root)
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "n_variants": len(manifest.variants),
        "n_runs": sum(int(payload["n_runs"]) for payload in payloads),
        "n_skipped_variants": sum(1 for payload in payloads if payload["skipped"]),
    }


def collect_manifest_outputs(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild project-level CSVs/plots from saved manifest seed summaries."""
    write_experiment_readme(manifest, runs_root=runs_root)
    project_dir = manifest.project_dir(runs_root)
    final_rows = collect_manifest_final_rows(manifest, runs_root=runs_root)
    summary_rows = aggregate_seed_grid_rows(final_rows)
    if final_rows:
        write_seed_grid_csvs(project_dir, final_rows, summary_rows)
        _write_seed_grid_plots(project_dir, summary_rows)
    derived_rows = collect_derived_metric_rows(manifest, runs_root=runs_root)
    if derived_rows:
        fieldnames = sorted({key for row in derived_rows for key in row})
        leading = ["variant", "run_seed", "estimator", "truth_source"]
        ordered = [key for key in leading if key in fieldnames] + [
            key for key in fieldnames if key not in leading
        ]
        write_rows_csv(project_dir / "derived_metrics.csv", derived_rows, ordered)
    return {
        "project_dir": str(project_dir),
        "n_final_rows": len(final_rows),
        "n_summary_rows": len(summary_rows),
        "n_derived_rows": len(derived_rows),
    }


def collect_manifest_final_rows(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Read per-seed summaries into the canonical seed-grid final row shape."""
    rows: list[dict[str, object]] = []
    for variant in manifest.variants:
        for seed in manifest.seeds.run_seeds:
            summary_path = manifest.variant_dir(variant, runs_root) / f"summary-seed-{seed}.json"
            if not summary_path.exists():
                continue
            payload = _read_json(summary_path)
            estimators = payload.get("estimators") or {}
            if not isinstance(estimators, Mapping):
                continue
            run_dir = str((payload.get("run") or {}).get("run_dir") or "")
            for estimator, estimator_payload in estimators.items():
                if not isinstance(estimator_payload, Mapping):
                    continue
                row: dict[str, object] = {
                    "variant": variant.name,
                    "run_seed": int(seed),
                    "run_dir": run_dir,
                    "estimator": str(estimator),
                    "final_u": _blank_none(estimator_payload.get("final_u")),
                    "final_value": _blank_none(estimator_payload.get("final_value")),
                    "runtime_sec": _blank_none(estimator_payload.get("runtime_sec")),
                    "mean_acceptance": _blank_none(estimator_payload.get("mean_acceptance")),
                    "constraint_violation": _blank_none(
                        estimator_payload.get("constraint_violation")
                    ),
                }
                row.update(_evaluation_fields("train", estimator_payload.get("train")))
                row.update(_evaluation_fields("test", estimator_payload.get("test")))
                rows.append(row)
    return rows


def collect_derived_metric_rows(
    manifest: ExperimentManifest,
    *,
    runs_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Read saved summaries and add metrics against the manifest truth source."""
    rows: list[dict[str, object]] = []
    truth_cache: dict[str, dict[str, object]] = {}
    for variant in manifest.variants:
        truth_payload = _truth_payload(manifest, variant, truth_cache)
        for seed in manifest.seeds.run_seeds:
            summary_path = manifest.variant_dir(variant, runs_root) / f"summary-seed-{seed}.json"
            if not summary_path.exists():
                continue
            payload = _read_json(summary_path)
            estimators = payload.get("estimators") or {}
            if not isinstance(estimators, Mapping):
                continue
            for estimator, estimator_payload in estimators.items():
                if not isinstance(estimator_payload, Mapping):
                    continue
                theta = _theta(estimator_payload)
                row: dict[str, object] = {
                    "variant": variant.name,
                    "run_seed": int(seed),
                    "estimator": str(estimator),
                    "summary_path": str(summary_path),
                    "final_value": _blank_none(estimator_payload.get("final_value")),
                    "mean_acceptance": _blank_none(estimator_payload.get("mean_acceptance")),
                    **truth_payload,
                }
                truth_theta = truth_payload.get("truth_theta")
                if theta is not None and truth_theta is not None:
                    truth_arr = np.asarray(truth_theta, dtype=float)
                    if theta.shape == truth_arr.shape:
                        row["theta_l2_gap"] = float(np.linalg.norm(theta - truth_arr))
                row.pop("truth_theta", None)
                rows.append(row)
    return rows


def _objective_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "objective" in payload:
        return _required_mapping(payload, "objective")
    if "base_preset" in payload:
        return {"preset": payload["base_preset"]}
    raise ValueError("Manifest must specify objective.preset.")


def _objective_modifications(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("objective_modifications must be a JSON array.")
    modifications: list[dict[str, Any]] = []
    for item in value:
        modification = coerce_objective_modification(item)
        modifications.append(modification_to_dict(modification))
    return modifications


def _seed_spec(payload: Mapping[str, Any]) -> SeedSpec:
    for key in ("run_seeds", "anchor_seed", "vary"):
        if key not in payload:
            raise ValueError(f"Manifest seeds.{key} is required.")
    run_seeds = tuple(int(seed) for seed in _sequence(payload["run_seeds"], "seeds.run_seeds"))
    if not run_seeds:
        raise ValueError("Manifest seeds.run_seeds must contain at least one seed.")
    vary = tuple(str(item) for item in _sequence(payload["vary"], "seeds.vary"))
    if not vary:
        raise ValueError("Manifest seeds.vary must contain at least one stream.")
    fixed_payload = payload.get("fixed")
    fixed = None
    if fixed_payload is not None:
        fixed_map = _optional_mapping(fixed_payload, field="seeds.fixed")
        fixed = {str(key): None if value is None else int(value) for key, value in fixed_map.items()}
    return SeedSpec(
        run_seeds=run_seeds,
        anchor_seed=int(payload["anchor_seed"]),
        vary=vary,
        fixed=fixed,
    )


def _truth_spec(payload: Mapping[str, Any], source_path: Path | None) -> TruthSpec:
    source = str(_mapping_value(payload, "source"))
    if source == "clean_base_objective":
        return TruthSpec(source="clean_base_objective")
    if source == "summary_json":
        raw_path = Path(str(_mapping_value(payload, "path")))
        path = raw_path if raw_path.is_absolute() or source_path is None else source_path.parent / raw_path
        estimator = str(_mapping_value(payload, "estimator"))
        if not estimator:
            raise ValueError("Manifest truth.estimator must be non-empty for summary_json truth.")
        return TruthSpec(source="summary_json", path=path, estimator=estimator)
    raise ValueError("Manifest truth.source must be 'clean_base_objective' or 'summary_json'.")


def _launch_spec(payload: Mapping[str, Any]) -> LaunchSpec:
    mode = str(_mapping_value(payload, "mode"))
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("Manifest launch.mode must be 'auto', 'local', or 'slurm'.")
    array = str(_mapping_value(payload, "array"))
    if array not in {"none", "variant"}:
        raise ValueError("Manifest launch.array must be 'none' or 'variant'.")
    max_parallel = payload.get("array_max_parallel")
    requires_jax = payload.get("requires_jax")
    return LaunchSpec(
        mode=mode,  # type: ignore[arg-type]
        array=array,  # type: ignore[arg-type]
        array_max_parallel=None if max_parallel is None else int(max_parallel),
        requires_jax=None if requires_jax is None else bool(requires_jax),
    )


def _manifest_variants(
    payload: Mapping[str, Any],
    base_overrides: Mapping[str, Any],
) -> list[ManifestVariant]:
    if "variants" in payload and "matrix" in payload:
        raise ValueError("Manifest must specify either variants or matrix, not both.")
    if "variants" in payload:
        return _explicit_variants(payload["variants"], base_overrides)
    matrix_value = payload["matrix"] if "matrix" in payload else {}
    return _matrix_variants(
        _optional_mapping(matrix_value, field="matrix"),
        base_overrides,
        run_name_template=payload.get("run_name_template"),
    )


def _explicit_variants(value: Any, base_overrides: Mapping[str, Any]) -> list[ManifestVariant]:
    items = _sequence(value, "variants")
    variants: list[ManifestVariant] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("Each variants[] item must be a JSON object.")
        name = str(item.get("name") or f"variant-{index:03d}")
        overrides = {
            **dict(base_overrides),
            **dict(_optional_mapping(item.get("overrides") or {}, field="variants[].overrides")),
        }
        axes = dict(_optional_mapping(item.get("axes") or {}, field="variants[].axes"))
        variants.append(ManifestVariant(name=name, overrides=_normalize_overrides(overrides), axes=axes))
    if not variants:
        raise ValueError("Manifest variants must contain at least one variant.")
    return variants


def _matrix_variants(
    matrix: Mapping[str, Any],
    base_overrides: Mapping[str, Any],
    *,
    run_name_template: Any,
) -> list[ManifestVariant]:
    if not matrix:
        return [ManifestVariant(name="base", overrides=dict(base_overrides), axes={})]

    axes = list(matrix.keys())
    axis_values = [_sequence(matrix[axis], f"matrix.{axis}") for axis in axes]
    variants: list[ManifestVariant] = []
    for index, combo in enumerate(product(*axis_values), start=1):
        overrides = dict(base_overrides)
        labels: dict[str, str] = {}
        axis_payload: dict[str, Any] = {}
        for axis, raw_value in zip(axes, combo):
            entry = _axis_entry(axis, raw_value)
            labels[axis] = entry["label"]
            axis_payload[axis] = entry["value"]
            overrides.update(entry["overrides"])
        if run_name_template is not None:
            # `labels` and `axis_payload` are both keyed by axis name, so expanding
            # them together would collide. Expose each axis as `{axis}` (its value)
            # and `{axis}_label` (its display label) so template keys stay distinct.
            template_vars: dict[str, Any] = {"index": index}
            for axis in axes:
                template_vars[axis] = axis_payload[axis]
                template_vars[f"{axis}_label"] = labels[axis]
            name = str(run_name_template).format(**template_vars)
        else:
            name = "__".join(labels[axis] for axis in axes)
        variants.append(
            ManifestVariant(name=name, overrides=_normalize_overrides(overrides), axes=axis_payload)
        )
    return variants


def _axis_entry(axis: str, raw_value: Any) -> dict[str, Any]:
    if isinstance(raw_value, Mapping) and ("label" in raw_value or "overrides" in raw_value):
        label = str(raw_value.get("label") or f"{axis}-{_path_part(raw_value.get('value', 'value'))}")
        value = raw_value.get("value", label)
        overrides = dict(_optional_mapping(raw_value.get("overrides") or {}, field=f"matrix.{axis}.overrides"))
        return {"label": label, "value": value, "overrides": overrides}
    return {
        "label": f"{axis}-{_path_part(raw_value)}",
        "value": raw_value,
        "overrides": {axis: raw_value},
    }


def _normalize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    if "objective_modifications" in normalized:
        normalized["objective_modifications"] = _objective_modifications(
            normalized["objective_modifications"]
        )
    return normalized


def _truth_payload(
    manifest: ExperimentManifest,
    variant: ManifestVariant,
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    if manifest.truth.source == "summary_json":
        assert manifest.truth.path is not None
        cache_key = f"summary:{manifest.truth.path}:{manifest.truth.estimator}"
        if cache_key not in cache:
            payload = _read_json(manifest.truth.path)
            estimators = payload.get("estimators") or {}
            if not isinstance(estimators, Mapping) or manifest.truth.estimator not in estimators:
                raise ValueError(
                    f"Truth summary {manifest.truth.path} does not contain estimator "
                    f"{manifest.truth.estimator!r}."
                )
            estimator_payload = estimators[manifest.truth.estimator]
            if not isinstance(estimator_payload, Mapping):
                raise ValueError("Truth summary estimator payload must be a JSON object.")
            theta = _theta(estimator_payload)
            cache[cache_key] = {
                "truth_source": "summary_json",
                "truth_summary_path": str(manifest.truth.path),
                "truth_estimator": str(manifest.truth.estimator),
                "truth_value": _blank_none(estimator_payload.get("final_value")),
                "truth_theta": theta,
            }
        return dict(cache[cache_key])

    cache_key = f"clean:{variant.name}"
    if cache_key not in cache:
        config = get_config(manifest.base_preset, overrides=_clean_truth_overrides(variant.overrides))
        objective = _base_objective(config.objective)
        optimal_theta_fn = getattr(objective, "optimal_theta", None)
        optimal_value_fn = getattr(objective, "optimal_value", None)
        payload: dict[str, object] = {
            "truth_source": "clean_base_objective",
            "truth_objective_type": type(objective).__name__,
        }
        if callable(optimal_theta_fn):
            payload["truth_theta"] = np.asarray(optimal_theta_fn(), dtype=float)
        if callable(optimal_value_fn):
            payload["truth_value"] = float(optimal_value_fn())
        cache[cache_key] = payload
    return dict(cache[cache_key])


def _clean_truth_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(overrides)
    clean["objective_modifications"] = []
    return clean


def _base_objective(objective: object) -> object:
    current = objective
    seen: set[int] = set()
    while hasattr(current, "base_objective") and id(current) not in seen:
        seen.add(id(current))
        current = getattr(current, "base_objective")
    return current


def _write_seed_grid_plots(project_dir: Path, summary_rows: Sequence[Mapping[str, object]]) -> None:
    if not summary_rows:
        return
    from reporting.visualization import plot_seed_grid_frontier, plot_seed_grid_metric_bars

    plot_dir = str(project_dir / "plots")
    for metric, y_label, filename in DEFAULT_SEED_METRIC_BARS:
        plot_seed_grid_metric_bars(summary_rows, plot_dir, metric=metric, y_label=y_label, filename=filename)
    plot_seed_grid_frontier(summary_rows, plot_dir)


def _evaluation_fields(prefix: str, payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {
            f"{prefix}_objective_value": "",
            f"{prefix}_objective_sum": "",
            f"{prefix}_mean_u": "",
            f"{prefix}_mean_acceptance": "",
        }
    return {
        f"{prefix}_objective_value": _blank_none(payload.get("objective_value")),
        f"{prefix}_objective_sum": _blank_none(payload.get("objective_sum")),
        f"{prefix}_mean_u": _blank_none(payload.get("mean_u")),
        f"{prefix}_mean_acceptance": _blank_none(payload.get("mean_acceptance")),
    }


def _theta(payload: Mapping[str, Any]) -> np.ndarray | None:
    if payload.get("theta") is None:
        return None
    arr = np.asarray(payload["theta"], dtype=float)
    return arr if arr.ndim == 1 else None


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def _mapping_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Manifest field {key} is required.")
    return payload[key]


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _mapping_value(payload, key)
    return _optional_mapping(value, field=key)


def _optional_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest {field} must be a JSON object.")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Manifest {field} must be a JSON array.")
    return value


def _blank_none(value: object) -> object:
    return "" if value is None else value


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _truth_description(truth: TruthSpec) -> str:
    if truth.source == "summary_json":
        return f"summary_json path={truth.path} estimator={truth.estimator}"
    return "clean_base_objective"


def _md_code(value: str) -> str:
    return "`" + value.replace("`", "\\`") + "`"


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-") or "run"


__all__ = [
    "ExperimentManifest",
    "LaunchSpec",
    "ManifestVariant",
    "SeedSpec",
    "TruthSpec",
    "collect_derived_metric_rows",
    "collect_manifest_final_rows",
    "collect_manifest_outputs",
    "load_experiment_manifest",
    "parse_experiment_manifest",
    "run_manifest_serial",
    "run_manifest_variant",
    "variant_complete",
    "write_experiment_readme",
]
