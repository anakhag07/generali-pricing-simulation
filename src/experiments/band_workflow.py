"""Fitted-band preparation, output integrity and saved-policy endpoint replay."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from experiments import execution
from experiments.configs import get_config
from experiments.envelopes.core import FittedEnvelope, TargetDomain
from experiments.envelopes.fitting import fit_band, load_band, save_band
from experiments.envelopes.support import fit_support_band
from experiments.provenance import file_record, read_json, save_arrays_atomic, write_json_atomic
from experiments.reporting.context import create_run_context
from experiments.seeds import replicate_seed_setup
from experiments.settings import prepare_setting

if TYPE_CHECKING:
    from experiments.manifest import ExperimentManifest, ManifestVariant


def _validate_record(record: Mapping[str, Any], *, required: bool = False) -> bool:
    path = Path(record["path"])
    if not path.exists():
        if required:
            raise ValueError(f"Saved fitted state or optimizer summary is missing: {path}.")
        return False
    if file_record(path) != record:
        raise ValueError(f"Saved band output or optimizer summary differs from recorded provenance: {path}.")
    return True


def band_outputs_complete(folder: Path, seeds: Sequence[int]) -> bool:
    """Validate existing artifacts even for partial runs; detect missing exports.

    Missing derived evaluations can be replayed from saved fitted state/theta.
    Changed bytes or missing recorded fitted state/optimizer summaries cannot.
    """
    band_path = folder / "band.json"
    binding_path = folder / "band-binding.json"
    if binding_path.exists():
        _validate_record(read_json(binding_path)["band"], required=True)
    elif band_path.exists():
        raise ValueError("Fitted-band provenance is incomplete; choose a fresh output location.")
    if band_path.exists():
        load_band(band_path)

    summary_path = folder / "band-summary.json"
    complete = summary_path.exists()
    if complete:
        record = read_json(summary_path)
        _validate_record(record["band"], required=True)
        complete = _validate_record(record["evaluation"])
    for seed in seeds:
        marker = folder / f"band-policy-seed-{seed}.json"
        if not marker.exists():
            complete = False
            continue
        outputs = read_json(marker)
        _validate_record(outputs["summary"], required=True)
        for item in outputs["evaluations"]:
            if not _validate_record(item):
                complete = False
    return complete


def _prepare_band(
    folder: Path,
    specification: Mapping[str, Any],
    setting_spec: Mapping[str, Any] | None,
    base: Path,
) -> FittedEnvelope:
    band_path = folder / "band.json"
    binding_path = folder / "band-binding.json"
    if band_path.exists():
        if not binding_path.exists() or read_json(binding_path)["band"] != file_record(band_path):
            raise ValueError("Fitted-band provenance changed or is incomplete; choose a fresh output location.")
        return load_band(band_path)
    if list(folder.glob("summary-seed-*.json")) or binding_path.exists():
        raise ValueError("Saved fitted state is missing; cannot refit a partially completed run.")
    if "artifact" in specification:
        source = Path(specification["artifact"])
        source = source if source.is_absolute() else base / source
        band = load_band(source)
        save_band(band, band_path)
        write_json_atomic(binding_path, {"band": file_record(band_path), "source": file_record(source)})
        return band

    setting = prepare_setting(setting_spec, base_path=base)
    settings = dict(specification)
    if settings["method"] == "support_grid":
        if setting.provenance.get("quantity") != settings["quantity"]:
            raise ValueError("Support band quantity must match the explicitly prepared Generali quantity.")
        band = fit_support_band(
            setting.X,
            setting.arrays["action_grid"],
            setting.arrays["predictor_grid"],
            settings,
            support=setting.arrays.get("support"),
            support_width=setting.arrays.get("support_width"),
            acceptance_grid=setting.arrays.get("acceptance_grid"),
            row_indices=setting.arrays.get("row_indices"),
        )
    else:
        multipliers = setting.multipliers
        if settings["method"] == "residual_wild":
            count = settings["B"]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 2
                or multipliers is None
                or count > len(multipliers)
            ):
                raise ValueError("B must select at least two available saved/paired multiplier rows.")
            multipliers = multipliers[:count]
        band = fit_band(
            setting.observations,
            setting.predictor,
            TargetDomain(setting.X),
            settings,
            multipliers=multipliers,
        )
    band.metadata["setting"] = setting.provenance
    save_band(band, band_path)
    write_json_atomic(binding_path, {"band": file_record(band_path)})
    return band


def _save_or_validate_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Saved band endpoints changed: {path}; preserve the run and use a fresh location.")
    else:
        save_arrays_atomic(path, **arrays)


def _write_band_evaluation(folder: Path, band: FittedEnvelope) -> None:
    if "actions" in band.state:
        X, actions = band.state["X"], band.state["actions"]
    else:
        contexts, grid = band.state["X"], band.state["action_grid"]
        X = np.repeat(contexts, len(grid), axis=0)
        actions = np.tile(grid, len(contexts))
    evaluated = band.evaluate(X, actions)
    evaluation_path = folder / "band-evaluation.npz"
    _save_or_validate_arrays(evaluation_path, {
        "X": X,
        "actions": actions,
        "center": evaluated.center,
        "lower": evaluated.lower,
        "upper": evaluated.upper,
        **evaluated.components,
    })
    write_json_atomic(folder / "band-summary.json", {
        "band": file_record(folder / "band.json"),
        "metadata": band.metadata,
        "evaluation": file_record(evaluation_path),
        "mean_lower_radius": float(np.mean(evaluated.center - evaluated.lower)),
        "mean_upper_radius": float(np.mean(evaluated.upper - evaluated.center)),
    })


def execute_band_variant(
    manifest: ExperimentManifest,
    variant: ManifestVariant,
    *,
    runs_root: str | Path | None = None,
) -> int:
    """Fit/load one band and run missing optimizer seeds through the run lifecycle.

    The manifest caller verifies the source/input contract and existing outputs.
    Fitting uses the setting's saved/explicit draws once per variant; optimizer
    seeds keep the existing replicate seed policy. Endpoint-only recovery replays
    saved theta without fitting or optimizing again. Returns newly executed runs.
    """
    folder = manifest.variant_dir(variant, runs_root)
    options = dict(variant.overrides)
    specification = options.pop("band")
    setting_spec = options.pop("setting", None)
    base = manifest.source_path.parent if manifest.source_path else Path.cwd()
    band = _prepare_band(folder, specification, setting_spec, base)
    _write_band_evaluation(folder, band)
    if manifest.task == "band":
        return 1

    band_reference = file_record(folder / "band.json")
    options.pop("plot", None)
    n_runs = 0
    for seed in manifest.seeds.run_seeds:
        summary = folder / f"summary-seed-{seed}.json"
        seeds = replicate_seed_setup(
            seed, manifest.seeds.anchor_seed,
            vary=manifest.seeds.vary, fixed=manifest.seeds.fixed,
        )
        config = get_config(manifest.base_preset, {
            **options, "fitted_band": band, "band_artifact": band_reference,
            "seed_setup": seeds, "plot": False,
        })
        context = create_run_context(
            variant.name,
            run_dir=folder / "seeds" / f"seed-{seed}",
            run_metadata={
                "preset_name": manifest.base_preset, "run_seed": seed,
                "variant_name": variant.name, "overrides": variant.overrides,
                "band": band_reference, "setting": band.metadata.get("setting"),
            },
        )
        if summary.exists():
            parameters = {
                name: np.asarray(item["theta"])
                for name, item in read_json(summary)["estimators"].items()
            }
        else:
            executed = execution.execute_experiment_run(
                variant.name, config, run_context=context,
                reporter_stack_factory=execution.seed_reporter_stack_factory(
                    folder, seed, per_seed_plots=False,
                ),
            )
            parameters = {name: result.theta for name, result in executed.result.results.items()}
            n_runs += 1
        records = []
        for estimator, theta in parameters.items():
            actions = config.objective.policy.value(theta, config.x_fixed)
            evaluated = band.evaluate(band.state["X"], actions)
            arrays = {
                "X": band.state["X"], "theta": theta, "actions": actions,
                "center": evaluated.center, "lower": evaluated.lower, "upper": evaluated.upper,
            }
            path = context.run_dir / f"band-policy-{estimator}.npz"
            _save_or_validate_arrays(path, arrays)
            records.append(file_record(path))
        write_json_atomic(folder / f"band-policy-seed-{seed}.json", {
            "summary": file_record(summary), "evaluations": records,
        })
    return n_runs


__all__ = ["band_outputs_complete", "execute_band_variant"]
