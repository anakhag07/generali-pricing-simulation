"""Run-lifecycle helpers for experiment execution with reporters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext, create_run_context
from experiments.results import ExperimentResult
from experiments.run import run_experiment

if TYPE_CHECKING:
    from experiments.reporting.base import ReporterStack
    from experiments.reporting.json_summary import JsonReporter


@dataclass(frozen=True)
class ExecutedRun:
    """Completed experiment run plus its output context."""

    name: str
    config: ExperimentConfig
    result: ExperimentResult
    run_context: RunContext


ReporterStackFactory = Callable[[ExperimentConfig], "ReporterStack"]


def default_reporter_stack(
    config: ExperimentConfig,
    *,
    json_reporter: "JsonReporter | None" = None,
    include_plots: bool = True,
) -> "ReporterStack":
    """Build the default reporter stack in its required execution order.

    This is the single source of truth for the reporter ordering contract:
    ``PolicyArtifactReporter`` before ``JsonReporter`` (so summaries can record
    artifact paths) and plots before the optional W&B upload. ``json_reporter``
    overrides the default ``JsonReporter()`` (e.g. seed sweeps inject a per-seed
    summary target) while keeping its slot; ``include_plots=False`` drops the
    ``PlotReporter`` for callers that skip per-run plots.
    """
    from experiments.reporting.artifacts import PolicyArtifactReporter
    from experiments.reporting.base import ReporterStack
    from experiments.reporting.console import ConsoleReporter
    from experiments.reporting.json_summary import JsonReporter
    from experiments.reporting.plots import PlotReporter
    from experiments.reporting.step_logger import FileStepLogger
    from experiments.reporting.wandb import WandbReporter

    reporter_list = [
        ConsoleReporter(verbose=config.verbose),
        FileStepLogger(),
        PolicyArtifactReporter(),
        json_reporter if json_reporter is not None else JsonReporter(),
    ]
    if include_plots:
        reporter_list.append(PlotReporter())
    if config.wandb_enabled:
        reporter_list.append(WandbReporter())
    return ReporterStack(reporter_list)


def execute_experiment_run(
    name: str,
    config: ExperimentConfig,
    *,
    runs_root: str | Path | None = None,
    reporter_stack_factory: ReporterStackFactory = default_reporter_stack,
    run_context: RunContext | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> ExecutedRun:
    """Create output context, run one experiment, and finalize reporters.

    Pass ``run_context`` to reuse a caller-built output directory (e.g. a per-seed
    replicate under a shared variant folder); otherwise one is created under
    ``runs_root`` or the shared external results root when ``runs_root`` is omitted.
    """
    if run_context is None:
        run_context = create_run_context(name, runs_root=runs_root, run_metadata=run_metadata)
    reporters = reporter_stack_factory(config)
    reporters.on_start(run_context, config)
    result = run_experiment(config, step_reporter=reporters)
    reporters.on_end(run_context, result)
    return ExecutedRun(
        name=name,
        config=config,
        result=result,
        run_context=run_context,
    )


__all__ = [
    "ExecutedRun",
    "ReporterStackFactory",
    "default_reporter_stack",
    "execute_experiment_run",
    "execute_band_variant",
]


def execute_band_variant(manifest, variant, *, runs_root=None):
    """Fit/load one band, then optionally run existing per-seed optimization and reporters."""
    import numpy as np
    from experiments.configs import get_config
    from experiments.envelopes.core import TargetDomain
    from experiments.envelopes.fitting import fit_band, load_band, save_band
    from experiments.envelopes.support import fit_support_band
    from experiments.provenance import (file_record, read_json, write_json_atomic, save_arrays_atomic)
    from experiments.seeds import replicate_seed_setup
    from experiments.settings import prepare_setting
    from experiments.sweep_utils import _seed_reporter_stack_factory

    folder = manifest.variant_dir(variant, runs_root)
    options = dict(variant.overrides)
    specification = options.pop("band")
    setting_spec = options.pop("setting", None)
    base = manifest.source_path.parent if manifest.source_path else Path.cwd()
    band_path = folder / "band.json"
    binding_path = folder / "band-binding.json"
    if band_path.exists():
        if not binding_path.exists() or read_json(binding_path)["band"] != file_record(band_path):
            raise ValueError("Fitted-band provenance changed or is incomplete; choose a fresh output location.")
        band = load_band(band_path)
    elif "artifact" in specification:
        source = Path(specification["artifact"])
        source = source if source.is_absolute() else base/source
        band = load_band(source)
        save_band(band, band_path)
        write_json_atomic(binding_path, {"band": file_record(band_path), "source": file_record(source)})
    else:
        setting = prepare_setting(setting_spec, base_path=base)
        settings = dict(specification)
        if settings["method"] == "support_grid":
            if setting.provenance.get("quantity") != settings["quantity"]:
                raise ValueError("Support band quantity must match the explicitly prepared Generali quantity.")
            band = fit_support_band(setting.X, setting.arrays["action_grid"],
                setting.arrays["predictor_grid"], settings, support=setting.arrays.get("support"),
                support_width=setting.arrays.get("support_width"),
                acceptance_grid=setting.arrays.get("acceptance_grid"), row_indices=setting.arrays.get("row_indices"))
        else:
            multipliers = setting.multipliers
            if settings["method"] == "residual_wild":
                count = settings["B"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 2 or multipliers is None or count > len(multipliers):
                    raise ValueError("B must select at least two available saved/paired multiplier rows.")
                multipliers = multipliers[:count]
            band = fit_band(setting.observations, setting.predictor, TargetDomain(setting.X),
                            settings, multipliers=multipliers)
        band.metadata["setting"] = setting.provenance
        save_band(band, band_path)
        write_json_atomic(binding_path, {"band": file_record(band_path)})

    # The saved fitted state is sufficient to reproduce these endpoint outputs.
    if "actions" in band.state:
        X, actions = band.state["X"], band.state["actions"]
    else:
        contexts, grid = band.state["X"], band.state["action_grid"]
        X, actions = np.repeat(contexts, len(grid), axis=0), np.tile(grid, len(contexts))
    evaluated = band.evaluate(X, actions)
    evaluation_path = folder / "band-evaluation.npz"
    if not evaluation_path.exists():
        save_arrays_atomic(evaluation_path, X=X, actions=actions, center=evaluated.center,
                           lower=evaluated.lower, upper=evaluated.upper, **evaluated.components)
    band_reference = file_record(band_path)
    write_json_atomic(folder / "band-summary.json", {"band": band_reference,
        "metadata": band.metadata, "evaluation": file_record(evaluation_path),
        "mean_lower_radius": float(np.mean(evaluated.center-evaluated.lower)),
        "mean_upper_radius": float(np.mean(evaluated.upper-evaluated.center))})
    if manifest.task == "band":
        return 1

    options.pop("plot", None)
    n_runs = 0
    for seed in manifest.seeds.run_seeds:
        summary = folder/f"summary-seed-{seed}.json"
        seeds = replicate_seed_setup(seed, manifest.seeds.anchor_seed,
                                     vary=manifest.seeds.vary, fixed=manifest.seeds.fixed)
        config = get_config(manifest.base_preset, {**options, "fitted_band": band,
            "band_artifact": band_reference, "seed_setup": seeds, "plot": False})
        context = create_run_context(variant.name, run_dir=folder/"seeds"/f"seed-{seed}",
            run_metadata={"preset_name": manifest.base_preset, "run_seed": seed,
                          "variant_name": variant.name, "overrides": variant.overrides,
                          "band": band_reference, "setting": band.metadata.get("setting")})
        if summary.exists():
            # Resume output-only work from exact saved optimizer results.
            parameters = {name: np.asarray(item["theta"]) for name, item in read_json(summary)["estimators"].items()}
        else:
            executed = execute_experiment_run(variant.name, config, run_context=context,
                reporter_stack_factory=_seed_reporter_stack_factory(folder, seed, per_seed_plots=False))
            parameters = {name: result.theta for name, result in executed.result.results.items()}
            n_runs += 1
        records = []
        for estimator, theta in parameters.items():
            actions = config.objective.policy.value(theta, config.x_fixed)
            evaluated = band.evaluate(band.state["X"], actions)
            arrays = dict(X=band.state["X"], theta=theta, actions=actions,
                          center=evaluated.center, lower=evaluated.lower, upper=evaluated.upper)
            path = context.run_dir / f"band-policy-{estimator}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                        raise ValueError("Saved policy endpoints changed; preserve the run and use a fresh location.")
            else:
                save_arrays_atomic(path, **arrays)
            records.append(file_record(path))
        write_json_atomic(folder/f"band-policy-seed-{seed}.json",
                          {"summary": file_record(summary), "evaluations": records})
    return n_runs
