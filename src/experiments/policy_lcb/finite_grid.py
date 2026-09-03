"""Exact finite-grid experiments for variable lower-confidence envelopes.

The public evaluation functions in this module are intentionally free of file-system
state.  A seed owns one standardized Gaussian vector, and every experimental
condition is a deterministic transformation of that vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    ORACLE_TOLERANCE,
    PolicyLCBLaunchSpec,
    gaussian_lcb_quantile,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
    sample_std,
    wilson_interval,
    write_json_atomic,
)
from experiments.seeds import derive_seed, rng_from_seed
from experiments.sweep_reporting import write_rows_csv


CalibrationKind = Literal["bonferroni_two_sided", "pointwise_two_sided"]
SelectorName = Literal["nominal", "uniform_lcb", "variable_lcb"]


@dataclass(frozen=True)
class FiniteGridSpec:
    """A sorted, inclusive, equally spaced one-dimensional grid."""

    lower: float
    upper: float
    count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("grid bounds must be finite.")
        if self.lower >= self.upper:
            raise ValueError("grid.lower must be smaller than grid.upper.")
        if int(self.count) < 2:
            raise ValueError("grid.count must be at least two.")

    def values(self) -> np.ndarray:
        return np.linspace(float(self.lower), float(self.upper), int(self.count))


@dataclass(frozen=True)
class ConcaveQuadraticSpec:
    """Parameters of ``linear*x - quadratic*x**2``."""

    linear: float
    quadratic: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.linear):
            raise ValueError("true_value.linear must be finite.")
        if not np.isfinite(self.quadratic) or self.quadratic <= 0.0:
            raise ValueError("true_value.quadratic must be finite and positive.")


@dataclass(frozen=True)
class ClippedDistanceRampSpec:
    """Variable uncertainty whose minimum is attained at each configured center."""

    centers: tuple[float, ...]
    minimum: float
    maximum: float
    ramp_radius: float

    def __post_init__(self) -> None:
        centers = tuple(float(value) for value in self.centers)
        if not centers or any(not np.isfinite(value) for value in centers):
            raise ValueError("uncertainty.centers must be a non-empty finite sequence.")
        if len(set(centers)) != len(centers):
            raise ValueError("uncertainty.centers must be unique.")
        if any(right <= left for left, right in zip(centers, centers[1:])):
            raise ValueError("uncertainty.centers must be strictly increasing.")
        if not np.isfinite(self.minimum) or self.minimum <= 0.0:
            raise ValueError("uncertainty.minimum must be finite and positive.")
        if not np.isfinite(self.maximum) or self.maximum < self.minimum:
            raise ValueError("uncertainty.maximum must be finite and at least minimum.")
        if not np.isfinite(self.ramp_radius) or self.ramp_radius <= 0.0:
            raise ValueError("uncertainty.ramp_radius must be finite and positive.")
        object.__setattr__(self, "centers", centers)


@dataclass(frozen=True)
class EnvelopeCalibrationSpec:
    """A named Gaussian envelope calibration."""

    name: str
    kind: CalibrationKind

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("confidence calibration names must be non-empty.")
        if self.kind not in {"bonferroni_two_sided", "pointwise_two_sided"}:
            raise ValueError(f"Unsupported confidence calibration type {self.kind!r}.")


@dataclass(frozen=True)
class VariableFiniteGridLCBSpec:
    """Resolved inputs for the paired variable-envelope experiment cube."""

    grid: FiniteGridSpec
    true_value: ConcaveQuadraticSpec
    uncertainty: ClippedDistanceRampSpec
    noise_scales: tuple[float, ...]
    delta: float
    calibrations: tuple[EnvelopeCalibrationSpec, ...]
    master_noise_seed: int
    run_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        noise_scales = tuple(float(value) for value in self.noise_scales)
        if not noise_scales or any(not np.isfinite(value) or value < 0.0 for value in noise_scales):
            raise ValueError("surrogate.noise_scales must be finite and non-negative.")
        if len(set(noise_scales)) != len(noise_scales):
            raise ValueError("surrogate.noise_scales must be unique.")
        if any(right <= left for left, right in zip(noise_scales, noise_scales[1:])):
            raise ValueError("surrogate.noise_scales must be strictly increasing.")
        if not np.isfinite(self.delta) or not 0.0 < self.delta < 1.0:
            raise ValueError("confidence.delta must lie in (0, 1).")
        if not self.calibrations:
            raise ValueError("confidence.calibrations must be non-empty.")
        names = [item.name for item in self.calibrations]
        kinds = [item.kind for item in self.calibrations]
        if len(set(names)) != len(names):
            raise ValueError("confidence calibration names must be unique.")
        if len(set(kinds)) != len(kinds):
            raise ValueError("confidence calibration types must be unique.")
        if set(kinds) != {"bonferroni_two_sided", "pointwise_two_sided"}:
            raise ValueError("confidence.calibrations must contain Bonferroni and pointwise types.")
        if int(self.master_noise_seed) < 0:
            raise ValueError("seeds.master_noise_seed must be non-negative.")
        run_seeds = tuple(int(seed) for seed in self.run_seeds)
        if not run_seeds or any(seed < 0 for seed in run_seeds):
            raise ValueError("seeds.run_seeds must resolve to non-negative integers.")
        if len(set(run_seeds)) != len(run_seeds):
            raise ValueError("seeds.run_seeds must be unique.")
        grid = self.grid.values()
        if any(center < grid[0] or center > grid[-1] for center in self.uncertainty.centers):
            raise ValueError("uncertainty.centers must lie within the grid bounds.")
        object.__setattr__(self, "noise_scales", noise_scales)
        object.__setattr__(self, "run_seeds", run_seeds)


@dataclass(frozen=True)
class VariableFiniteGridLCBLaunchSpec(PolicyLCBLaunchSpec):
    """Launch settings for the variable finite-grid adapter."""


@dataclass(frozen=True)
class VariableFiniteGridLCBManifest:
    """Resolved ``finite_grid_variable_lcb`` manifest."""

    name: str
    spec: VariableFiniteGridLCBSpec
    launch: VariableFiniteGridLCBLaunchSpec
    per_seed_plots: bool = False
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / path_part(self.name)

    def seed_result_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / "seeds" / f"seed-{run_seed}" / "result.json"


@dataclass(frozen=True)
class GridConditionResult:
    """Validity, tightness, and certificate diagnostics for one condition."""

    run_seed: int
    noise_seed: int
    noise_scale: float
    uncertainty_center: float
    calibration: str
    calibration_type: CalibrationKind
    delta: float
    quantile: float
    optimum_x: float
    optimum_index: int
    deterministic_target_x: float
    deterministic_target_index: int
    simultaneous_coverage: bool
    fraction_covered: float
    average_half_width: float
    maximum_half_width: float
    optimum_half_width: float
    average_full_width: float
    maximum_full_width: float
    optimum_full_width: float
    optimum_lower_bound_gap: float
    lcb_regret_certificate: float
    variable_lcb_regret: float
    certificate_slack: float
    certificate_event_holds: bool


@dataclass(frozen=True)
class GridSelectorResult:
    """Selection and optimizer-protection diagnostics for one selector."""

    run_seed: int
    noise_seed: int
    noise_scale: float
    uncertainty_center: float
    calibration: str
    calibration_type: CalibrationKind
    selector: SelectorName
    selected_index: int
    selected_x: float
    selected_true_value: float
    selected_surrogate_value: float
    selected_objective_value: float
    regret: float
    distance_to_optimum: float
    distance_to_uncertainty_center: float
    selected_point_covered: bool


@dataclass(frozen=True)
class VariableFiniteGridLCBSeedResult:
    """Every paired condition evaluated from one replayable Gaussian vector."""

    run_seed: int
    noise_seed: int
    z: tuple[float, ...]
    conditions: tuple[GridConditionResult, ...]
    selectors: tuple[GridSelectorResult, ...]


def clipped_distance_uncertainty(
    x: Sequence[float] | np.ndarray,
    center: float,
    uncertainty: ClippedDistanceRampSpec,
) -> np.ndarray:
    """Evaluate the clipped distance-ramp standard-deviation profile."""
    x_arr = np.asarray(x, dtype=float)
    fraction = np.minimum(np.abs(x_arr - float(center)) / uncertainty.ramp_radius, 1.0)
    return uncertainty.minimum + (uncertainty.maximum - uncertainty.minimum) * fraction


def true_objective(
    x: Sequence[float] | np.ndarray,
    objective: ConcaveQuadraticSpec,
) -> np.ndarray:
    """Evaluate the concave quadratic objective."""
    x_arr = np.asarray(x, dtype=float)
    return objective.linear * x_arr - objective.quadratic * x_arr**2


def calibration_quantile(calibration: EnvelopeCalibrationSpec, delta: float, count: int) -> float:
    """Return the two-sided Gaussian quantile for one calibration."""
    multiplicity = count if calibration.kind == "bonferroni_two_sided" else 1
    return gaussian_lcb_quantile(delta, multiplicity)


def variable_grid_noise_seed(spec: VariableFiniteGridLCBSpec, run_seed: int) -> int:
    """Derive a seed using only the master seed and run seed."""
    if int(run_seed) not in spec.run_seeds:
        raise ValueError(f"Unknown run seed {run_seed}.")
    return derive_seed(
        spec.master_noise_seed,
        f"finite-grid-variable-lcb:run-seed:{int(run_seed)}",
    )


def evaluate_variable_finite_grid_lcb_seed(
    spec: VariableFiniteGridLCBSpec,
    run_seed: int,
) -> VariableFiniteGridLCBSeedResult:
    """Draw one standardized Gaussian vector and evaluate the full paired cube."""
    noise_seed = variable_grid_noise_seed(spec, run_seed)
    z = rng_from_seed(noise_seed).normal(size=spec.grid.count)
    return evaluate_variable_finite_grid_lcb_draw(
        spec,
        run_seed=run_seed,
        noise_seed=noise_seed,
        z=z,
    )


def evaluate_variable_finite_grid_lcb_draw(
    spec: VariableFiniteGridLCBSpec,
    *,
    run_seed: int,
    noise_seed: int,
    z: Sequence[float],
) -> VariableFiniteGridLCBSeedResult:
    """Evaluate every center, noise scale, calibration, and selector for ``z``."""
    grid = spec.grid.values()
    z_arr = np.asarray(z, dtype=float)
    if z_arr.shape != grid.shape or not np.all(np.isfinite(z_arr)):
        raise ValueError("z must be finite and have one entry per grid point.")
    values = true_objective(grid, spec.true_value)
    optimum_index = int(np.argmax(values))
    optimum_x = float(grid[optimum_index])
    optimum_value = float(values[optimum_index])
    conditions: list[GridConditionResult] = []
    selectors: list[GridSelectorResult] = []

    for center in spec.uncertainty.centers:
        sigma = clipped_distance_uncertainty(grid, center, spec.uncertainty)
        for noise_scale in spec.noise_scales:
            surrogate = values + noise_scale * sigma * z_arr
            nominal_index = int(np.argmax(surrogate))
            for calibration in spec.calibrations:
                quantile = calibration_quantile(calibration, spec.delta, grid.size)
                half_width = noise_scale * quantile * sigma
                covered = np.abs(surrogate - values) <= half_width + ORACLE_TOLERANCE
                variable_objective = surrogate - half_width
                uniform_objective = surrogate - float(np.max(half_width))
                variable_index = int(np.argmax(variable_objective))
                uniform_index = int(np.argmax(uniform_objective))
                deterministic_index = int(np.argmax(values - half_width))
                if uniform_index != nominal_index:
                    raise AssertionError("A constant uniform envelope changed the nominal argmax.")
                variable_regret = optimum_value - float(values[variable_index])
                certificate = float(2.0 * half_width[optimum_index])
                slack = certificate - variable_regret
                conditions.append(
                    GridConditionResult(
                        run_seed=int(run_seed),
                        noise_seed=int(noise_seed),
                        noise_scale=float(noise_scale),
                        uncertainty_center=float(center),
                        calibration=calibration.name,
                        calibration_type=calibration.kind,
                        delta=float(spec.delta),
                        quantile=quantile,
                        optimum_x=optimum_x,
                        optimum_index=optimum_index,
                        deterministic_target_x=float(grid[deterministic_index]),
                        deterministic_target_index=deterministic_index,
                        simultaneous_coverage=bool(np.all(covered)),
                        fraction_covered=float(np.mean(covered)),
                        average_half_width=float(np.mean(half_width)),
                        maximum_half_width=float(np.max(half_width)),
                        optimum_half_width=float(half_width[optimum_index]),
                        average_full_width=float(2.0 * np.mean(half_width)),
                        maximum_full_width=float(2.0 * np.max(half_width)),
                        optimum_full_width=certificate,
                        optimum_lower_bound_gap=float(
                            optimum_value - variable_objective[optimum_index]
                        ),
                        lcb_regret_certificate=certificate,
                        variable_lcb_regret=variable_regret,
                        certificate_slack=slack,
                        certificate_event_holds=bool(
                            np.all(covered) and slack >= -ORACLE_TOLERANCE
                        ),
                    )
                )
                for selector, selected_index, objective_values in (
                    ("nominal", nominal_index, surrogate),
                    ("uniform_lcb", uniform_index, uniform_objective),
                    ("variable_lcb", variable_index, variable_objective),
                ):
                    selectors.append(
                        GridSelectorResult(
                            run_seed=int(run_seed),
                            noise_seed=int(noise_seed),
                            noise_scale=float(noise_scale),
                            uncertainty_center=float(center),
                            calibration=calibration.name,
                            calibration_type=calibration.kind,
                            selector=selector,  # type: ignore[arg-type]
                            selected_index=selected_index,
                            selected_x=float(grid[selected_index]),
                            selected_true_value=float(values[selected_index]),
                            selected_surrogate_value=float(surrogate[selected_index]),
                            selected_objective_value=float(objective_values[selected_index]),
                            regret=float(optimum_value - values[selected_index]),
                            distance_to_optimum=float(abs(grid[selected_index] - optimum_x)),
                            distance_to_uncertainty_center=float(abs(grid[selected_index] - center)),
                            selected_point_covered=bool(covered[selected_index]),
                        )
                    )
    return VariableFiniteGridLCBSeedResult(
        run_seed=int(run_seed),
        noise_seed=int(noise_seed),
        z=tuple(float(value) for value in z_arr),
        conditions=tuple(conditions),
        selectors=tuple(selectors),
    )


def load_variable_finite_grid_lcb_manifest(
    path: str | Path,
) -> VariableFiniteGridLCBManifest:
    """Load and validate a ``kind=finite_grid_variable_lcb`` JSON manifest."""
    manifest_path = Path(path)
    return parse_variable_finite_grid_lcb_manifest(
        read_json(manifest_path), source_path=manifest_path
    )


def parse_variable_finite_grid_lcb_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> VariableFiniteGridLCBManifest:
    """Resolve and validate the variable finite-grid manifest interface."""
    if not isinstance(payload, Mapping):
        raise ValueError("Variable finite-grid LCB manifest must be a JSON object.")
    if payload.get("kind") != "finite_grid_variable_lcb":
        raise ValueError("Manifest kind must be 'finite_grid_variable_lcb'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Manifest name must be non-empty.")

    grid_payload = required_mapping(payload, "grid")
    if grid_payload.get("type") != "linspace":
        raise ValueError("grid.type must be 'linspace'.")
    objective_payload = required_mapping(payload, "true_value")
    if objective_payload.get("type") != "concave_quadratic":
        raise ValueError("true_value.type must be 'concave_quadratic'.")
    uncertainty_payload = required_mapping(payload, "uncertainty")
    if uncertainty_payload.get("type") != "clipped_distance_ramp":
        raise ValueError("uncertainty.type must be 'clipped_distance_ramp'.")
    surrogate_payload = required_mapping(payload, "surrogate")
    if surrogate_payload.get("type") != "independent_gaussian":
        raise ValueError("surrogate.type must be 'independent_gaussian'.")
    confidence_payload = required_mapping(payload, "confidence")
    seeds_payload = required_mapping(payload, "seeds")
    run_seeds_payload = required_mapping(seeds_payload, "run_seeds")
    if run_seeds_payload.get("type") != "range":
        raise ValueError("seeds.run_seeds.type must be 'range'.")
    start = int(run_seeds_payload.get("start", -1))
    count = int(run_seeds_payload.get("count", 0))
    if start < 0 or count <= 0:
        raise ValueError("seeds.run_seeds start must be non-negative and count positive.")

    calibration_payloads = confidence_payload.get("calibrations")
    if not isinstance(calibration_payloads, Sequence) or isinstance(
        calibration_payloads, (str, bytes)
    ):
        raise ValueError("confidence.calibrations must be a sequence.")
    calibrations: list[EnvelopeCalibrationSpec] = []
    for item in calibration_payloads:
        if not isinstance(item, Mapping):
            raise ValueError("Each confidence calibration must be an object.")
        calibrations.append(
            EnvelopeCalibrationSpec(
                name=str(item.get("name") or ""),
                kind=str(item.get("type") or ""),  # type: ignore[arg-type]
            )
        )

    launch_payload = required_mapping(payload, "launch")
    mode = str(launch_payload.get("mode") or "")
    array = str(launch_payload.get("array") or "")
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("launch.mode must be auto, local, or slurm.")
    if array not in {"none", "seed"}:
        raise ValueError("launch.array must be none or seed.")
    parallel_raw = launch_payload.get("array_max_parallel")
    array_max_parallel = None if parallel_raw is None else int(parallel_raw)
    if array_max_parallel is not None and array_max_parallel <= 0:
        raise ValueError("launch.array_max_parallel must be positive when provided.")

    spec = VariableFiniteGridLCBSpec(
        grid=FiniteGridSpec(
            lower=float(grid_payload.get("lower", np.nan)),
            upper=float(grid_payload.get("upper", np.nan)),
            count=int(grid_payload.get("count", 0)),
        ),
        true_value=ConcaveQuadraticSpec(
            linear=float(objective_payload.get("linear", np.nan)),
            quadratic=float(objective_payload.get("quadratic", np.nan)),
        ),
        uncertainty=ClippedDistanceRampSpec(
            centers=number_sequence(uncertainty_payload.get("centers"), "uncertainty.centers"),
            minimum=float(uncertainty_payload.get("minimum", np.nan)),
            maximum=float(uncertainty_payload.get("maximum", np.nan)),
            ramp_radius=float(uncertainty_payload.get("ramp_radius", np.nan)),
        ),
        noise_scales=number_sequence(
            surrogate_payload.get("noise_scales"), "surrogate.noise_scales"
        ),
        delta=float(confidence_payload.get("delta", np.nan)),
        calibrations=tuple(calibrations),
        master_noise_seed=int(seeds_payload.get("master_noise_seed", -1)),
        run_seeds=tuple(range(start, start + count)),
    )
    per_seed_plots = payload.get("per_seed_plots", False)
    if not isinstance(per_seed_plots, bool):
        raise ValueError("per_seed_plots must be boolean.")
    if per_seed_plots:
        raise ValueError("per_seed_plots=true is not supported by this aggregate-report adapter.")
    return VariableFiniteGridLCBManifest(
        name=name,
        spec=spec,
        launch=VariableFiniteGridLCBLaunchSpec(
            mode=mode,  # type: ignore[arg-type]
            array=array,  # type: ignore[arg-type]
            array_max_parallel=array_max_parallel,
        ),
        per_seed_plots=per_seed_plots,
        source_path=None if source_path is None else Path(source_path),
    )


def variable_finite_grid_lcb_seed_complete(
    manifest: VariableFiniteGridLCBManifest,
    run_seed: int,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return whether one durable seed result already exists."""
    return manifest.seed_result_path(run_seed, runs_root).exists()


def run_variable_finite_grid_lcb_manifest_seed(
    manifest: VariableFiniteGridLCBManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run one array task and atomically persist its replayable result."""
    if index < 0 or index >= len(manifest.spec.run_seeds):
        raise IndexError(f"Seed task index {index} is out of range.")
    run_seed = manifest.spec.run_seeds[index]
    write_variable_finite_grid_lcb_experiment_readme(manifest, runs_root=runs_root)
    path = manifest.seed_result_path(run_seed, runs_root)
    if path.exists() and not force:
        return {
            "project_dir": str(manifest.project_dir(runs_root)),
            "run_seed": run_seed,
            "skipped": True,
            "n_condition_rows": 0,
            "n_selector_rows": 0,
        }
    result = evaluate_variable_finite_grid_lcb_seed(manifest.spec, run_seed)
    write_json_atomic(path, asdict(result))
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "run_seed": run_seed,
        "skipped": False,
        "n_condition_rows": len(result.conditions),
        "n_selector_rows": len(result.selectors),
    }


def run_variable_finite_grid_lcb_manifest_serial(
    manifest: VariableFiniteGridLCBManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run every seed serially, then produce aggregate artifacts."""
    payloads = [
        run_variable_finite_grid_lcb_manifest_seed(
            manifest, index, runs_root=runs_root, force=force
        )
        for index in range(len(manifest.spec.run_seeds))
    ]
    collected = collect_variable_finite_grid_lcb_outputs(manifest, runs_root=runs_root)
    return {
        **collected,
        "n_executed_condition_rows": sum(
            int(payload["n_condition_rows"]) for payload in payloads
        ),
        "n_executed_selector_rows": sum(
            int(payload["n_selector_rows"]) for payload in payloads
        ),
        "n_skipped_seeds": sum(bool(payload["skipped"]) for payload in payloads),
    }


def collect_variable_finite_grid_lcb_outputs(
    manifest: VariableFiniteGridLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, object]:
    """Aggregate all expected seed JSONs into raw tables, summaries, and plots."""
    project_dir = manifest.project_dir(runs_root)
    write_variable_finite_grid_lcb_experiment_readme(manifest, runs_root=runs_root)
    results = [
        _read_variable_seed_result(manifest.seed_result_path(seed, runs_root))
        for seed in manifest.spec.run_seeds
    ]
    condition_rows = [asdict(row) for result in results for row in result.conditions]
    selector_rows = [asdict(row) for result in results for row in result.selectors]
    comparison_rows = _comparison_rows(condition_rows, selector_rows)

    simultaneous = [
        row for row in comparison_rows if row["calibration_type"] == "bonferroni_two_sided"
    ]
    experiment_1 = _aggregate_rows(
        simultaneous,
        group_fields=("uncertainty_center", "noise_scale"),
        numeric_fields=(
            "fraction_covered",
            "nominal_regret",
            "variable_lcb_regret",
            "nominal_minus_variable_regret",
            "nominal_selected_x",
            "variable_lcb_selected_x",
            "nominal_distance_to_optimum",
            "variable_lcb_distance_to_optimum",
            "lcb_regret_certificate",
            "certificate_slack",
        ),
        boolean_fields=(
            "simultaneous_coverage",
            "nominal_selected_point_covered",
            "variable_lcb_selected_point_covered",
            "certificate_event_holds",
        ),
    )
    experiment_2 = _aggregate_rows(
        comparison_rows,
        group_fields=("uncertainty_center", "noise_scale", "calibration", "calibration_type"),
        numeric_fields=(
            "fraction_covered",
            "nominal_regret",
            "variable_lcb_regret",
            "nominal_minus_variable_regret",
        ),
        boolean_fields=(
            "simultaneous_coverage",
            "nominal_selected_point_covered",
            "variable_lcb_selected_point_covered",
        ),
    )
    experiment_3 = _aggregate_rows(
        simultaneous,
        group_fields=("uncertainty_center", "noise_scale"),
        numeric_fields=(
            "nominal_regret",
            "uniform_lcb_regret",
            "variable_lcb_regret",
            "nominal_selected_x",
            "uniform_lcb_selected_x",
            "variable_lcb_selected_x",
            "average_half_width",
            "maximum_half_width",
            "optimum_half_width",
            "average_full_width",
            "maximum_full_width",
            "optimum_full_width",
            "optimum_lower_bound_gap",
            "nominal_minus_variable_regret",
        ),
        boolean_fields=(),
    )
    experiment_4 = _aggregate_rows(
        simultaneous,
        group_fields=("uncertainty_center", "noise_scale"),
        numeric_fields=(
            "optimum_x",
            "deterministic_target_x",
            "nominal_selected_x",
            "uniform_lcb_selected_x",
            "variable_lcb_selected_x",
            "nominal_regret",
            "uniform_lcb_regret",
            "variable_lcb_regret",
            "nominal_minus_variable_regret",
            "nominal_distance_to_uncertainty_center",
            "variable_lcb_distance_to_uncertainty_center",
        ),
        boolean_fields=(),
    )

    write_rows_csv(
        project_dir / "seed_condition_metrics.csv",
        condition_rows,
        tuple(GridConditionResult.__dataclass_fields__),
    )
    write_rows_csv(
        project_dir / "seed_selector_metrics.csv",
        selector_rows,
        tuple(GridSelectorResult.__dataclass_fields__),
    )
    summaries = {
        "experiment_1_noise_scale_summary.csv": experiment_1,
        "experiment_2_calibration_summary.csv": experiment_2,
        "experiment_3_envelope_shape_summary.csv": experiment_3,
        "experiment_4_center_summary.csv": experiment_4,
    }
    for filename, rows in summaries.items():
        write_rows_csv(project_dir / filename, rows, tuple(rows[0]))
    _write_variable_grid_plots(
        manifest.spec,
        experiment_1,
        experiment_2,
        experiment_3,
        experiment_4,
        results[0],
        project_dir / "plots",
    )
    return {
        "project_dir": str(project_dir),
        "n_seed_results": len(results),
        "n_condition_rows": len(condition_rows),
        "n_selector_rows": len(selector_rows),
    }


def write_variable_finite_grid_lcb_experiment_readme(
    manifest: VariableFiniteGridLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write the resolved formulas and pairing contract beside the outputs."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    source = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    spec = manifest.spec
    text = f"""# {manifest.name}

- Manifest source: `{source}`
- Grid: `linspace({spec.grid.lower}, {spec.grid.upper}, {spec.grid.count})`
- Objective: `{spec.true_value.linear} x - {spec.true_value.quadratic} x^2`
- Noise scales: `{list(spec.noise_scales)}`
- Uncertainty centers: `{list(spec.uncertainty.centers)}`
- Envelope failure probability: `{spec.delta}`
- Master noise seed: `{spec.master_noise_seed}`
- Run seeds: `{spec.run_seeds[0]}..{spec.run_seeds[-1]}` ({len(spec.run_seeds)} total)
- Launch: `{manifest.launch.mode}` / `{manifest.launch.array}`

Each seed draws one independent standard-normal vector `Z[x]`. That exact vector is
reused across every uncertainty center, noise scale, calibration, and selector. The
uncertainty center `m` is the minimum-uncertainty point; it is distinct from the
true maximizer `x*` and the deterministic penalized target `x_dagger`.

```text
f(x) = linear*x - quadratic*x^2
sigma_m(x) = minimum + (maximum-minimum)*min(|x-m|/ramp_radius, 1)
f_hat(x) = f(x) + c*sigma_m(x)*Z[x]
E(x) = c*q*sigma_m(x)
LCB(x) = f_hat(x) - E(x)
```

The simultaneous calibration uses the two-sided Bonferroni quantile
`Phi^-1(1-delta/(2K))`; the pointwise calibration uses `Phi^-1(1-delta/2)`.
For positive `c` and `sigma`, simultaneous coverage is exactly the same event
`all(|Z[x]| <= q)` for every center and scale, so no center multiplicity correction
is introduced. At `c=0`, coverage is deterministically perfect.

The uniform envelope is the constant `max_x E(x)`, hence its exact argmax equals
the nominal surrogate argmax. Ties are resolved by the lowest grid point. On the
simultaneous coverage event, variable-LCB regret is certified by `2*E(x*)`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _read_variable_seed_result(path: Path) -> VariableFiniteGridLCBSeedResult:
    if not path.exists():
        raise FileNotFoundError(f"Missing variable finite-grid LCB seed result: {path}")
    payload = read_json(path)
    return VariableFiniteGridLCBSeedResult(
        run_seed=int(payload["run_seed"]),
        noise_seed=int(payload["noise_seed"]),
        z=tuple(float(value) for value in payload["z"]),
        conditions=tuple(GridConditionResult(**row) for row in payload["conditions"]),
        selectors=tuple(GridSelectorResult(**row) for row in payload["selectors"]),
    )


def _comparison_rows(
    condition_rows: Sequence[Mapping[str, object]],
    selector_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join condition and selector diagnostics at the paired-seed grain."""
    key_fields = ("run_seed", "noise_scale", "uncertainty_center", "calibration")
    selector_lookup: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in selector_rows:
        key = tuple(row[field] for field in key_fields) + (row["selector"],)
        selector_lookup[key] = row
    joined: list[dict[str, object]] = []
    for condition in condition_rows:
        key = tuple(condition[field] for field in key_fields)
        row = dict(condition)
        for selector in ("nominal", "uniform_lcb", "variable_lcb"):
            selected = selector_lookup[key + (selector,)]
            for field in (
                "selected_x",
                "regret",
                "distance_to_optimum",
                "distance_to_uncertainty_center",
                "selected_point_covered",
            ):
                row[f"{selector}_{field}"] = selected[field]
        row["nominal_minus_variable_regret"] = float(row["nominal_regret"]) - float(
            row["variable_lcb_regret"]
        )
        joined.append(row)
    return joined


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: tuple[str, ...],
    numeric_fields: tuple[str, ...],
    boolean_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """Return deterministic mean/variability summaries over run seeds."""
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary: dict[str, object] = dict(zip(group_fields, key))
        summary["n_seeds"] = len(group)
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = sample_std(values)
            summary[f"{field}_q05"] = float(np.quantile(values, 0.05))
            summary[f"{field}_median"] = float(np.median(values))
            summary[f"{field}_q95"] = float(np.quantile(values, 0.95))
        for field in boolean_fields:
            count = sum(bool(row[field]) for row in group)
            low, high = wilson_interval(count, len(group))
            summary[f"{field}_count"] = count
            summary[f"{field}_rate"] = count / len(group)
            summary[f"{field}_wilson_95_low"] = low
            summary[f"{field}_wilson_95_high"] = high
        output.append(summary)
    return output


def _write_variable_grid_plots(
    spec: VariableFiniteGridLCBSpec,
    experiment_1: Sequence[Mapping[str, object]],
    experiment_2: Sequence[Mapping[str, object]],
    experiment_3: Sequence[Mapping[str, object]],
    experiment_4: Sequence[Mapping[str, object]],
    representative_result: VariableFiniteGridLCBSeedResult,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_noise_regret(spec, experiment_1, plots_dir / "experiment_1_noise_scale_regret.png")
    _plot_noise_validity(spec, experiment_1, plots_dir / "experiment_1_validity.png")
    _plot_realized_landscapes(
        spec,
        representative_result,
        plots_dir / "experiment_1_realized_landscapes.png",
    )
    _plot_calibration_coverage(
        spec, experiment_2, plots_dir / "experiment_2_calibration_coverage.png"
    )
    _plot_calibration_regret(
        spec, experiment_2, plots_dir / "experiment_2_calibration_regret.png"
    )
    _plot_calibration_regret_by_center(
        spec,
        experiment_2,
        plots_dir / "experiment_2_calibration_regret_by_center.png",
    )
    _plot_envelope_shape(
        spec, experiment_3, plots_dir / "experiment_3_envelope_shape_regret.png"
    )
    _plot_envelope_shape_by_center(
        spec,
        experiment_3,
        plots_dir / "experiment_3_envelope_shape_regret_by_center.png",
    )
    _plot_center_regret(spec, experiment_4, plots_dir / "experiment_4_center_regret.png")
    _plot_center_selection(spec, experiment_4, plots_dir / "experiment_4_center_selection.png")


def _plot_realized_landscapes(
    spec: VariableFiniteGridLCBSpec,
    result: VariableFiniteGridLCBSeedResult,
    path: Path,
) -> None:
    """Show one paired surrogate draw, confidence band, and selectors across ``c``."""
    plt = _load_pyplot()
    grid = spec.grid.values()
    values = true_objective(grid, spec.true_value)
    z = np.asarray(result.z, dtype=float)
    optimum_index = int(np.argmax(values))
    target_center = spec.grid.lower + 0.25 * (spec.grid.upper - spec.grid.lower)
    center = min(spec.uncertainty.centers, key=lambda item: abs(item - target_center))
    sigma = clipped_distance_uncertainty(grid, center, spec.uncertainty)
    calibration = next(
        item for item in spec.calibrations if item.kind == "bonferroni_two_sided"
    )
    quantile = calibration_quantile(calibration, spec.delta, spec.grid.count)

    columns = min(3, len(spec.noise_scales))
    row_count = int(np.ceil(len(spec.noise_scales) / columns))
    fig, raw_axes = plt.subplots(
        row_count,
        columns,
        figsize=(5.2 * columns, 3.9 * row_count),
        squeeze=False,
    )
    axes = list(raw_axes.flat)
    for axis in axes[len(spec.noise_scales) :]:
        axis.set_visible(False)
    fig.suptitle(
        "One paired surrogate realization as the noise/envelope scale changes\n"
        f"run seed {result.run_seed}; fixed minimum-uncertainty location m={center:g}; "
        "simultaneous Bonferroni envelope"
    )

    for axis, noise_scale in zip(axes, spec.noise_scales):
        surrogate = values + noise_scale * sigma * z
        half_width = noise_scale * quantile * sigma
        lower = surrogate - half_width
        upper = surrogate + half_width
        covered = np.abs(surrogate - values) <= half_width + ORACLE_TOLERANCE
        nominal_index = int(np.argmax(surrogate))
        variable_index = int(np.argmax(lower))

        axis.fill_between(
            grid,
            lower,
            upper,
            color="tab:blue",
            alpha=0.15,
            label=r"Envelope $\hat f\pm E$",
        )
        axis.plot(grid, values, color="black", linewidth=2.0, label=r"True $f$")
        axis.plot(
            grid,
            surrogate,
            color="tab:blue",
            linewidth=1.1,
            alpha=0.9,
            label=r"Surrogate $\hat f$",
        )
        axis.plot(
            grid,
            lower,
            color="tab:orange",
            linewidth=1.5,
            label=r"LCB $\hat f-E$",
        )
        axis.axvline(center, color="0.45", linestyle=":", linewidth=1.3, label=r"$m$")
        axis.axvline(
            grid[optimum_index],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$x^*$",
        )
        axis.scatter(
            [grid[nominal_index]],
            [surrogate[nominal_index]],
            color="tab:blue",
            edgecolor="white",
            linewidth=0.8,
            s=55,
            zorder=5,
            label=r"Nominal $\hat x$",
        )
        axis.scatter(
            [grid[variable_index]],
            [lower[variable_index]],
            color="tab:orange",
            edgecolor="white",
            marker="s",
            linewidth=0.8,
            s=50,
            zorder=5,
            label=r"LCB $\hat x$",
        )
        validity = "yes" if bool(np.all(covered)) else "no"
        axis.set_title(
            f"c={noise_scale:g}; max E={float(np.max(half_width)):.3g}; "
            f"whole-grid valid: {validity}"
        )
        axis.set_xlabel("Grid point x")
        axis.set_ylabel("Objective value / bound")
        axis.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.018,
        r"Shading is the two-sided band $[\hat f-E,\hat f+E]$; its lower edge is the LCB. "
        r"Nominal maximizes $\hat f$; variable LCB maximizes $\hat f-E$.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.86))
    _save_figure(fig, path)


def _plot_noise_regret(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fig, axes = _center_facets(rows, "True regret versus noise/envelope scale c")
    for axis, center in axes:
        group = _rows_for(rows, uncertainty_center=center)
        x = [float(row["noise_scale"]) for row in group]
        axis.plot(x, [float(row["nominal_regret_mean"]) for row in group], marker="o", label="Nominal")
        axis.plot(x, [float(row["variable_lcb_regret_mean"]) for row in group], marker="s", label="Variable LCB")
        axis.set_title(f"m={center:g}")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        axes,
        "Noise/envelope scale c",
        "Mean true regret R",
        path,
        note=f"{_regret_plot_note(spec)}\n{_geometry_plot_note(spec)}",
    )


def _plot_noise_validity(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fig, axes = _center_facets(rows, "Bonferroni validity versus noise/envelope scale c")
    for axis, center in axes:
        group = _rows_for(rows, uncertainty_center=center)
        x = [float(row["noise_scale"]) for row in group]
        axis.plot(x, [float(row["simultaneous_coverage_rate"]) for row in group], marker="o", label="Simultaneous")
        axis.plot(x, [float(row["fraction_covered_mean"]) for row in group], marker="s", label="Fraction covered")
        axis.set_title(f"m={center:g}")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        axes,
        "Noise/envelope scale c",
        "Coverage",
        path,
        note=_geometry_plot_note(spec),
    )


def _plot_calibration_coverage(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    plt = _load_pyplot()
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    for calibration in sorted({str(row["calibration"]) for row in rows}):
        group = [row for row in rows if row["calibration"] == calibration]
        centers = sorted({float(row["uncertainty_center"]) for row in group})
        values = [np.mean([float(row["simultaneous_coverage_rate"]) for row in group if float(row["uncertainty_center"]) == center]) for center in centers]
        axis.plot(centers, values, marker="o", label=calibration)
    axis.set(
        xlabel="Minimum-uncertainty location m",
        ylabel="Simultaneous coverage",
        ylim=(0.0, 1.05),
        title="Calibration validity (averaged over configured c values)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_ylim(bottom=0.0)
    _finish_single_figure(fig, path, note=_geometry_plot_note(spec))


def _plot_calibration_regret(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    plt = _load_pyplot()
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    for calibration in sorted({str(row["calibration"]) for row in rows}):
        group = [row for row in rows if row["calibration"] == calibration]
        scales = sorted({float(row["noise_scale"]) for row in group})
        values = [np.mean([float(row["variable_lcb_regret_mean"]) for row in group if float(row["noise_scale"]) == scale]) for scale in scales]
        axis.plot(scales, values, marker="o", label=calibration)
    axis.set(
        xlabel="Noise/envelope scale c",
        ylabel="Mean true regret R",
        title="LCB regret by calibration (averaged over m)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_ylim(bottom=0.0)
    _finish_single_figure(
        fig,
        path,
        note=f"{_regret_plot_note(spec)}\n{_geometry_plot_note(spec)}",
    )


def _plot_calibration_regret_by_center(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Compare nominal and calibrated LCB regret separately for every center."""
    fig, facets = _center_facets(
        rows, "Experiment 2: calibration regret by minimum-uncertainty location m"
    )
    styles = (
        ("Nominal (no envelope)", "nominal_regret", "black", "o", "--"),
        ("Pointwise LCB", "variable_lcb_regret", "tab:orange", "s", "-"),
        ("Simultaneous LCB", "variable_lcb_regret", "tab:green", "^", "-"),
    )
    for axis, center in facets:
        center_rows = _rows_for(rows, uncertainty_center=center)
        pointwise = [
            row
            for row in center_rows
            if row["calibration_type"] == "pointwise_two_sided"
        ]
        simultaneous = [
            row
            for row in center_rows
            if row["calibration_type"] == "bonferroni_two_sided"
        ]
        for label, metric, color, marker, linestyle in styles:
            group = pointwise if label != "Simultaneous LCB" else simultaneous
            scales = np.asarray([float(row["noise_scale"]) for row in group])
            means = np.asarray([float(row[f"{metric}_mean"]) for row in group])
            q05 = np.asarray([float(row[f"{metric}_q05"]) for row in group])
            q95 = np.asarray([float(row[f"{metric}_q95"]) for row in group])
            axis.fill_between(scales, q05, q95, color=color, alpha=0.10)
            axis.plot(
                scales,
                means,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
            )
        axis.set_title(f"m={center:g}")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        facets,
        "Noise/envelope scale c",
        "Mean true regret R",
        path,
        note=(
            "Shading shows the 5th--95th seed percentiles.  "
            f"{_regret_plot_note(spec)}\n{_geometry_plot_note(spec)}"
        ),
    )


def _plot_envelope_shape(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    plt = _load_pyplot()
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    scales = sorted({float(row["noise_scale"]) for row in rows})
    for selector in ("nominal", "uniform_lcb", "variable_lcb"):
        values = [np.mean([float(row[f"{selector}_regret_mean"]) for row in rows if float(row["noise_scale"]) == scale]) for scale in scales]
        axis.plot(scales, values, marker="o", label=_selector_label(selector))
    axis.set(
        xlabel="Noise/envelope scale c",
        ylabel="Mean true regret R",
        title="Uniform versus variable Bonferroni envelope (averaged over m)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_ylim(bottom=0.0)
    _finish_single_figure(
        fig,
        path,
        note=f"{_regret_plot_note(spec)}\n{_geometry_plot_note(spec)}",
    )


def _plot_envelope_shape_by_center(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Show the effect of valid envelope geometry separately for every center."""
    fig, facets = _center_facets(
        rows, "Experiment 3: valid envelope geometry by minimum-uncertainty location m"
    )
    styles = (
        (
            "Nominal = Uniform LCB",
            "nominal_regret",
            "black",
            "o",
            "--",
        ),
        ("Variable LCB", "variable_lcb_regret", "tab:green", "s", "-"),
    )
    for axis, center in facets:
        group = _rows_for(rows, uncertainty_center=center)
        scales = np.asarray([float(row["noise_scale"]) for row in group])
        for label, metric, color, marker, linestyle in styles:
            means = np.asarray([float(row[f"{metric}_mean"]) for row in group])
            q05 = np.asarray([float(row[f"{metric}_q05"]) for row in group])
            q95 = np.asarray([float(row[f"{metric}_q95"]) for row in group])
            axis.fill_between(scales, q05, q95, color=color, alpha=0.10)
            axis.plot(
                scales,
                means,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
            )
        axis.set_title(f"m={center:g}")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        facets,
        "Noise/envelope scale c",
        "Mean true regret R",
        path,
        note=(
            "Both envelopes are Bonferroni-valid; the uniform penalty is constant and "
            "cannot change the nominal argmax. Shading shows 5th--95th seed percentiles.\n"
            f"{_regret_plot_note(spec)}  {_geometry_plot_note(spec)}"
        ),
    )


def _plot_center_regret(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fig, facets = _scale_facets(
        spec, "True regret versus minimum-uncertainty location m"
    )
    for axis, scale in facets:
        group = _rows_for(rows, noise_scale=scale)
        centers = [float(row["uncertainty_center"]) for row in group]
        for selector, marker in (("nominal", "o"), ("uniform_lcb", "^"), ("variable_lcb", "s")):
            axis.plot(
                centers,
                [float(row[f"{selector}_regret_mean"]) for row in group],
                marker=marker,
                label=_selector_label(selector),
            )
        axis.set_title(f"c={scale:g}")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        facets,
        "Minimum-uncertainty location m",
        "Mean true regret R",
        path,
        note=f"{_regret_plot_note(spec)}\n{_geometry_plot_note(spec)}",
    )


def _plot_center_selection(
    spec: VariableFiniteGridLCBSpec,
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    fig, facets = _scale_facets(
        spec, "Selected location versus minimum-uncertainty location m"
    )
    for axis, scale in facets:
        group = _rows_for(rows, noise_scale=scale)
        centers = [float(row["uncertainty_center"]) for row in group]
        axis.plot(centers, centers, color="0.45", linestyle=":", label="m")
        axis.plot(centers, [float(row["optimum_x_mean"]) for row in group], color="black", linestyle="--", label="x*")
        axis.plot(centers, [float(row["deterministic_target_x_mean"]) for row in group], marker="^", label="x dagger")
        axis.plot(centers, [float(row["variable_lcb_selected_x_mean"]) for row in group], marker="o", label="Variable LCB")
        axis.set_title(f"c={scale:g}")
        axis.set_ylim(spec.grid.lower - 0.03, spec.grid.upper + 0.03)
        axis.grid(alpha=0.25)
    _finish_facets(
        fig,
        facets,
        "Minimum-uncertainty location m",
        "Mean selected x",
        path,
        note=_geometry_plot_note(spec),
    )


def _center_facets(
    rows: Sequence[Mapping[str, object]], title: str
) -> tuple[Figure, list[tuple[Axes, float]]]:
    plt = _load_pyplot()
    centers = sorted({float(row["uncertainty_center"]) for row in rows})
    columns = min(4, len(centers))
    row_count = int(np.ceil(len(centers) / columns))
    fig, raw_axes = plt.subplots(row_count, columns, figsize=(4.2 * columns, 3.4 * row_count), squeeze=False)
    fig.suptitle(title)
    axes = list(raw_axes.flat)
    for axis in axes[len(centers):]:
        axis.set_visible(False)
    return fig, list(zip(axes, centers))


def _scale_facets(
    spec: VariableFiniteGridLCBSpec, title: str
) -> tuple[Figure, list[tuple[Axes, float]]]:
    plt = _load_pyplot()
    scales = list(spec.noise_scales)
    columns = min(3, len(scales))
    row_count = int(np.ceil(len(scales) / columns))
    fig, raw_axes = plt.subplots(row_count, columns, figsize=(4.8 * columns, 3.6 * row_count), squeeze=False)
    fig.suptitle(title)
    axes = list(raw_axes.flat)
    for axis in axes[len(scales):]:
        axis.set_visible(False)
    return fig, list(zip(axes, scales))


def _rows_for(
    rows: Sequence[Mapping[str, object]], **criteria: float
) -> list[Mapping[str, object]]:
    return sorted(
        [row for row in rows if all(float(row[key]) == value for key, value in criteria.items())],
        key=lambda row: (float(row.get("uncertainty_center", 0.0)), float(row.get("noise_scale", 0.0))),
    )


def _finish_facets(
    fig: Figure,
    facets: Sequence[tuple[Axes, float]],
    xlabel: str,
    ylabel: str,
    path: Path,
    *,
    note: str,
) -> None:
    for axis, _ in facets:
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    if facets:
        facets[0][0].legend(fontsize=8)
    fig.text(0.5, 0.022, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    _save_figure(fig, path)


def _finish_single_figure(fig: Figure, path: Path, *, note: str) -> None:
    fig.text(0.5, 0.022, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    _save_figure(fig, path)


def _regret_plot_note(spec: VariableFiniteGridLCBSpec) -> str:
    """Return the exact true-regret calculation displayed on regret plots."""
    grid = spec.grid.values()
    values = true_objective(grid, spec.true_value)
    optimum_index = int(np.argmax(values))
    optimum_x = float(grid[optimum_index])
    optimum_value = float(values[optimum_index])
    linear = spec.true_value.linear
    quadratic = spec.true_value.quadratic
    formula = (
        rf"$R(\hat{{x}})=f(x^\star)-f(\hat{{x}})="
        rf"{optimum_value:g}-[{linear:g}\hat{{x}}-{quadratic:g}\hat{{x}}^2]$"
    )
    if np.isclose(linear, 2.0 * quadratic * optimum_x) and np.isclose(
        optimum_value, quadratic * optimum_x**2
    ):
        formula += rf" $={quadratic:g}(\hat{{x}}-{optimum_x:g})^2$."
    return formula


def _geometry_plot_note(spec: VariableFiniteGridLCBSpec) -> str:
    """Define the center and scale axes displayed throughout the report."""
    return (
        rf"$m$: location of minimum uncertainty, $\sigma_m(m)="
        rf"{spec.uncertainty.minimum:g}$; "
        r"$c$: common scale in $\hat f=f+c\sigma_m Z$ and $E=cq\sigma_m$."
    )


def _selector_label(selector: str) -> str:
    return {
        "nominal": "Nominal",
        "uniform_lcb": "Uniform LCB",
        "variable_lcb": "Variable LCB",
    }[selector]


def _save_figure(fig: Figure, path: Path) -> None:
    plt = _load_pyplot()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_pyplot() -> Any:
    """Load the non-interactive plotting backend only when collection needs it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


__all__ = [
    "CalibrationKind",
    "ClippedDistanceRampSpec",
    "ConcaveQuadraticSpec",
    "EnvelopeCalibrationSpec",
    "FiniteGridSpec",
    "GridConditionResult",
    "GridSelectorResult",
    "SelectorName",
    "VariableFiniteGridLCBLaunchSpec",
    "VariableFiniteGridLCBManifest",
    "VariableFiniteGridLCBSeedResult",
    "VariableFiniteGridLCBSpec",
    "calibration_quantile",
    "clipped_distance_uncertainty",
    "collect_variable_finite_grid_lcb_outputs",
    "evaluate_variable_finite_grid_lcb_draw",
    "evaluate_variable_finite_grid_lcb_seed",
    "load_variable_finite_grid_lcb_manifest",
    "parse_variable_finite_grid_lcb_manifest",
    "run_variable_finite_grid_lcb_manifest_seed",
    "run_variable_finite_grid_lcb_manifest_serial",
    "true_objective",
    "variable_grid_noise_seed",
    "variable_finite_grid_lcb_seed_complete",
    "write_variable_finite_grid_lcb_experiment_readme",
]
