"""Exact finite-grid experiments for variable lower-confidence envelopes.

The public evaluation functions in this module are intentionally free of file-system
state.  A seed owns one standardized Gaussian vector, and every experimental
condition is a deterministic transformation of that vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    ORACLE_TOLERANCE,
    PolicyLCBLaunchSpec,
    gaussian_lcb_quantile,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
)
from experiments.seeds import derive_seed, rng_from_seed


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
    "evaluate_variable_finite_grid_lcb_draw",
    "evaluate_variable_finite_grid_lcb_seed",
    "load_variable_finite_grid_lcb_manifest",
    "parse_variable_finite_grid_lcb_manifest",
    "true_objective",
    "variable_grid_noise_seed",
]
