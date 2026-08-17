"""Manifest-driven continuous-GP regret decomposition experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    PolicyLCBLaunchSpec,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
    write_json_atomic,
)
from experiments.policy_lcb.continuous_gp_core import (
    DecomposedGPLandscape,
    FiniteFourierGPSpec,
    FourierGPDraw,
    GlobalMaximumResult,
    GlobalReferenceSpec,
    SmoothClippedUncertaintySpec,
    UniformConfidenceSpec,
    analytic_uniform_certificate,
    certified_coverage_probability,
    certified_global_maximum,
    certified_shape_ratio,
    certified_weighted_gp_supremum,
    smooth_clipped_uncertainty,
    true_regret,
    true_value,
)
from experiments.seeds import derive_seed, rng_from_seed
from experiments.sweep_reporting import write_rows_csv


EstimatorName = Literal["finite_difference", "stein_difference"]
CoverageStatus = Literal["covered", "violated", "undecided"]


@dataclass(frozen=True)
class DecompositionOptimizerSpec:
    """Projected zeroth-order optimizer and retained checkpoint settings."""

    enabled_estimators: tuple[EstimatorName, ...]
    starts: tuple[float, ...]
    checkpoint_steps: tuple[int, ...]
    step_size: float
    perturbation_radius: float
    n_stein_perturbations: int

    def __post_init__(self) -> None:
        estimators = tuple(str(value) for value in self.enabled_estimators)
        if not estimators or len(set(estimators)) != len(estimators):
            raise ValueError("optimizer.enabled_estimators must be non-empty and unique.")
        if set(estimators) - {"finite_difference", "stein_difference"}:
            raise ValueError("optimizer.enabled_estimators contains an unknown method.")
        starts = tuple(float(value) for value in self.starts)
        if not starts or any(not 0.0 <= value <= 1.0 for value in starts):
            raise ValueError("optimizer.starts must lie in [0, 1].")
        checkpoints = tuple(int(value) for value in self.checkpoint_steps)
        if not checkpoints or checkpoints[0] != 0 or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("optimizer.checkpoint_steps must be unique, increasing, and start at zero.")
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("optimizer.step_size must be positive.")
        if not np.isfinite(self.perturbation_radius) or self.perturbation_radius <= 0.0:
            raise ValueError("optimizer.perturbation_radius must be positive.")
        if int(self.n_stein_perturbations) <= 0:
            raise ValueError("optimizer.n_stein_perturbations must be positive.")
        object.__setattr__(self, "enabled_estimators", estimators)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "checkpoint_steps", checkpoints)

    @property
    def max_steps(self) -> int:
        return self.checkpoint_steps[-1]


@dataclass(frozen=True)
class DecompositionDesignSpec:
    """Explicit scale, center, factorial, and robustness grids."""

    surrogate_centers: tuple[float, ...]
    one_axis_scales: tuple[float, ...]
    envelope_centers: tuple[float, ...]
    factorial_scales: tuple[float, ...]
    robustness_scale_pairs: tuple[tuple[float, float], ...]
    robustness_envelope_centers: tuple[tuple[float, tuple[float, ...]], ...]
    include_perfect_control: bool

    def __post_init__(self) -> None:
        for label, values in (
            ("surrogate_centers", self.surrogate_centers),
            ("one_axis_scales", self.one_axis_scales),
            ("envelope_centers", self.envelope_centers),
            ("factorial_scales", self.factorial_scales),
        ):
            values = tuple(float(value) for value in values)
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"design.{label} must be unique and increasing.")
            if label.endswith("centers") and any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"design.{label} must lie in [0, 1].")
            if label.endswith("scales") and any(value < 0.0 or not np.isfinite(value) for value in values):
                raise ValueError(f"design.{label} must be finite and non-negative.")
            object.__setattr__(self, label, values)
        if 1.0 not in self.one_axis_scales:
            raise ValueError("design.one_axis_scales must include the matched value 1.")
        if any(scale <= 0.0 for scale in self.factorial_scales):
            raise ValueError("design.factorial_scales must be strictly positive.")
        center_map = dict(self.robustness_envelope_centers)
        if set(center_map) != set(self.surrogate_centers):
            raise ValueError("shape robustness must specify every surrogate center.")
        if any(cf <= 0.0 or ce <= 0.0 for cf, ce in self.robustness_scale_pairs):
            raise ValueError("shape robustness scale pairs must be positive.")


@dataclass(frozen=True)
class DecompositionCondition:
    """One de-duplicated analytic landscape and its analysis memberships."""

    condition_id: str
    memberships: tuple[str, ...]
    surrogate_center: float
    surrogate_scale: float
    envelope_center: float
    envelope_scale: float
    run_optimizer: bool


@dataclass(frozen=True)
class ContinuousGPDecompositionSpec:
    """Resolved inputs for the paired regret-decomposition experiment."""

    domain: tuple[float, float]
    gp: FiniteFourierGPSpec
    uncertainty: SmoothClippedUncertaintySpec
    confidence: UniformConfidenceSpec
    global_reference: GlobalReferenceSpec
    optimizer: DecompositionOptimizerSpec
    design: DecompositionDesignSpec
    master_gp_seed: int
    master_optimizer_seed: int
    reporting_seed: int
    run_seeds: tuple[int, ...]
    diagnostic_run_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if tuple(float(value) for value in self.domain) != (0.0, 1.0):
            raise ValueError("domain must be exactly [0, 1].")
        seeds = tuple(int(value) for value in self.run_seeds)
        diagnostics = tuple(int(value) for value in self.diagnostic_run_seeds)
        if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise ValueError("seeds.run_seeds must be unique and non-negative.")
        if not set(diagnostics).issubset(seeds):
            raise ValueError("seeds.diagnostic_run_seeds must be a subset of run_seeds.")
        if any(value < 0 for value in (self.master_gp_seed, self.master_optimizer_seed, self.reporting_seed)):
            raise ValueError("master and reporting seeds must be non-negative.")
        object.__setattr__(self, "run_seeds", seeds)
        object.__setattr__(self, "diagnostic_run_seeds", diagnostics)

    def conditions(self) -> tuple[DecompositionCondition, ...]:
        return build_decomposition_conditions(self.design)


@dataclass(frozen=True)
class ContinuousGPDecompositionLaunchSpec(PolicyLCBLaunchSpec):
    """Launch settings for the decomposition manifest adapter."""


@dataclass(frozen=True)
class ContinuousGPDecompositionManifest:
    """Resolved ``continuous_gp_regret_decomposition`` manifest."""

    name: str
    spec: ContinuousGPDecompositionSpec
    launch: ContinuousGPDecompositionLaunchSpec
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / path_part(self.name)

    def seed_dir(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / "seeds" / f"seed-{run_seed}"

    def seed_result_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "result.json"

    def seed_checkpoint_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "optimizer_checkpoints.npz"


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_decomposition_conditions(
    design: DecompositionDesignSpec,
) -> tuple[DecompositionCondition, ...]:
    """Build and de-duplicate every predeclared one-axis and combined condition."""
    memberships: dict[tuple[float, float, float, float], set[str]] = {}
    optimizer_keys: set[tuple[float, float, float, float]] = set()

    def add(mf: float, cf: float, me: float, ce: float, member: str, optimize: bool) -> None:
        key = (float(mf), float(cf), float(me), float(ce))
        memberships.setdefault(key, set()).add(member)
        if optimize:
            optimizer_keys.add(key)

    for mf in design.surrogate_centers:
        for cf in design.one_axis_scales:
            add(mf, cf, mf, 1.0, "axis_1_surrogate_scale", False)
        for ce in design.one_axis_scales:
            add(mf, 1.0, mf, ce, "axis_2_envelope_scale", False)
        for me in design.envelope_centers:
            add(mf, 1.0, me, 1.0, "axis_2_envelope_shape", False)
        add(mf, 1.0, mf, 1.0, "axis_3_optimizer", True)
        for cf in design.factorial_scales:
            for ce in design.factorial_scales:
                add(mf, cf, mf, ce, "combined_factorial", True)
        robust_centers = dict(design.robustness_envelope_centers)[mf]
        for cf, ce in design.robustness_scale_pairs:
            for me in robust_centers:
                add(mf, cf, me, ce, "shape_robustness", True)
    if design.include_perfect_control:
        add(0.5, 0.0, 0.5, 0.0, "perfect_surrogate_control", True)

    output = []
    for mf, cf, me, ce in sorted(memberships):
        condition_id = f"mf-{_slug(mf)}__cf-{_slug(cf)}__me-{_slug(me)}__ce-{_slug(ce)}"
        key = (mf, cf, me, ce)
        output.append(
            DecompositionCondition(
                condition_id=condition_id,
                memberships=tuple(sorted(memberships[key])),
                surrogate_center=mf,
                surrogate_scale=cf,
                envelope_center=me,
                envelope_scale=ce,
                run_optimizer=key in optimizer_keys,
            )
        )
    return tuple(output)


def continuous_gp_decomposition_seed(spec: ContinuousGPDecompositionSpec, run_seed: int) -> int:
    """Derive the exact §3.6.6 Fourier-path seed for a run identifier."""
    if int(run_seed) not in spec.run_seeds:
        raise ValueError(f"Unknown run seed {run_seed}.")
    return derive_seed(spec.master_gp_seed, f"continuous-gp-variable-lcb:run:{int(run_seed)}")


def continuous_gp_decomposition_optimizer_seed(spec: ContinuousGPDecompositionSpec) -> int:
    """Return the fixed §3.6.6 Stein perturbation stream."""
    return derive_seed(spec.master_optimizer_seed, "continuous-gp-variable-lcb:stein")


def draw_decomposition_gp(spec: ContinuousGPDecompositionSpec, run_seed: int) -> FourierGPDraw:
    """Draw the replayable Fourier path for one run seed."""
    rng = rng_from_seed(continuous_gp_decomposition_seed(spec, run_seed))
    return FourierGPDraw(
        spec.gp,
        tuple(float(value) for value in rng.normal(size=spec.gp.rank)),
        tuple(float(value) for value in rng.normal(size=spec.gp.rank)),
    )


@dataclass(frozen=True)
class DecompositionConditionMetric:
    """Certified geometry and exact-LCB selection for one condition."""

    run_seed: int
    gp_seed: int
    condition_id: str
    memberships: str
    surrogate_center: float
    surrogate_scale: float
    envelope_center: float
    envelope_scale: float
    shape_mismatch: float
    envelope_distance_to_optimum: float
    shape_ratio_value: float
    shape_ratio_lower: float
    shape_ratio_upper: float
    effective_gp_threshold: float
    certified_coverage_probability: float
    surrogate_error_lower: float
    surrogate_error_upper: float
    violation_lower: float
    violation_upper: float
    coverage_status: CoverageStatus
    global_x: float
    global_lower_value: float
    global_upper_value: float
    global_bound_gap: float
    selected_true_value: float
    selected_surrogate_value: float
    true_regret: float
    envelope_term: float
    optimization_error_lower: float
    optimization_error_upper: float
    decomposition_rhs_lower: float
    decomposition_rhs_upper: float
    certificate_slack_lower: float
    certificate_slack_upper: float
    certificate_eligible: bool


@dataclass(frozen=True)
class DecompositionCheckpointMetric:
    """One raw or best-of-start optimizer checkpoint."""

    run_seed: int
    gp_seed: int
    optimizer_seed: int
    condition_id: str
    memberships: str
    surrogate_center: float
    surrogate_scale: float
    envelope_center: float
    envelope_scale: float
    shape_mismatch: float
    envelope_distance_to_optimum: float
    estimator: EstimatorName
    start_x: float
    is_best_start: bool
    step: int
    oracle_queries: int
    x: float
    selected_true_value: float
    selected_surrogate_value: float
    selected_lower_value: float
    selected_point_valid: bool
    true_regret: float
    envelope_term: float
    optimization_error_lower: float
    optimization_error_upper: float
    decomposition_rhs_lower: float
    decomposition_rhs_upper: float
    certificate_slack_lower: float
    certificate_slack_upper: float
    certificate_eligible: bool


@dataclass(frozen=True)
class ContinuousGPDecompositionSeedResult:
    """All replayable transformations of one frozen Fourier path."""

    run_seed: int
    gp_seed: int
    optimizer_seed: int
    a_coefficients: tuple[float, ...]
    b_coefficients: tuple[float, ...]
    conditions: tuple[DecompositionConditionMetric, ...]
    checkpoints: tuple[DecompositionCheckpointMetric, ...]


def _condition_metric(
    *,
    spec: ContinuousGPDecompositionSpec,
    draw: FourierGPDraw,
    run_seed: int,
    gp_seed: int,
    condition: DecompositionCondition,
    quantile: float,
    weighted_suprema: Mapping[float, Any],
    shape_ratios: Mapping[tuple[float, float], Any],
) -> tuple[DecompositionConditionMetric, DecomposedGPLandscape, GlobalMaximumResult]:
    mf = condition.surrogate_center
    cf = condition.surrogate_scale
    me = condition.envelope_center
    ce = condition.envelope_scale
    landscape = DecomposedGPLandscape(draw, spec.uncertainty, mf, cf, me, ce, quantile)
    global_result = certified_global_maximum(
        landscape.evaluate,
        second_derivative_bound=landscape.second_derivative_bound(),
        reference=spec.global_reference,
        breakpoints=landscape.breakpoints(),
    )
    violation = replace(landscape, target="violation")
    violation_global = certified_global_maximum(
        violation.evaluate,
        second_derivative_bound=violation.second_derivative_bound(),
        reference=spec.global_reference,
        breakpoints=violation.breakpoints(),
    )
    if violation_global.upper_bound <= 0.0:
        coverage_status: CoverageStatus = "covered"
    elif violation_global.value > 0.0:
        coverage_status = "violated"
    else:
        coverage_status = "undecided"
    ratio = shape_ratios[(mf, me)]
    if cf == 0.0:
        threshold = float("inf")
        coverage_probability = 1.0
    elif ce == 0.0:
        threshold = 0.0
        coverage_probability = 0.0
    else:
        threshold = quantile * ce * ratio.lower_bound / cf
        coverage_probability = certified_coverage_probability(
            spec.gp, spec.confidence, threshold
        )
    weighted = weighted_suprema[mf]
    surrogate_error_lower = cf * weighted.lower_bound
    surrogate_error_upper = cf * weighted.upper_bound
    x = global_result.x
    lower_value = float(landscape.evaluate(x))
    surrogate_value = float(replace(landscape, target="surrogate").evaluate(x))
    regret = float(true_regret(x))
    envelope_term = float(true_value(0.5) - landscape.evaluate(0.5))
    epsilon_lower = max(0.0, global_result.value - lower_value)
    epsilon_upper = max(0.0, global_result.upper_bound - lower_value)
    eligible = coverage_status == "covered"
    return (
        DecompositionConditionMetric(
            run_seed=run_seed,
            gp_seed=gp_seed,
            condition_id=condition.condition_id,
            memberships="|".join(condition.memberships),
            surrogate_center=mf,
            surrogate_scale=cf,
            envelope_center=me,
            envelope_scale=ce,
            shape_mismatch=abs(me - mf),
            envelope_distance_to_optimum=abs(me - 0.5),
            shape_ratio_value=ratio.value,
            shape_ratio_lower=ratio.lower_bound,
            shape_ratio_upper=ratio.upper_bound,
            effective_gp_threshold=threshold,
            certified_coverage_probability=coverage_probability,
            surrogate_error_lower=surrogate_error_lower,
            surrogate_error_upper=surrogate_error_upper,
            violation_lower=violation_global.value,
            violation_upper=violation_global.upper_bound,
            coverage_status=coverage_status,
            global_x=x,
            global_lower_value=global_result.value,
            global_upper_value=global_result.upper_bound,
            global_bound_gap=global_result.bound_gap,
            selected_true_value=float(true_value(x)),
            selected_surrogate_value=surrogate_value,
            true_regret=regret,
            envelope_term=envelope_term,
            optimization_error_lower=epsilon_lower,
            optimization_error_upper=epsilon_upper,
            decomposition_rhs_lower=envelope_term + epsilon_lower,
            decomposition_rhs_upper=envelope_term + epsilon_upper,
            certificate_slack_lower=envelope_term + epsilon_lower - regret,
            certificate_slack_upper=envelope_term + epsilon_upper - regret,
            certificate_eligible=eligible,
        ),
        landscape,
        global_result,
    )


def _checkpoint_row(
    *,
    spec: ContinuousGPDecompositionSpec,
    landscape: DecomposedGPLandscape,
    global_result: GlobalMaximumResult,
    condition: DecompositionCondition,
    condition_metric: DecompositionConditionMetric,
    run_seed: int,
    gp_seed: int,
    optimizer_seed: int,
    estimator: EstimatorName,
    start: float,
    step: int,
    x: float,
    is_best: bool,
) -> DecompositionCheckpointMetric:
    lower_value = float(landscape.evaluate(x))
    surrogate_value = float(replace(landscape, target="surrogate").evaluate(x))
    true = float(true_value(x))
    regret = float(true_regret(x))
    epsilon_lower = max(0.0, global_result.value - lower_value)
    epsilon_upper = max(0.0, global_result.upper_bound - lower_value)
    query_multiplier = 2 if estimator == "finite_difference" else 2 * spec.optimizer.n_stein_perturbations
    return DecompositionCheckpointMetric(
        run_seed=run_seed,
        gp_seed=gp_seed,
        optimizer_seed=optimizer_seed,
        condition_id=condition.condition_id,
        memberships="|".join(condition.memberships),
        surrogate_center=condition.surrogate_center,
        surrogate_scale=condition.surrogate_scale,
        envelope_center=condition.envelope_center,
        envelope_scale=condition.envelope_scale,
        shape_mismatch=abs(condition.envelope_center - condition.surrogate_center),
        envelope_distance_to_optimum=abs(condition.envelope_center - 0.5),
        estimator=estimator,
        start_x=start,
        is_best_start=is_best,
        step=step,
        oracle_queries=query_multiplier * step,
        x=x,
        selected_true_value=true,
        selected_surrogate_value=surrogate_value,
        selected_lower_value=lower_value,
        selected_point_valid=lower_value <= true,
        true_regret=regret,
        envelope_term=condition_metric.envelope_term,
        optimization_error_lower=epsilon_lower,
        optimization_error_upper=epsilon_upper,
        decomposition_rhs_lower=condition_metric.envelope_term + epsilon_lower,
        decomposition_rhs_upper=condition_metric.envelope_term + epsilon_upper,
        certificate_slack_lower=condition_metric.envelope_term + epsilon_lower - regret,
        certificate_slack_upper=condition_metric.envelope_term + epsilon_upper - regret,
        certificate_eligible=condition_metric.coverage_status == "covered",
    )


def _run_condition_optimizer(
    *,
    spec: ContinuousGPDecompositionSpec,
    landscape: DecomposedGPLandscape,
    global_result: GlobalMaximumResult,
    condition: DecompositionCondition,
    condition_metric: DecompositionConditionMetric,
    run_seed: int,
    gp_seed: int,
    optimizer_seed: int,
    epsilon_samples: np.ndarray,
) -> tuple[DecompositionCheckpointMetric, ...]:
    optimizer = spec.optimizer
    retained: list[DecompositionCheckpointMetric] = []
    by_estimator_step: dict[tuple[str, int], list[DecompositionCheckpointMetric]] = {}
    for estimator in optimizer.enabled_estimators:
        for start in optimizer.starts:
            x = float(start)
            checkpoint_set = set(optimizer.checkpoint_steps)
            for step in range(optimizer.max_steps + 1):
                if step in checkpoint_set:
                    row = _checkpoint_row(
                        spec=spec,
                        landscape=landscape,
                        global_result=global_result,
                        condition=condition,
                        condition_metric=condition_metric,
                        run_seed=run_seed,
                        gp_seed=gp_seed,
                        optimizer_seed=optimizer_seed,
                        estimator=estimator,
                        start=start,
                        step=step,
                        x=x,
                        is_best=False,
                    )
                    retained.append(row)
                    by_estimator_step.setdefault((estimator, step), []).append(row)
                if step == optimizer.max_steps:
                    break
                radius = optimizer.perturbation_radius
                if estimator == "finite_difference":
                    gradient = float(
                        (landscape.evaluate(x + radius) - landscape.evaluate(x - radius))
                        / (2.0 * radius)
                    )
                else:
                    epsilon = epsilon_samples[step]
                    plus = np.asarray(landscape.evaluate(x + radius * epsilon), dtype=float)
                    minus = np.asarray(landscape.evaluate(x - radius * epsilon), dtype=float)
                    gradient = float(np.mean((plus - minus) * epsilon) / (2.0 * radius))
                x = float(np.clip(x + optimizer.step_size * gradient, 0.0, 1.0))
    best_rows = []
    for key in sorted(by_estimator_step):
        candidates = by_estimator_step[key]
        best = min(candidates, key=lambda row: (-row.selected_lower_value, row.start_x))
        best_rows.append(replace(best, is_best_start=True))
    return tuple(retained + best_rows)


def evaluate_continuous_gp_decomposition_seed(
    spec: ContinuousGPDecompositionSpec, run_seed: int
) -> ContinuousGPDecompositionSeedResult:
    """Evaluate every de-duplicated condition on one frozen Fourier path."""
    gp_seed = continuous_gp_decomposition_seed(spec, run_seed)
    draw = draw_decomposition_gp(spec, run_seed)
    optimizer_seed = continuous_gp_decomposition_optimizer_seed(spec)
    quantile = analytic_uniform_certificate(spec.gp, spec.confidence).quantile
    weighted_suprema = {
        center: certified_weighted_gp_supremum(draw, spec.uncertainty, center, spec.global_reference)
        for center in spec.design.surrogate_centers
    }
    shape_ratios = {
        (mf, me): certified_shape_ratio(spec.uncertainty, mf, me, spec.global_reference)
        for mf in spec.design.surrogate_centers
        for me in spec.design.envelope_centers
    }
    epsilon_samples = rng_from_seed(optimizer_seed).normal(
        size=(spec.optimizer.max_steps, spec.optimizer.n_stein_perturbations)
    )
    metrics: list[DecompositionConditionMetric] = []
    checkpoints: list[DecompositionCheckpointMetric] = []
    for condition in spec.conditions():
        metric, landscape, global_result = _condition_metric(
            spec=spec,
            draw=draw,
            run_seed=int(run_seed),
            gp_seed=gp_seed,
            condition=condition,
            quantile=quantile,
            weighted_suprema=weighted_suprema,
            shape_ratios=shape_ratios,
        )
        metrics.append(metric)
        if condition.run_optimizer:
            checkpoints.extend(
                _run_condition_optimizer(
                    spec=spec,
                    landscape=landscape,
                    global_result=global_result,
                    condition=condition,
                    condition_metric=metric,
                    run_seed=int(run_seed),
                    gp_seed=gp_seed,
                    optimizer_seed=optimizer_seed,
                    epsilon_samples=epsilon_samples,
                )
            )
    return ContinuousGPDecompositionSeedResult(
        run_seed=int(run_seed),
        gp_seed=gp_seed,
        optimizer_seed=optimizer_seed,
        a_coefficients=draw.a,
        b_coefficients=draw.b,
        conditions=tuple(metrics),
        checkpoints=tuple(checkpoints),
    )


def _range_from_payload(payload: Mapping[str, Any], label: str) -> tuple[int, ...]:
    if payload.get("type") != "range":
        raise ValueError(f"{label}.type must be 'range'.")
    start = int(payload.get("start", -1))
    count = int(payload.get("count", 0))
    if start < 0 or count <= 0:
        raise ValueError(f"{label} start must be non-negative and count positive.")
    return tuple(range(start, start + count))


def load_continuous_gp_decomposition_manifest(
    path: str | Path,
) -> ContinuousGPDecompositionManifest:
    """Load and validate a regret-decomposition manifest."""
    manifest_path = Path(path)
    return parse_continuous_gp_decomposition_manifest(
        read_json(manifest_path), source_path=manifest_path
    )


def parse_continuous_gp_decomposition_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ContinuousGPDecompositionManifest:
    """Resolve the explicit decomposition grids and seed contract."""
    if not isinstance(payload, Mapping):
        raise ValueError("Continuous-GP decomposition manifest must be a JSON object.")
    if payload.get("kind") != "continuous_gp_regret_decomposition":
        raise ValueError("Manifest kind must be 'continuous_gp_regret_decomposition'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Manifest name must be non-empty.")
    domain = number_sequence(payload.get("domain"), "domain")
    objective = required_mapping(payload, "true_value")
    if (
        objective.get("type") != "concave_quadratic"
        or float(objective.get("linear", np.nan)) != 5.0
        or float(objective.get("quadratic", np.nan)) != 5.0
    ):
        raise ValueError("true_value must be concave_quadratic with linear=quadratic=5.")
    gp_payload = required_mapping(payload, "gp")
    if gp_payload.get("type") != "deterministic_spectral_finite_rank":
        raise ValueError("gp.type must be 'deterministic_spectral_finite_rank'.")
    uncertainty_payload = required_mapping(payload, "uncertainty")
    if uncertainty_payload.get("type") != "smooth_clipped_distance_ramp":
        raise ValueError("uncertainty.type must be 'smooth_clipped_distance_ramp'.")
    confidence_payload = required_mapping(payload, "confidence")
    if confidence_payload.get("type") != "bonferroni_net_smoothness":
        raise ValueError("confidence.type must be 'bonferroni_net_smoothness'.")
    reference_payload = required_mapping(payload, "global_reference")
    if reference_payload.get("type") != "certified_branch_and_bound":
        raise ValueError("global_reference.type must be 'certified_branch_and_bound'.")
    design_payload = required_mapping(payload, "design")
    robustness = required_mapping(design_payload, "shape_robustness")
    raw_pairs = robustness.get("scale_pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, (str, bytes)):
        raise ValueError("design.shape_robustness.scale_pairs must be a sequence.")
    scale_pairs: list[tuple[float, float]] = []
    for pair in raw_pairs:
        values = number_sequence(pair, "design.shape_robustness.scale_pairs item")
        if len(values) != 2:
            raise ValueError("Each robustness scale pair must contain c_f and c_E.")
        scale_pairs.append((values[0], values[1]))
    raw_center_map = required_mapping(robustness, "envelope_centers_by_surrogate")
    center_map = tuple(
        sorted(
            (
                float(key),
                number_sequence(value, f"shape robustness centers for {key}"),
            )
            for key, value in raw_center_map.items()
        )
    )
    optimizer_payload = required_mapping(payload, "optimizer")
    if optimizer_payload.get("step_rule") != "projected_constant":
        raise ValueError("optimizer.step_rule must be 'projected_constant'.")
    if optimizer_payload.get("probe_domain") != "analytic_real_line_extension":
        raise ValueError("optimizer.probe_domain must document the analytic real-line extension.")
    estimators = optimizer_payload.get("enabled_estimators")
    if not isinstance(estimators, Sequence) or isinstance(estimators, (str, bytes)):
        raise ValueError("optimizer.enabled_estimators must be a sequence.")
    seeds = required_mapping(payload, "seeds")
    run_seeds = _range_from_payload(required_mapping(seeds, "run_seeds"), "seeds.run_seeds")
    diagnostics = tuple(int(value) for value in seeds.get("diagnostic_run_seeds", ()))
    launch_payload = required_mapping(payload, "launch")
    mode = str(launch_payload.get("mode") or "")
    array = str(launch_payload.get("array") or "")
    if mode not in {"auto", "local", "slurm"} or array not in {"none", "seed"}:
        raise ValueError("launch mode/array must use the registered local, slurm, auto and none, seed values.")
    parallel_raw = launch_payload.get("array_max_parallel")
    parallel = None if parallel_raw is None else int(parallel_raw)
    if parallel is not None and parallel <= 0:
        raise ValueError("launch.array_max_parallel must be positive.")

    design = DecompositionDesignSpec(
        surrogate_centers=number_sequence(design_payload.get("surrogate_centers"), "design.surrogate_centers"),
        one_axis_scales=number_sequence(design_payload.get("one_axis_scales"), "design.one_axis_scales"),
        envelope_centers=number_sequence(design_payload.get("envelope_centers"), "design.envelope_centers"),
        factorial_scales=number_sequence(design_payload.get("factorial_scales"), "design.factorial_scales"),
        robustness_scale_pairs=tuple(scale_pairs),
        robustness_envelope_centers=center_map,
        include_perfect_control=bool(design_payload.get("include_perfect_control", False)),
    )
    all_centers = tuple(sorted(set(design.surrogate_centers) | set(design.envelope_centers)))
    spec = ContinuousGPDecompositionSpec(
        domain=(domain[0], domain[1]),
        gp=FiniteFourierGPSpec(
            rank=int(gp_payload.get("rank", 0)),
            lengthscale=float(gp_payload.get("lengthscale", np.nan)),
        ),
        uncertainty=SmoothClippedUncertaintySpec(
            centers=all_centers,
            minimum=float(uncertainty_payload.get("minimum", np.nan)),
            maximum=float(uncertainty_payload.get("maximum", np.nan)),
            ramp_radius=float(uncertainty_payload.get("ramp_radius", np.nan)),
        ),
        confidence=UniformConfidenceSpec(
            delta=float(confidence_payload.get("delta", np.nan)),
            net_count=int(confidence_payload.get("net_count", 0)),
        ),
        global_reference=GlobalReferenceSpec(
            value_tolerance=float(reference_payload.get("value_tolerance", np.nan)),
            max_intervals=int(reference_payload.get("max_intervals", 0)),
            initial_grid_count=int(reference_payload.get("initial_grid_count", 65)),
        ),
        optimizer=DecompositionOptimizerSpec(
            enabled_estimators=tuple(str(value) for value in estimators),  # type: ignore[arg-type]
            starts=number_sequence(optimizer_payload.get("starts"), "optimizer.starts"),
            checkpoint_steps=tuple(int(value) for value in optimizer_payload.get("checkpoint_steps", ())),
            step_size=float(optimizer_payload.get("step_size", np.nan)),
            perturbation_radius=float(optimizer_payload.get("perturbation_radius", np.nan)),
            n_stein_perturbations=int(optimizer_payload.get("n_stein_perturbations", 0)),
        ),
        design=design,
        master_gp_seed=int(seeds.get("master_gp_seed", -1)),
        master_optimizer_seed=int(seeds.get("master_optimizer_seed", -1)),
        reporting_seed=int(seeds.get("reporting_seed", -1)),
        run_seeds=run_seeds,
        diagnostic_run_seeds=diagnostics,
    )
    return ContinuousGPDecompositionManifest(
        name=name,
        spec=spec,
        launch=ContinuousGPDecompositionLaunchSpec(
            mode=mode,  # type: ignore[arg-type]
            array=array,  # type: ignore[arg-type]
            array_max_parallel=parallel,
        ),
        source_path=None if source_path is None else Path(source_path),
    )


def _write_checkpoint_rows(path: Path, rows: Sequence[DecompositionCheckpointMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = {
        field.name: np.asarray([getattr(row, field.name) for row in rows])
        for field in fields(DecompositionCheckpointMetric)
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **columns)
    temporary.replace(path)


def _read_checkpoint_rows(path: Path) -> tuple[DecompositionCheckpointMetric, ...]:
    with np.load(path, allow_pickle=False) as data:
        count = len(data["run_seed"])
        return tuple(
            DecompositionCheckpointMetric(
                **{
                    field.name: data[field.name][index].item()
                    for field in fields(DecompositionCheckpointMetric)
                }
            )
            for index in range(count)
        )


def _read_seed_result(path: Path) -> ContinuousGPDecompositionSeedResult:
    payload = read_json(path)
    return ContinuousGPDecompositionSeedResult(
        run_seed=int(payload["run_seed"]),
        gp_seed=int(payload["gp_seed"]),
        optimizer_seed=int(payload["optimizer_seed"]),
        a_coefficients=tuple(float(value) for value in payload["a_coefficients"]),
        b_coefficients=tuple(float(value) for value in payload["b_coefficients"]),
        conditions=tuple(DecompositionConditionMetric(**row) for row in payload["conditions"]),
        checkpoints=(),
    )


def continuous_gp_decomposition_seed_complete(
    manifest: ContinuousGPDecompositionManifest,
    run_seed: int,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return whether both durable files for one seed exist."""
    return manifest.seed_result_path(run_seed, runs_root).exists() and manifest.seed_checkpoint_path(
        run_seed, runs_root
    ).exists()


def run_continuous_gp_decomposition_manifest_seed(
    manifest: ContinuousGPDecompositionManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run and atomically persist one decomposition seed task."""
    if index < 0 or index >= len(manifest.spec.run_seeds):
        raise IndexError(f"Seed task index {index} is out of range.")
    run_seed = manifest.spec.run_seeds[index]
    write_continuous_gp_decomposition_experiment_readme(manifest, runs_root=runs_root)
    if continuous_gp_decomposition_seed_complete(manifest, run_seed, runs_root=runs_root) and not force:
        return {"project_dir": str(manifest.project_dir(runs_root)), "run_seed": run_seed, "skipped": True}
    result = evaluate_continuous_gp_decomposition_seed(manifest.spec, run_seed)
    payload = asdict(result)
    payload.pop("checkpoints")
    write_json_atomic(manifest.seed_result_path(run_seed, runs_root), payload)
    _write_checkpoint_rows(manifest.seed_checkpoint_path(run_seed, runs_root), result.checkpoints)
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "run_seed": run_seed,
        "skipped": False,
        "n_condition_rows": len(result.conditions),
        "n_checkpoint_rows": len(result.checkpoints),
    }


def run_continuous_gp_decomposition_manifest_serial(
    manifest: ContinuousGPDecompositionManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run all decomposition seeds serially and collect outputs."""
    payloads = [
        run_continuous_gp_decomposition_manifest_seed(
            manifest, index, runs_root=runs_root, force=force
        )
        for index in range(len(manifest.spec.run_seeds))
    ]
    collected = collect_continuous_gp_decomposition_outputs(manifest, runs_root=runs_root)
    return {**collected, "n_skipped_seeds": sum(bool(row["skipped"]) for row in payloads)}


def collect_continuous_gp_decomposition_outputs(
    manifest: ContinuousGPDecompositionManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, object]:
    """Collect raw seed tables and delegate deterministic aggregate reporting."""
    project_dir = manifest.project_dir(runs_root)
    write_continuous_gp_decomposition_experiment_readme(manifest, runs_root=runs_root)
    results = [
        _read_seed_result(manifest.seed_result_path(seed, runs_root))
        for seed in manifest.spec.run_seeds
    ]
    condition_rows = [asdict(row) for result in results for row in result.conditions]
    checkpoint_objects = [
        row
        for seed in manifest.spec.run_seeds
        for row in _read_checkpoint_rows(manifest.seed_checkpoint_path(seed, runs_root))
    ]
    checkpoint_rows = [asdict(row) for row in checkpoint_objects]
    best_rows = [row for row in checkpoint_rows if bool(row["is_best_start"])]
    write_rows_csv(project_dir / "seed_condition_metrics.csv", condition_rows, tuple(condition_rows[0]))
    write_rows_csv(project_dir / "seed_optimizer_checkpoints.csv", checkpoint_rows, tuple(checkpoint_rows[0]))
    write_rows_csv(project_dir / "seed_optimizer_best.csv", best_rows, tuple(best_rows[0]))
    from experiments.policy_lcb.continuous_gp_decomposition_reporting import (
        write_decomposition_reports,
    )

    write_decomposition_reports(
        manifest,
        results=results,
        condition_rows=condition_rows,
        checkpoint_rows=checkpoint_rows,
        best_rows=best_rows,
        project_dir=project_dir,
    )
    return {
        "project_dir": str(project_dir),
        "n_seed_results": len(results),
        "n_condition_rows": len(condition_rows),
        "n_checkpoint_rows": len(checkpoint_rows),
        "n_best_rows": len(best_rows),
    }


def write_continuous_gp_decomposition_experiment_readme(
    manifest: ContinuousGPDecompositionManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write representation, grid, certificate, and seed decisions beside results."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    spec = manifest.spec
    certificate = analytic_uniform_certificate(spec.gp, spec.confidence)
    source = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    path.write_text(
        f"""# {manifest.name}

- Manifest: `{source}`
- Domain: continuous `[0, 1]`; every path is evaluated from its analytic Fourier formula
- GP: rank `{spec.gp.rank}`, lengthscale `{spec.gp.lengthscale}`
- Surrogate centers `m_f`: `{list(spec.design.surrogate_centers)}`
- Envelope centers `m_E`: `{list(spec.design.envelope_centers)}`
- One-axis scales: `{list(spec.design.one_axis_scales)}`
- Factorial scales: `{list(spec.design.factorial_scales)}`
- Run seeds: `{spec.run_seeds[0]}..{spec.run_seeds[-1]}` ({len(spec.run_seeds)} paths)
- Diagnostic paths: `{list(spec.diagnostic_run_seeds)}`
- Dedicated GP / optimizer / reporting seeds: `{spec.master_gp_seed}` / `{spec.master_optimizer_seed}` / `{spec.reporting_seed}`

## Representation and optimization

`f_hat(x) = f(x) + c_f*sigma_mf(x)*G_s(x)` and
`lower(x) = f_hat(x) - c_E*q*sigma_mE(x)`, with `q={certificate.quantile:.12g}`.
`sigma_m` is the fixed C2 clipped smoothstep profile with minimum
`{spec.uncertainty.minimum:g}` at `m` and maximum `{spec.uncertainty.maximum:g}`.
Changing `c` changes global magnitude; changing `m` reallocates amplitude over the
domain. Plot grids only render the analytic functions and never define them.

Finite-difference and `{spec.optimizer.n_stein_perturbations}`-perturbation antithetic
Stein probes use the analytic real-line extension. Iterates are projected to `[0,1]`.
All conditions and paths share the dedicated optimizer perturbation stream, while one
run seed owns one Fourier coefficient draw reused by every condition and start.

## Certification

The reported analytic coverage level inverts the original two-sided covering-net plus
coefficient-norm proof at `q_eff = q*(c_E/c_f)*inf_x sigma_mE/sigma_mf`.
Realized one-sided envelope validity, surrogate sup error, global lower-envelope values,
and optimizer errors are separately bracketed by branch-and-bound to value tolerance
`{spec.global_reference.value_tolerance:g}`. The true objective is used only for
post-selection metrics and never clips or modifies the optimized lower envelope.
""",
        encoding="utf-8",
    )
    return path


__all__ = [
    "ContinuousGPDecompositionLaunchSpec",
    "ContinuousGPDecompositionManifest",
    "ContinuousGPDecompositionSeedResult",
    "ContinuousGPDecompositionSpec",
    "DecompositionCheckpointMetric",
    "DecompositionCondition",
    "DecompositionConditionMetric",
    "DecompositionDesignSpec",
    "DecompositionOptimizerSpec",
    "build_decomposition_conditions",
    "collect_continuous_gp_decomposition_outputs",
    "continuous_gp_decomposition_optimizer_seed",
    "continuous_gp_decomposition_seed",
    "continuous_gp_decomposition_seed_complete",
    "draw_decomposition_gp",
    "evaluate_continuous_gp_decomposition_seed",
    "load_continuous_gp_decomposition_manifest",
    "parse_continuous_gp_decomposition_manifest",
    "run_continuous_gp_decomposition_manifest_seed",
    "run_continuous_gp_decomposition_manifest_serial",
    "write_continuous_gp_decomposition_experiment_readme",
]
