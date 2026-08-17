"""Continuous finite-Fourier GP experiments for variable LCB envelopes.

The random function is analytic at every real-valued query.  Evaluation grids
are used only by deterministic certificates and reporting; they never define or
interpolate a path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import heapq
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import chi2, norm

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    PolicyLCBLaunchSpec,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
    write_json_atomic,
)
from experiments.seeds import derive_seed, rng_from_seed
from experiments.sweep_reporting import write_rows_csv


TargetName = Literal["nominal", "variable_lcb"]
EstimatorName = Literal["first_order", "finite_difference", "stein_difference"]


@dataclass(frozen=True)
class FiniteFourierGPSpec:
    """Deterministic spectral basis for one exact finite-rank GP."""

    rank: int
    lengthscale: float

    def __post_init__(self) -> None:
        if int(self.rank) <= 0:
            raise ValueError("gp.rank must be positive.")
        if not np.isfinite(self.lengthscale) or self.lengthscale <= 0.0:
            raise ValueError("gp.lengthscale must be finite and positive.")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "lengthscale", float(self.lengthscale))

    def frequencies(self) -> np.ndarray:
        probabilities = (np.arange(self.rank, dtype=float) + 0.5) / self.rank
        return norm.ppf((1.0 + probabilities) / 2.0) / self.lengthscale


@dataclass(frozen=True)
class SmoothClippedUncertaintySpec:
    """C2 uncertainty bowl with its minimum at each configured center."""

    centers: tuple[float, ...]
    minimum: float
    maximum: float
    ramp_radius: float

    def __post_init__(self) -> None:
        centers = tuple(float(value) for value in self.centers)
        if not centers or any(not np.isfinite(value) for value in centers):
            raise ValueError("uncertainty.centers must be a non-empty finite sequence.")
        if tuple(sorted(set(centers))) != centers:
            raise ValueError("uncertainty.centers must be unique and strictly increasing.")
        if not np.isfinite(self.minimum) or self.minimum <= 0.0:
            raise ValueError("uncertainty.minimum must be finite and positive.")
        if not np.isfinite(self.maximum) or self.maximum < self.minimum:
            raise ValueError("uncertainty.maximum must be at least uncertainty.minimum.")
        if not np.isfinite(self.ramp_radius) or self.ramp_radius <= 0.0:
            raise ValueError("uncertainty.ramp_radius must be finite and positive.")
        object.__setattr__(self, "centers", centers)


@dataclass(frozen=True)
class UniformConfidenceSpec:
    """Analytic net-plus-smoothness simultaneous confidence certificate."""

    delta: float
    net_count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta) or not 0.0 < self.delta < 1.0:
            raise ValueError("confidence.delta must lie in (0, 1).")
        if int(self.net_count) < 2:
            raise ValueError("confidence.net_count must be at least two.")
        object.__setattr__(self, "delta", float(self.delta))
        object.__setattr__(self, "net_count", int(self.net_count))


@dataclass(frozen=True)
class GlobalReferenceSpec:
    """Controls for certified one-dimensional branch-and-bound."""

    value_tolerance: float
    max_intervals: int
    initial_grid_count: int = 65

    def __post_init__(self) -> None:
        if not np.isfinite(self.value_tolerance) or self.value_tolerance <= 0.0:
            raise ValueError("global_reference.value_tolerance must be positive.")
        if int(self.max_intervals) <= 0:
            raise ValueError("global_reference.max_intervals must be positive.")
        if int(self.initial_grid_count) < 3:
            raise ValueError("global_reference.initial_grid_count must be at least three.")


@dataclass(frozen=True)
class ContinuousGPOptimizerSpec:
    """Matched projected optimizer settings for the convergence subset."""

    enabled_estimators: tuple[EstimatorName, ...]
    starts: tuple[float, ...]
    t_steps: int
    step_size: float
    perturbation_radius: float
    n_stein_perturbations: int
    checkpoint_every: int

    def __post_init__(self) -> None:
        allowed = {"first_order", "finite_difference", "stein_difference"}
        estimators = tuple(str(value) for value in self.enabled_estimators)
        if not estimators or len(set(estimators)) != len(estimators):
            raise ValueError("optimizer.enabled_estimators must be non-empty and unique.")
        if set(estimators) - allowed:
            raise ValueError("optimizer.enabled_estimators contains an unknown method.")
        starts = tuple(float(value) for value in self.starts)
        if not starts or any(not 0.0 <= value <= 1.0 for value in starts):
            raise ValueError("optimizer.starts must lie in [0, 1].")
        if int(self.t_steps) <= 0 or int(self.checkpoint_every) <= 0:
            raise ValueError("optimizer steps and checkpoint interval must be positive.")
        if not np.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("optimizer.step_size must be positive.")
        if not np.isfinite(self.perturbation_radius) or self.perturbation_radius <= 0.0:
            raise ValueError("optimizer.perturbation_radius must be positive.")
        if int(self.n_stein_perturbations) <= 0:
            raise ValueError("optimizer.n_stein_perturbations must be positive.")
        object.__setattr__(self, "enabled_estimators", estimators)
        object.__setattr__(self, "starts", starts)


@dataclass(frozen=True)
class AnalyticUniformCertificate:
    """Resolved constants in the continuum-wide coverage proof."""

    delta: float
    net_count: int
    covering_radius: float
    feature_lipschitz: float
    net_quantile: float
    coefficient_radius: float
    remainder: float
    quantile: float


@dataclass(frozen=True)
class GlobalMaximumResult:
    """A value-certified global maximum and its numerical representative."""

    x: float
    value: float
    upper_bound: float
    bound_gap: float
    certified: bool
    intervals_created: int
    evaluations: int


@dataclass(frozen=True)
class GPSupremumResult:
    """Certified lower/upper bracket for one realized absolute GP supremum."""

    lower_bound: float
    upper_bound: float
    bound_gap: float
    certified: bool
    maximizing_x: float
    intervals_created: int
    evaluations: int


@dataclass(frozen=True)
class GPConditionResult:
    """Validity, tightness, and deterministic geometry for one (m, c)."""

    run_seed: int
    gp_seed: int
    uncertainty_center: float
    noise_scale: float
    quantile: float
    simultaneous_coverage: bool
    coverage_certified: bool
    average_half_width: float
    maximum_half_width: float
    optimum_half_width: float
    deterministic_target_x: float
    deterministic_target_certified: bool
    optimum_lower_bound_gap: float
    lcb_regret_certificate: float


@dataclass(frozen=True)
class GPSelectorResult:
    """Globally referenced selection and regret for one target."""

    run_seed: int
    gp_seed: int
    uncertainty_center: float
    noise_scale: float
    target: TargetName
    selected_x: float
    selected_true_value: float
    selected_surrogate_value: float
    selected_target_value: float
    regret: float
    distance_to_optimum: float
    distance_to_uncertainty_center: float
    selected_point_covered: bool
    global_upper_bound: float
    global_bound_gap: float
    global_reference_certified: bool
    certificate_slack: float | None


@dataclass(frozen=True)
class GPOptimizerFinalResult:
    """Final result for one target, method, and start."""

    run_seed: int
    gp_seed: int
    optimizer_seed: int
    uncertainty_center: float
    noise_scale: float
    target: TargetName
    estimator: EstimatorName
    start_x: float
    final_x: float
    final_target_value: float
    final_true_value: float
    true_regret: float
    global_target_value: float
    optimization_gap: float
    distance_to_global_x: float
    selected_point_covered: bool
    certificate_bound: float | None
    certificate_slack: float | None


@dataclass(frozen=True)
class GPOptimizerTrajectoryRow:
    """One retained optimizer checkpoint."""

    run_seed: int
    uncertainty_center: float
    noise_scale: float
    target: TargetName
    estimator: EstimatorName
    start_x: float
    step: int
    x: float
    target_value: float
    true_regret: float
    optimization_gap: float


@dataclass(frozen=True)
class ContinuousGPSeedResult:
    """All deterministic transformations of one replayable GP coefficient draw."""

    run_seed: int
    gp_seed: int
    optimizer_seed: int
    a_coefficients: tuple[float, ...]
    b_coefficients: tuple[float, ...]
    gp_supremum: GPSupremumResult
    conditions: tuple[GPConditionResult, ...]
    selectors: tuple[GPSelectorResult, ...]
    optimizer_finals: tuple[GPOptimizerFinalResult, ...]
    trajectories: tuple[GPOptimizerTrajectoryRow, ...]


def smoothstep(t: float | Sequence[float] | np.ndarray, derivative: int = 0) -> Any:
    """Evaluate the clipped quintic smoothstep or its first two derivatives."""
    values = np.asarray(t, dtype=float)
    inside = (values > 0.0) & (values < 1.0)
    clipped = np.clip(values, 0.0, 1.0)
    if derivative == 0:
        result = 6.0 * clipped**5 - 15.0 * clipped**4 + 10.0 * clipped**3
    elif derivative == 1:
        result = np.where(inside, 30.0 * clipped**2 * (clipped - 1.0) ** 2, 0.0)
    elif derivative == 2:
        result = np.where(
            inside,
            60.0 * clipped * (2.0 * clipped**2 - 3.0 * clipped + 1.0),
            0.0,
        )
    else:
        raise ValueError("smoothstep supports derivative orders 0, 1, and 2.")
    return float(result) if values.ndim == 0 else result


def smooth_clipped_uncertainty(
    x: float | Sequence[float] | np.ndarray,
    center: float,
    spec: SmoothClippedUncertaintySpec,
    derivative: int = 0,
) -> Any:
    """Evaluate sigma_m or its first two real-line derivatives."""
    values = np.asarray(x, dtype=float)
    displacement = values - float(center)
    scaled_distance = np.abs(displacement) / spec.ramp_radius
    amplitude = spec.maximum - spec.minimum
    if derivative == 0:
        result = spec.minimum + amplitude * smoothstep(scaled_distance)
    elif derivative == 1:
        result = (
            amplitude
            * smoothstep(scaled_distance, 1)
            * np.sign(displacement)
            / spec.ramp_radius
        )
    elif derivative == 2:
        result = amplitude * smoothstep(scaled_distance, 2) / spec.ramp_radius**2
    else:
        raise ValueError("uncertainty supports derivative orders 0, 1, and 2.")
    return float(result) if values.ndim == 0 else result


def uncertainty_derivative_bound(spec: SmoothClippedUncertaintySpec, derivative: int) -> float:
    """Return a global absolute derivative bound for the smooth uncertainty."""
    amplitude = spec.maximum - spec.minimum
    if derivative == 0:
        return float(spec.maximum)
    if derivative == 1:
        return float(amplitude * (15.0 / 8.0) / spec.ramp_radius)
    if derivative == 2:
        return float(amplitude * (10.0 * np.sqrt(3.0) / 3.0) / spec.ramp_radius**2)
    raise ValueError("uncertainty bounds support derivative orders 0, 1, and 2.")


def true_value(x: float | Sequence[float] | np.ndarray, derivative: int = 0) -> Any:
    """Evaluate f(x)=5x-5x^2 or either analytic derivative."""
    values = np.asarray(x, dtype=float)
    if derivative == 0:
        result = 5.0 * values - 5.0 * values**2
    elif derivative == 1:
        result = 5.0 - 10.0 * values
    elif derivative == 2:
        result = np.full_like(values, -10.0)
    else:
        raise ValueError("true_value supports derivative orders 0, 1, and 2.")
    return float(result) if values.ndim == 0 else result


def true_regret(x: float | Sequence[float] | np.ndarray) -> Any:
    """Return f(0.5)-f(x)=5(x-0.5)^2."""
    values = np.asarray(x, dtype=float)
    result = 5.0 * (values - 0.5) ** 2
    return float(result) if values.ndim == 0 else result


@dataclass(frozen=True)
class FourierGPDraw:
    """One replayable analytic function behind a small evaluation interface."""

    spec: FiniteFourierGPSpec
    a: tuple[float, ...]
    b: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.a) != self.spec.rank or len(self.b) != self.spec.rank:
            raise ValueError("Fourier coefficients must match gp.rank.")
        if not np.all(np.isfinite(self.a)) or not np.all(np.isfinite(self.b)):
            raise ValueError("Fourier coefficients must be finite.")

    def evaluate(
        self,
        x: float | Sequence[float] | np.ndarray,
        derivative: int = 0,
    ) -> Any:
        """Evaluate the exact Fourier formula, never an interpolant."""
        values = np.asarray(x, dtype=float)
        flat = values.reshape(-1)
        omega = self.spec.frequencies()
        phase = np.outer(flat, omega)
        a = np.asarray(self.a)
        b = np.asarray(self.b)
        scale = np.sqrt(self.spec.rank)
        if derivative == 0:
            result = (np.cos(phase) @ a + np.sin(phase) @ b) / scale
        elif derivative == 1:
            result = (-np.sin(phase) @ (a * omega) + np.cos(phase) @ (b * omega)) / scale
        elif derivative == 2:
            result = (-np.cos(phase) @ (a * omega**2) - np.sin(phase) @ (b * omega**2)) / scale
        else:
            raise ValueError("FourierGPDraw supports derivative orders 0, 1, and 2.")
        reshaped = result.reshape(values.shape)
        return float(reshaped) if values.ndim == 0 else reshaped

    def derivative_bound(self, derivative: int) -> float:
        """Return a coefficient-wise uniform bound on |G^(derivative)|."""
        if derivative not in {0, 1, 2}:
            raise ValueError("GP bounds support derivative orders 0, 1, and 2.")
        amplitudes = np.hypot(np.asarray(self.a), np.asarray(self.b))
        omega = self.spec.frequencies()
        return float(np.sum(amplitudes * omega**derivative) / np.sqrt(self.spec.rank))


def analytic_uniform_certificate(
    gp: FiniteFourierGPSpec,
    confidence: UniformConfidenceSpec,
) -> AnalyticUniformCertificate:
    """Compute the seed-independent 1-delta simultaneous band multiplier."""
    omega = gp.frequencies()
    lipschitz = float(np.sqrt(np.mean(omega**2)))
    radius = 1.0 / (2.0 * (confidence.net_count - 1))
    net_quantile = float(norm.ppf(1.0 - confidence.delta / (4.0 * confidence.net_count)))
    coefficient_radius = float(np.sqrt(chi2.ppf(1.0 - confidence.delta / 2.0, 2 * gp.rank)))
    remainder = radius * lipschitz * coefficient_radius
    return AnalyticUniformCertificate(
        delta=confidence.delta,
        net_count=confidence.net_count,
        covering_radius=radius,
        feature_lipschitz=lipschitz,
        net_quantile=net_quantile,
        coefficient_radius=coefficient_radius,
        remainder=remainder,
        quantile=net_quantile + remainder,
    )


@dataclass(frozen=True)
class ContinuousGPLandscape:
    """Analytic nominal, variable-LCB, or deterministic penalized landscape."""

    draw: FourierGPDraw | None
    uncertainty: SmoothClippedUncertaintySpec
    center: float
    noise_scale: float
    quantile: float
    target: TargetName

    def evaluate(self, x: float | Sequence[float] | np.ndarray, derivative: int = 0) -> Any:
        sigma = smooth_clipped_uncertainty(x, self.center, self.uncertainty, derivative)
        result = np.asarray(true_value(x, derivative), dtype=float)
        if self.draw is not None and self.noise_scale > 0.0:
            if derivative == 0:
                noise = sigma * self.draw.evaluate(x, 0)
            elif derivative == 1:
                noise = sigma * self.draw.evaluate(x, 0) + smooth_clipped_uncertainty(
                    x, self.center, self.uncertainty, 0
                ) * self.draw.evaluate(x, 1)
            elif derivative == 2:
                sigma0 = smooth_clipped_uncertainty(x, self.center, self.uncertainty, 0)
                sigma1 = smooth_clipped_uncertainty(x, self.center, self.uncertainty, 1)
                noise = (
                    sigma * self.draw.evaluate(x, 0)
                    + 2.0 * sigma1 * self.draw.evaluate(x, 1)
                    + sigma0 * self.draw.evaluate(x, 2)
                )
            else:
                raise ValueError("landscapes support derivative orders 0, 1, and 2.")
            result = result + self.noise_scale * np.asarray(noise)
        if self.target == "variable_lcb":
            result = result - self.noise_scale * self.quantile * np.asarray(sigma)
        values = np.asarray(x)
        return float(result) if values.ndim == 0 else result

    def second_derivative_bound(self) -> float:
        sigma0 = uncertainty_derivative_bound(self.uncertainty, 0)
        sigma1 = uncertainty_derivative_bound(self.uncertainty, 1)
        sigma2 = uncertainty_derivative_bound(self.uncertainty, 2)
        bound = 10.0
        if self.draw is not None and self.noise_scale > 0.0:
            bound += self.noise_scale * (
                sigma2 * self.draw.derivative_bound(0)
                + 2.0 * sigma1 * self.draw.derivative_bound(1)
                + sigma0 * self.draw.derivative_bound(2)
            )
        if self.target == "variable_lcb":
            bound += self.noise_scale * self.quantile * sigma2
        return float(bound)

    def breakpoints(self) -> tuple[float, ...]:
        candidates = (0.0, 1.0, self.center, self.center - self.uncertainty.ramp_radius, self.center + self.uncertainty.ramp_radius)
        return tuple(sorted({float(np.clip(value, 0.0, 1.0)) for value in candidates}))


def certified_global_maximum(
    value_fn: Any,
    *,
    second_derivative_bound: float,
    reference: GlobalReferenceSpec,
    breakpoints: Sequence[float] = (0.0, 1.0),
) -> GlobalMaximumResult:
    """Certify a scalar global maximum using interpolation-error upper bounds."""
    if not np.isfinite(second_derivative_bound) or second_derivative_bound < 0.0:
        raise ValueError("second_derivative_bound must be finite and non-negative.")
    base = np.linspace(0.0, 1.0, reference.initial_grid_count)
    points = np.unique(np.concatenate([base, np.asarray(breakpoints, dtype=float)]))
    points = points[(points >= 0.0) & (points <= 1.0)]
    values = np.asarray([float(value_fn(float(x))) for x in points])
    evaluations = len(points)

    candidates: list[tuple[float, float]] = [(float(x), float(y)) for x, y in zip(points, values)]
    for index in range(1, len(points) - 1):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            optimized = minimize_scalar(
                lambda x: -float(value_fn(float(x))),
                bounds=(float(points[index - 1]), float(points[index + 1])),
                method="bounded",
                options={"xatol": min(reference.value_tolerance, 1e-10)},
            )
            evaluations += int(optimized.nfev)
            candidates.append((float(optimized.x), float(-optimized.fun)))
    best_x, best_value = min(candidates, key=lambda item: (-item[1], item[0]))

    def upper(a: float, b: float, fa: float, fb: float) -> float:
        return max(fa, fb) + second_derivative_bound * (b - a) ** 2 / 8.0

    heap: list[tuple[float, float, float, float, float]] = []
    created = 0
    for left, right, f_left, f_right in zip(points[:-1], points[1:], values[:-1], values[1:]):
        bound = upper(float(left), float(right), float(f_left), float(f_right))
        heapq.heappush(heap, (-bound, float(left), float(right), float(f_left), float(f_right)))
        created += 1

    while heap and created < reference.max_intervals:
        maximum_upper = -heap[0][0]
        if maximum_upper - best_value <= reference.value_tolerance:
            break
        _, left, right, f_left, f_right = heapq.heappop(heap)
        midpoint = (left + right) / 2.0
        f_mid = float(value_fn(midpoint))
        evaluations += 1
        if f_mid > best_value or (
            abs(f_mid - best_value) <= reference.value_tolerance and midpoint < best_x
        ):
            best_x, best_value = midpoint, f_mid
        for a, b, fa, fb in (
            (left, midpoint, f_left, f_mid),
            (midpoint, right, f_mid, f_right),
        ):
            bound = upper(a, b, fa, fb)
            if bound - best_value > reference.value_tolerance:
                heapq.heappush(heap, (-bound, a, b, fa, fb))
            created += 1

    maximum_upper = max(best_value, -heap[0][0] if heap else best_value)
    gap = max(0.0, maximum_upper - best_value)
    return GlobalMaximumResult(
        x=float(best_x),
        value=float(best_value),
        upper_bound=float(maximum_upper),
        bound_gap=float(gap),
        certified=bool(gap <= reference.value_tolerance),
        intervals_created=created,
        evaluations=evaluations,
    )


def certified_gp_supremum(
    draw: FourierGPDraw,
    reference: GlobalReferenceSpec,
) -> GPSupremumResult:
    """Return a certified bracket for sup_[0,1] |G|."""
    second = draw.derivative_bound(2)
    positive = certified_global_maximum(
        lambda x: draw.evaluate(x),
        second_derivative_bound=second,
        reference=reference,
    )
    negative = certified_global_maximum(
        lambda x: -draw.evaluate(x),
        second_derivative_bound=second,
        reference=reference,
    )
    winner = positive if positive.value >= negative.value else negative
    lower = max(positive.value, negative.value)
    upper = max(positive.upper_bound, negative.upper_bound)
    return GPSupremumResult(
        lower_bound=float(lower),
        upper_bound=float(upper),
        bound_gap=float(max(0.0, upper - lower)),
        certified=positive.certified and negative.certified,
        maximizing_x=winner.x,
        intervals_created=positive.intervals_created + negative.intervals_created,
        evaluations=positive.evaluations + negative.evaluations,
    )


# The pure analytic interface now lives in ``continuous_gp_core``.  Rebinding
# here preserves every legacy import and result type while the §3.6.6 adapter
# continues to own only its manifest, condition cube, persistence, and plots.
from experiments.policy_lcb.continuous_gp_core import (  # noqa: E402
    AnalyticUniformCertificate as AnalyticUniformCertificate,
    FiniteFourierGPSpec as FiniteFourierGPSpec,
    FourierGPDraw as FourierGPDraw,
    GPSupremumResult as GPSupremumResult,
    GlobalMaximumResult as GlobalMaximumResult,
    GlobalReferenceSpec as GlobalReferenceSpec,
    SmoothClippedUncertaintySpec as SmoothClippedUncertaintySpec,
    UniformConfidenceSpec as UniformConfidenceSpec,
    analytic_uniform_certificate as analytic_uniform_certificate,
    certified_global_maximum as certified_global_maximum,
    certified_gp_supremum as certified_gp_supremum,
    smooth_clipped_uncertainty as smooth_clipped_uncertainty,
    smoothstep as smoothstep,
    true_regret as true_regret,
    true_value as true_value,
    uncertainty_derivative_bound as uncertainty_derivative_bound,
)


@dataclass(frozen=True)
class ContinuousGPVariableLCBSpec:
    """Resolved inputs for the paired continuous-GP experiment."""

    domain: tuple[float, float]
    gp: FiniteFourierGPSpec
    uncertainty: SmoothClippedUncertaintySpec
    noise_scales: tuple[float, ...]
    confidence: UniformConfidenceSpec
    global_reference: GlobalReferenceSpec
    optimizer: ContinuousGPOptimizerSpec
    master_gp_seed: int
    master_optimizer_seed: int
    reporting_seed: int
    run_seeds: tuple[int, ...]
    optimizer_run_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        domain = tuple(float(value) for value in self.domain)
        if domain != (0.0, 1.0):
            raise ValueError("domain must be exactly [0, 1].")
        scales = tuple(float(value) for value in self.noise_scales)
        if not scales or tuple(sorted(set(scales))) != scales:
            raise ValueError("surrogate.noise_scales must be unique and increasing.")
        if any(not np.isfinite(value) or value < 0.0 for value in scales):
            raise ValueError("surrogate.noise_scales must be finite and non-negative.")
        if any(not 0.0 <= center <= 1.0 for center in self.uncertainty.centers):
            raise ValueError("uncertainty.centers must lie in [0, 1].")
        run_seeds = tuple(int(value) for value in self.run_seeds)
        optimizer_seeds = tuple(int(value) for value in self.optimizer_run_seeds)
        if not run_seeds or len(set(run_seeds)) != len(run_seeds) or min(run_seeds) < 0:
            raise ValueError("seeds.run_seeds must be unique and non-negative.")
        if not set(optimizer_seeds).issubset(run_seeds):
            raise ValueError("seeds.optimizer_run_seeds must be a subset of run_seeds.")
        if any(int(value) < 0 for value in (self.master_gp_seed, self.master_optimizer_seed, self.reporting_seed)):
            raise ValueError("master and reporting seeds must be non-negative.")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "noise_scales", scales)
        object.__setattr__(self, "run_seeds", run_seeds)
        object.__setattr__(self, "optimizer_run_seeds", optimizer_seeds)


@dataclass(frozen=True)
class ContinuousGPVariableLCBLaunchSpec(PolicyLCBLaunchSpec):
    """Launch settings for the continuous-GP adapter."""


@dataclass(frozen=True)
class ContinuousGPVariableLCBManifest:
    """Resolved ``continuous_gp_variable_lcb`` manifest."""

    name: str
    spec: ContinuousGPVariableLCBSpec
    launch: ContinuousGPVariableLCBLaunchSpec
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / path_part(self.name)

    def seed_dir(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / "seeds" / f"seed-{run_seed}"

    def seed_result_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "result.json"

    def seed_trajectory_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "trajectories.npz"


def continuous_gp_seed_for_run(spec: ContinuousGPVariableLCBSpec, run_seed: int) -> int:
    """Derive one GP coefficient seed without involving any sweep axis."""
    if int(run_seed) not in spec.run_seeds:
        raise ValueError(f"Unknown run seed {run_seed}.")
    return derive_seed(spec.master_gp_seed, f"continuous-gp-variable-lcb:run:{int(run_seed)}")


def continuous_gp_optimizer_seed(spec: ContinuousGPVariableLCBSpec) -> int:
    """Return the fixed paired Stein perturbation stream."""
    return derive_seed(spec.master_optimizer_seed, "continuous-gp-variable-lcb:stein")


def draw_fourier_gp(spec: FiniteFourierGPSpec, seed: int) -> FourierGPDraw:
    """Draw one replayable coefficient vector from its dedicated seed."""
    rng = rng_from_seed(int(seed))
    return FourierGPDraw(
        spec=spec,
        a=tuple(float(value) for value in rng.normal(size=spec.rank)),
        b=tuple(float(value) for value in rng.normal(size=spec.rank)),
    )


def _average_uncertainty(center: float, uncertainty: SmoothClippedUncertaintySpec) -> float:
    value, _ = quad(
        lambda x: smooth_clipped_uncertainty(x, center, uncertainty),
        0.0,
        1.0,
        points=[point for point in (center - uncertainty.ramp_radius, center, center + uncertainty.ramp_radius) if 0.0 < point < 1.0],
        epsabs=1e-12,
    )
    return float(value)


def _maximum_uncertainty(center: float, uncertainty: SmoothClippedUncertaintySpec) -> float:
    farthest = 0.0 if center >= 0.5 else 1.0
    return float(smooth_clipped_uncertainty(farthest, center, uncertainty))


def _selector_result(
    *,
    run_seed: int,
    gp_seed: int,
    center: float,
    noise_scale: float,
    target: TargetName,
    landscape: ContinuousGPLandscape,
    nominal: ContinuousGPLandscape,
    global_result: GlobalMaximumResult,
    draw: FourierGPDraw,
    quantile: float,
    optimum_half_width: float,
) -> GPSelectorResult:
    x = global_result.x
    regret = float(true_regret(x))
    slack = None if target == "nominal" else 2.0 * optimum_half_width - regret
    return GPSelectorResult(
        run_seed=run_seed,
        gp_seed=gp_seed,
        uncertainty_center=center,
        noise_scale=noise_scale,
        target=target,
        selected_x=x,
        selected_true_value=float(true_value(x)),
        selected_surrogate_value=float(nominal.evaluate(x)),
        selected_target_value=float(landscape.evaluate(x)),
        regret=regret,
        distance_to_optimum=abs(x - 0.5),
        distance_to_uncertainty_center=abs(x - center),
        selected_point_covered=bool(noise_scale == 0.0 or abs(draw.evaluate(x)) <= quantile),
        global_upper_bound=global_result.upper_bound,
        global_bound_gap=global_result.bound_gap,
        global_reference_certified=global_result.certified,
        certificate_slack=slack,
    )


def _run_optimizer_start(
    *,
    spec: ContinuousGPVariableLCBSpec,
    landscape: ContinuousGPLandscape,
    global_result: GlobalMaximumResult,
    run_seed: int,
    gp_seed: int,
    optimizer_seed: int,
    center: float,
    noise_scale: float,
    target: TargetName,
    estimator: EstimatorName,
    start: float,
    epsilon_samples: np.ndarray,
    quantile: float,
) -> tuple[GPOptimizerFinalResult, tuple[GPOptimizerTrajectoryRow, ...]]:
    optimizer = spec.optimizer
    radius = optimizer.perturbation_radius
    x = float(start)
    rows: list[GPOptimizerTrajectoryRow] = []

    def append(step: int) -> None:
        value = float(landscape.evaluate(x))
        rows.append(
            GPOptimizerTrajectoryRow(
                run_seed=run_seed,
                uncertainty_center=center,
                noise_scale=noise_scale,
                target=target,
                estimator=estimator,
                start_x=start,
                step=step,
                x=x,
                target_value=value,
                true_regret=float(true_regret(x)),
                optimization_gap=max(0.0, global_result.value - value),
            )
        )

    append(0)
    for step in range(1, optimizer.t_steps + 1):
        if estimator == "first_order":
            gradient = float(landscape.evaluate(x, 1))
        elif estimator == "finite_difference":
            gradient = float(
                (landscape.evaluate(x + radius) - landscape.evaluate(x - radius))
                / (2.0 * radius)
            )
        else:
            epsilon = epsilon_samples[step - 1]
            plus = np.asarray(landscape.evaluate(x + radius * epsilon), dtype=float)
            minus = np.asarray(landscape.evaluate(x - radius * epsilon), dtype=float)
            gradient = float(np.mean((plus - minus) * epsilon) / (2.0 * radius))
        x = float(np.clip(x + optimizer.step_size * gradient, 0.0, 1.0))
        if step % optimizer.checkpoint_every == 0 or step == optimizer.t_steps:
            append(step)

    final_value = float(landscape.evaluate(x))
    regret = float(true_regret(x))
    optimization_gap = max(0.0, global_result.value - final_value)
    if target == "variable_lcb":
        optimum_width = noise_scale * quantile * float(
            smooth_clipped_uncertainty(0.5, center, spec.uncertainty)
        )
        certificate_bound = 2.0 * optimum_width + optimization_gap
        certificate_slack = certificate_bound - regret
    else:
        certificate_bound = None
        certificate_slack = None
    return (
        GPOptimizerFinalResult(
            run_seed=run_seed,
            gp_seed=gp_seed,
            optimizer_seed=optimizer_seed,
            uncertainty_center=center,
            noise_scale=noise_scale,
            target=target,
            estimator=estimator,
            start_x=start,
            final_x=x,
            final_target_value=final_value,
            final_true_value=float(true_value(x)),
            true_regret=regret,
            global_target_value=global_result.value,
            optimization_gap=optimization_gap,
            distance_to_global_x=abs(x - global_result.x),
            selected_point_covered=bool(noise_scale == 0.0 or abs(landscape.draw.evaluate(x)) <= quantile),  # type: ignore[union-attr]
            certificate_bound=certificate_bound,
            certificate_slack=certificate_slack,
        ),
        tuple(rows),
    )


def evaluate_continuous_gp_variable_lcb_seed(
    spec: ContinuousGPVariableLCBSpec,
    run_seed: int,
) -> ContinuousGPSeedResult:
    """Draw one analytic GP and evaluate its full paired condition cube."""
    gp_seed = continuous_gp_seed_for_run(spec, run_seed)
    draw = draw_fourier_gp(spec.gp, gp_seed)
    return evaluate_continuous_gp_variable_lcb_draw(
        spec,
        run_seed=run_seed,
        gp_seed=gp_seed,
        draw=draw,
    )


def evaluate_continuous_gp_variable_lcb_draw(
    spec: ContinuousGPVariableLCBSpec,
    *,
    run_seed: int,
    gp_seed: int,
    draw: FourierGPDraw,
) -> ContinuousGPSeedResult:
    """Evaluate all selectors and the optional optimizer subset for one path."""
    if draw.spec != spec.gp:
        raise ValueError("draw GP specification does not match the experiment.")
    certificate = analytic_uniform_certificate(spec.gp, spec.confidence)
    supremum = certified_gp_supremum(draw, spec.global_reference)
    coverage_decided = supremum.upper_bound <= certificate.quantile or supremum.lower_bound > certificate.quantile
    covered = supremum.upper_bound <= certificate.quantile
    optimizer_seed = continuous_gp_optimizer_seed(spec)
    epsilon_samples = rng_from_seed(optimizer_seed).normal(
        size=(spec.optimizer.t_steps, spec.optimizer.n_stein_perturbations)
    )
    run_optimizers = int(run_seed) in spec.optimizer_run_seeds

    conditions: list[GPConditionResult] = []
    selectors: list[GPSelectorResult] = []
    finals: list[GPOptimizerFinalResult] = []
    trajectories: list[GPOptimizerTrajectoryRow] = []
    zero_optimizer_cache: tuple[tuple[GPOptimizerFinalResult, ...], tuple[GPOptimizerTrajectoryRow, ...]] | None = None

    for center in spec.uncertainty.centers:
        average_sigma = _average_uncertainty(center, spec.uncertainty)
        maximum_sigma = _maximum_uncertainty(center, spec.uncertainty)
        optimum_sigma = float(smooth_clipped_uncertainty(0.5, center, spec.uncertainty))
        for noise_scale in spec.noise_scales:
            nominal = ContinuousGPLandscape(
                draw, spec.uncertainty, center, noise_scale, certificate.quantile, "nominal"
            )
            lcb = ContinuousGPLandscape(
                draw, spec.uncertainty, center, noise_scale, certificate.quantile, "variable_lcb"
            )
            deterministic = ContinuousGPLandscape(
                None, spec.uncertainty, center, noise_scale, certificate.quantile, "variable_lcb"
            )
            if noise_scale == 0.0:
                exact = GlobalMaximumResult(0.5, 1.25, 1.25, 0.0, True, 0, 0)
                nominal_global = lcb_global = deterministic_global = exact
            else:
                nominal_global = certified_global_maximum(
                    nominal.evaluate,
                    second_derivative_bound=nominal.second_derivative_bound(),
                    reference=spec.global_reference,
                    breakpoints=nominal.breakpoints(),
                )
                lcb_global = certified_global_maximum(
                    lcb.evaluate,
                    second_derivative_bound=lcb.second_derivative_bound(),
                    reference=spec.global_reference,
                    breakpoints=lcb.breakpoints(),
                )
                deterministic_global = certified_global_maximum(
                    deterministic.evaluate,
                    second_derivative_bound=deterministic.second_derivative_bound(),
                    reference=spec.global_reference,
                    breakpoints=deterministic.breakpoints(),
                )
            optimum_half_width = noise_scale * certificate.quantile * optimum_sigma
            conditions.append(
                GPConditionResult(
                    run_seed=int(run_seed),
                    gp_seed=int(gp_seed),
                    uncertainty_center=center,
                    noise_scale=noise_scale,
                    quantile=certificate.quantile,
                    simultaneous_coverage=bool(noise_scale == 0.0 or covered),
                    coverage_certified=bool(noise_scale == 0.0 or coverage_decided),
                    average_half_width=noise_scale * certificate.quantile * average_sigma,
                    maximum_half_width=noise_scale * certificate.quantile * maximum_sigma,
                    optimum_half_width=optimum_half_width,
                    deterministic_target_x=deterministic_global.x,
                    deterministic_target_certified=deterministic_global.certified,
                    optimum_lower_bound_gap=float(true_value(0.5) - lcb.evaluate(0.5)),
                    lcb_regret_certificate=2.0 * optimum_half_width,
                )
            )
            target_items = (
                ("nominal", nominal, nominal_global),
                ("variable_lcb", lcb, lcb_global),
            )
            for target, landscape, global_result in target_items:
                selectors.append(
                    _selector_result(
                        run_seed=int(run_seed),
                        gp_seed=int(gp_seed),
                        center=center,
                        noise_scale=noise_scale,
                        target=target,
                        landscape=landscape,
                        nominal=nominal,
                        global_result=global_result,
                        draw=draw,
                        quantile=certificate.quantile,
                        optimum_half_width=optimum_half_width,
                    )
                )
                if not run_optimizers:
                    continue
                if noise_scale == 0.0 and zero_optimizer_cache is not None:
                    cached_finals, cached_trajectories = zero_optimizer_cache
                    for row in cached_finals:
                        if target == "variable_lcb":
                            finals.append(
                                replace(
                                    row,
                                    uncertainty_center=center,
                                    target=target,
                                    certificate_bound=row.optimization_gap,
                                    certificate_slack=row.optimization_gap - row.true_regret,
                                )
                            )
                        else:
                            finals.append(
                                replace(row, uncertainty_center=center, target=target)
                            )
                    trajectories.extend(
                        replace(row, uncertainty_center=center, target=target)
                        for row in cached_trajectories
                    )
                    continue
                current_finals: list[GPOptimizerFinalResult] = []
                current_trajectories: list[GPOptimizerTrajectoryRow] = []
                for estimator in spec.optimizer.enabled_estimators:
                    for start in spec.optimizer.starts:
                        final, trace = _run_optimizer_start(
                            spec=spec,
                            landscape=landscape,
                            global_result=global_result,
                            run_seed=int(run_seed),
                            gp_seed=int(gp_seed),
                            optimizer_seed=optimizer_seed,
                            center=center,
                            noise_scale=noise_scale,
                            target=target,
                            estimator=estimator,
                            start=start,
                            epsilon_samples=epsilon_samples,
                            quantile=certificate.quantile,
                        )
                        current_finals.append(final)
                        current_trajectories.extend(trace)
                finals.extend(current_finals)
                trajectories.extend(current_trajectories)
                if noise_scale == 0.0 and zero_optimizer_cache is None:
                    zero_optimizer_cache = (tuple(current_finals), tuple(current_trajectories))

    return ContinuousGPSeedResult(
        run_seed=int(run_seed),
        gp_seed=int(gp_seed),
        optimizer_seed=optimizer_seed,
        a_coefficients=draw.a,
        b_coefficients=draw.b,
        gp_supremum=supremum,
        conditions=tuple(conditions),
        selectors=tuple(selectors),
        optimizer_finals=tuple(finals),
        trajectories=tuple(trajectories),
    )


def _range_from_payload(payload: Mapping[str, Any], label: str) -> tuple[int, ...]:
    if payload.get("type") != "range":
        raise ValueError(f"{label}.type must be 'range'.")
    start = int(payload.get("start", -1))
    count = int(payload.get("count", 0))
    if start < 0 or count <= 0:
        raise ValueError(f"{label} start must be non-negative and count positive.")
    return tuple(range(start, start + count))


def load_continuous_gp_variable_lcb_manifest(
    path: str | Path,
) -> ContinuousGPVariableLCBManifest:
    """Load and validate a ``continuous_gp_variable_lcb`` manifest."""
    manifest_path = Path(path)
    return parse_continuous_gp_variable_lcb_manifest(
        read_json(manifest_path), source_path=manifest_path
    )


def parse_continuous_gp_variable_lcb_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ContinuousGPVariableLCBManifest:
    """Resolve the explicit continuous representation and seed contract."""
    if not isinstance(payload, Mapping):
        raise ValueError("Continuous-GP manifest must be a JSON object.")
    if payload.get("kind") != "continuous_gp_variable_lcb":
        raise ValueError("Manifest kind must be 'continuous_gp_variable_lcb'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Manifest name must be non-empty.")
    domain = number_sequence(payload.get("domain"), "domain")
    if len(domain) != 2:
        raise ValueError("domain must contain two endpoints.")

    objective = required_mapping(payload, "true_value")
    if objective.get("type") != "concave_quadratic" or float(objective.get("linear", np.nan)) != 5.0 or float(objective.get("quadratic", np.nan)) != 5.0:
        raise ValueError("true_value must be concave_quadratic with linear=quadratic=5.")
    gp_payload = required_mapping(payload, "gp")
    if gp_payload.get("type") != "deterministic_spectral_finite_rank":
        raise ValueError("gp.type must be 'deterministic_spectral_finite_rank'.")
    uncertainty_payload = required_mapping(payload, "uncertainty")
    if uncertainty_payload.get("type") != "smooth_clipped_distance_ramp":
        raise ValueError("uncertainty.type must be 'smooth_clipped_distance_ramp'.")
    surrogate = required_mapping(payload, "surrogate")
    if surrogate.get("type") != "scaled_nonstationary_gp":
        raise ValueError("surrogate.type must be 'scaled_nonstationary_gp'.")
    confidence_payload = required_mapping(payload, "confidence")
    if confidence_payload.get("type") != "bonferroni_net_smoothness":
        raise ValueError("confidence.type must be 'bonferroni_net_smoothness'.")
    reference_payload = required_mapping(payload, "global_reference")
    if reference_payload.get("type") != "certified_branch_and_bound":
        raise ValueError("global_reference.type must be 'certified_branch_and_bound'.")
    optimizer_payload = required_mapping(payload, "optimizer")
    if optimizer_payload.get("step_rule") != "projected_constant":
        raise ValueError("optimizer.step_rule must be 'projected_constant'.")
    if optimizer_payload.get("probe_domain") != "analytic_real_line_extension":
        raise ValueError("optimizer.probe_domain must document the analytic real-line extension.")
    estimators = optimizer_payload.get("enabled_estimators")
    if not isinstance(estimators, Sequence) or isinstance(estimators, (str, bytes)):
        raise ValueError("optimizer.enabled_estimators must be a sequence.")
    starts = number_sequence(optimizer_payload.get("starts"), "optimizer.starts")

    seeds = required_mapping(payload, "seeds")
    run_seeds = _range_from_payload(required_mapping(seeds, "run_seeds"), "seeds.run_seeds")
    optimizer_run_seeds = _range_from_payload(
        required_mapping(seeds, "optimizer_run_seeds"), "seeds.optimizer_run_seeds"
    )
    launch_payload = required_mapping(payload, "launch")
    mode = str(launch_payload.get("mode") or "")
    array = str(launch_payload.get("array") or "")
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("launch.mode must be auto, local, or slurm.")
    if array not in {"none", "seed"}:
        raise ValueError("launch.array must be none or seed.")
    parallel_raw = launch_payload.get("array_max_parallel")
    parallel = None if parallel_raw is None else int(parallel_raw)
    if parallel is not None and parallel <= 0:
        raise ValueError("launch.array_max_parallel must be positive.")

    spec = ContinuousGPVariableLCBSpec(
        domain=(domain[0], domain[1]),
        gp=FiniteFourierGPSpec(
            rank=int(gp_payload.get("rank", 0)),
            lengthscale=float(gp_payload.get("lengthscale", np.nan)),
        ),
        uncertainty=SmoothClippedUncertaintySpec(
            centers=number_sequence(uncertainty_payload.get("centers"), "uncertainty.centers"),
            minimum=float(uncertainty_payload.get("minimum", np.nan)),
            maximum=float(uncertainty_payload.get("maximum", np.nan)),
            ramp_radius=float(uncertainty_payload.get("ramp_radius", np.nan)),
        ),
        noise_scales=number_sequence(surrogate.get("noise_scales"), "surrogate.noise_scales"),
        confidence=UniformConfidenceSpec(
            delta=float(confidence_payload.get("delta", np.nan)),
            net_count=int(confidence_payload.get("net_count", 0)),
        ),
        global_reference=GlobalReferenceSpec(
            value_tolerance=float(reference_payload.get("value_tolerance", np.nan)),
            max_intervals=int(reference_payload.get("max_intervals", 0)),
            initial_grid_count=int(reference_payload.get("initial_grid_count", 65)),
        ),
        optimizer=ContinuousGPOptimizerSpec(
            enabled_estimators=tuple(str(value) for value in estimators),  # type: ignore[arg-type]
            starts=starts,
            t_steps=int(optimizer_payload.get("t_steps", 0)),
            step_size=float(optimizer_payload.get("step_size", np.nan)),
            perturbation_radius=float(optimizer_payload.get("perturbation_radius", np.nan)),
            n_stein_perturbations=int(optimizer_payload.get("n_stein_perturbations", 0)),
            checkpoint_every=int(optimizer_payload.get("checkpoint_every", 0)),
        ),
        master_gp_seed=int(seeds.get("master_gp_seed", -1)),
        master_optimizer_seed=int(seeds.get("master_optimizer_seed", -1)),
        reporting_seed=int(seeds.get("reporting_seed", -1)),
        run_seeds=run_seeds,
        optimizer_run_seeds=optimizer_run_seeds,
    )
    return ContinuousGPVariableLCBManifest(
        name=name,
        spec=spec,
        launch=ContinuousGPVariableLCBLaunchSpec(
            mode=mode,  # type: ignore[arg-type]
            array=array,  # type: ignore[arg-type]
            array_max_parallel=parallel,
        ),
        source_path=None if source_path is None else Path(source_path),
    )


def continuous_gp_variable_lcb_seed_complete(
    manifest: ContinuousGPVariableLCBManifest,
    run_seed: int,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return whether all durable files required for one seed exist."""
    if not manifest.seed_result_path(run_seed, runs_root).exists():
        return False
    return (
        int(run_seed) not in manifest.spec.optimizer_run_seeds
        or manifest.seed_trajectory_path(run_seed, runs_root).exists()
    )


def _write_trajectory_rows(path: Path, rows: Sequence[GPOptimizerTrajectoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = {
        "run_seed": np.asarray([row.run_seed for row in rows], dtype=np.int64),
        "uncertainty_center": np.asarray([row.uncertainty_center for row in rows]),
        "noise_scale": np.asarray([row.noise_scale for row in rows]),
        "target": np.asarray([row.target for row in rows], dtype="U16"),
        "estimator": np.asarray([row.estimator for row in rows], dtype="U24"),
        "start_x": np.asarray([row.start_x for row in rows]),
        "step": np.asarray([row.step for row in rows], dtype=np.int64),
        "x": np.asarray([row.x for row in rows]),
        "target_value": np.asarray([row.target_value for row in rows]),
        "true_regret": np.asarray([row.true_regret for row in rows]),
        "optimization_gap": np.asarray([row.optimization_gap for row in rows]),
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **columns)
    temporary.replace(path)


def _read_continuous_gp_seed_result(path: Path) -> ContinuousGPSeedResult:
    payload = read_json(path)
    return ContinuousGPSeedResult(
        run_seed=int(payload["run_seed"]),
        gp_seed=int(payload["gp_seed"]),
        optimizer_seed=int(payload["optimizer_seed"]),
        a_coefficients=tuple(float(value) for value in payload["a_coefficients"]),
        b_coefficients=tuple(float(value) for value in payload["b_coefficients"]),
        gp_supremum=GPSupremumResult(**payload["gp_supremum"]),
        conditions=tuple(GPConditionResult(**row) for row in payload["conditions"]),
        selectors=tuple(GPSelectorResult(**row) for row in payload["selectors"]),
        optimizer_finals=tuple(GPOptimizerFinalResult(**row) for row in payload["optimizer_finals"]),
        trajectories=(),
    )


def run_continuous_gp_variable_lcb_manifest_seed(
    manifest: ContinuousGPVariableLCBManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run and atomically persist one replayable GP seed task."""
    if index < 0 or index >= len(manifest.spec.run_seeds):
        raise IndexError(f"Seed task index {index} is out of range.")
    run_seed = manifest.spec.run_seeds[index]
    write_continuous_gp_variable_lcb_experiment_readme(manifest, runs_root=runs_root)
    if continuous_gp_variable_lcb_seed_complete(manifest, run_seed, runs_root=runs_root) and not force:
        return {"project_dir": str(manifest.project_dir(runs_root)), "run_seed": run_seed, "skipped": True}
    result = evaluate_continuous_gp_variable_lcb_seed(manifest.spec, run_seed)
    payload = asdict(result)
    payload.pop("trajectories")
    write_json_atomic(manifest.seed_result_path(run_seed, runs_root), payload)
    if run_seed in manifest.spec.optimizer_run_seeds:
        _write_trajectory_rows(manifest.seed_trajectory_path(run_seed, runs_root), result.trajectories)
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "run_seed": run_seed,
        "skipped": False,
        "n_condition_rows": len(result.conditions),
        "n_selector_rows": len(result.selectors),
        "n_optimizer_rows": len(result.optimizer_finals),
    }


def run_continuous_gp_variable_lcb_manifest_serial(
    manifest: ContinuousGPVariableLCBManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run all seeds serially and then collect aggregate outputs."""
    payloads = [
        run_continuous_gp_variable_lcb_manifest_seed(
            manifest, index, runs_root=runs_root, force=force
        )
        for index in range(len(manifest.spec.run_seeds))
    ]
    collected = collect_continuous_gp_variable_lcb_outputs(manifest, runs_root=runs_root)
    return {
        **collected,
        "n_skipped_seeds": sum(bool(row["skipped"]) for row in payloads),
    }


def _aggregate_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...],
    numeric_fields: tuple[str, ...],
    boolean_fields: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        group = groups[key]
        record: dict[str, object] = dict(zip(group_fields, key))
        record["n"] = len(group)
        for field in numeric_fields:
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            record[f"{field}_mean"] = float(np.mean(values))
            record[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            record[f"{field}_q05"] = float(np.quantile(values, 0.05))
            record[f"{field}_q95"] = float(np.quantile(values, 0.95))
        for field in boolean_fields:
            record[f"{field}_rate"] = float(np.mean([bool(row[field]) for row in group]))
        output.append(record)
    return output


def _trajectory_summary(
    manifest: ContinuousGPVariableLCBManifest,
    *,
    runs_root: str | Path | None,
) -> list[dict[str, object]]:
    fields = ("uncertainty_center", "noise_scale", "target", "estimator", "start_x", "step")
    template: dict[str, np.ndarray] | None = None
    gap_sum: np.ndarray | None = None
    gap_sq: np.ndarray | None = None
    regret_sum: np.ndarray | None = None
    regret_sq: np.ndarray | None = None
    success_2: np.ndarray | None = None
    success_3: np.ndarray | None = None
    success_4: np.ndarray | None = None
    x_sum: np.ndarray | None = None
    count = 0
    for seed in manifest.spec.optimizer_run_seeds:
        path = manifest.seed_trajectory_path(seed, runs_root)
        with np.load(path, allow_pickle=False) as data:
            current = {field: np.asarray(data[field]) for field in fields}
            if template is None:
                template = {field: values.copy() for field, values in current.items()}
                size = len(template["step"])
                gap_sum = np.zeros(size, dtype=float)
                gap_sq = np.zeros(size, dtype=float)
                regret_sum = np.zeros(size, dtype=float)
                regret_sq = np.zeros(size, dtype=float)
                success_2 = np.zeros(size, dtype=float)
                success_3 = np.zeros(size, dtype=float)
                success_4 = np.zeros(size, dtype=float)
                x_sum = np.zeros(size, dtype=float)
            elif any(
                current[field].shape != template[field].shape
                or not np.array_equal(current[field], template[field])
                for field in fields
            ):
                raise ValueError(
                    f"Optimizer trajectory layout for seed {seed} does not match the first seed."
                )

            gap = np.asarray(data["optimization_gap"], dtype=float)
            regret = np.asarray(data["true_regret"], dtype=float)
            selected_x = np.asarray(data["x"], dtype=float)
            if template is None or gap.shape != template["step"].shape:
                raise ValueError(f"Optimizer trajectory values for seed {seed} have invalid shape.")
            assert gap_sum is not None and gap_sq is not None
            assert regret_sum is not None and regret_sq is not None
            assert success_2 is not None and success_3 is not None and success_4 is not None
            assert x_sum is not None
            gap_sum += gap
            gap_sq += np.square(gap)
            regret_sum += regret
            regret_sq += np.square(regret)
            success_2 += gap <= 1e-2
            success_3 += gap <= 1e-3
            success_4 += gap <= 1e-4
            x_sum += selected_x
            count += 1

    if template is None or count == 0:
        return []
    assert gap_sum is not None and gap_sq is not None
    assert regret_sum is not None and regret_sq is not None
    assert success_2 is not None and success_3 is not None and success_4 is not None
    assert x_sum is not None
    output: list[dict[str, object]] = []
    for index in range(len(template["step"])):
        gap_var = (
            max(0.0, (gap_sq[index] - gap_sum[index] ** 2 / count) / (count - 1.0))
            if count > 1
            else 0.0
        )
        regret_var = (
            max(
                0.0,
                (regret_sq[index] - regret_sum[index] ** 2 / count) / (count - 1.0),
            )
            if count > 1
            else 0.0
        )
        record: dict[str, object] = {
            "uncertainty_center": float(template["uncertainty_center"][index]),
            "noise_scale": float(template["noise_scale"][index]),
            "target": str(template["target"][index]),
            "estimator": str(template["estimator"][index]),
            "start_x": float(template["start_x"][index]),
            "step": int(template["step"][index]),
        }
        record.update(
            n=count,
            optimization_gap_mean=gap_sum[index] / count,
            optimization_gap_std=float(np.sqrt(gap_var)),
            true_regret_mean=regret_sum[index] / count,
            true_regret_std=float(np.sqrt(regret_var)),
            selected_x_mean=x_sum[index] / count,
            success_1e_2_rate=success_2[index] / count,
            success_1e_3_rate=success_3[index] / count,
            success_1e_4_rate=success_4[index] / count,
        )
        output.append(record)
    output.sort(key=lambda row: tuple(str(row[field]) for field in fields))
    return output


def collect_continuous_gp_variable_lcb_outputs(
    manifest: ContinuousGPVariableLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, object]:
    """Collect raw tables, compact summaries, and deterministic plots."""
    project_dir = manifest.project_dir(runs_root)
    write_continuous_gp_variable_lcb_experiment_readme(manifest, runs_root=runs_root)
    results = [
        _read_continuous_gp_seed_result(manifest.seed_result_path(seed, runs_root))
        for seed in manifest.spec.run_seeds
    ]
    gp_rows = [
        {
            "run_seed": result.run_seed,
            "gp_seed": result.gp_seed,
            **asdict(result.gp_supremum),
            "quantile": analytic_uniform_certificate(manifest.spec.gp, manifest.spec.confidence).quantile,
            "simultaneous_coverage": result.gp_supremum.upper_bound
            <= analytic_uniform_certificate(manifest.spec.gp, manifest.spec.confidence).quantile,
        }
        for result in results
    ]
    condition_rows = [asdict(row) for result in results for row in result.conditions]
    selector_rows = [asdict(row) for result in results for row in result.selectors]
    optimizer_rows = [asdict(row) for result in results for row in result.optimizer_finals]
    condition_summary = _aggregate_records(
        condition_rows,
        group_fields=("uncertainty_center", "noise_scale"),
        numeric_fields=("average_half_width", "maximum_half_width", "optimum_half_width", "deterministic_target_x", "optimum_lower_bound_gap", "lcb_regret_certificate"),
        boolean_fields=("simultaneous_coverage", "coverage_certified", "deterministic_target_certified"),
    )
    selector_summary = _aggregate_records(
        selector_rows,
        group_fields=("uncertainty_center", "noise_scale", "target"),
        numeric_fields=("selected_x", "regret", "distance_to_optimum", "distance_to_uncertainty_center", "global_bound_gap"),
        boolean_fields=("selected_point_covered", "global_reference_certified"),
    )
    optimizer_summary = _aggregate_records(
        optimizer_rows,
        group_fields=("uncertainty_center", "noise_scale", "target", "estimator", "start_x"),
        numeric_fields=("final_x", "true_regret", "optimization_gap", "distance_to_global_x"),
        boolean_fields=("selected_point_covered",),
    ) if optimizer_rows else []
    trajectory_summary = _trajectory_summary(manifest, runs_root=runs_root) if manifest.spec.optimizer_run_seeds else []
    coverage_values = np.asarray([bool(row["simultaneous_coverage"]) for row in gp_rows], dtype=float)
    reporting_rng = rng_from_seed(manifest.spec.reporting_seed)
    bootstrap_means = np.mean(
        reporting_rng.choice(coverage_values, size=(2000, len(coverage_values)), replace=True),
        axis=1,
    )
    coverage_summary = [
        {
            "n_seeds": len(coverage_values),
            "analytic_target": 1.0 - manifest.spec.confidence.delta,
            "empirical_coverage": float(np.mean(coverage_values)),
            "bootstrap_95_lower": float(np.quantile(bootstrap_means, 0.025)),
            "bootstrap_95_upper": float(np.quantile(bootstrap_means, 0.975)),
            "reporting_seed": manifest.spec.reporting_seed,
        }
    ]

    tables = (
        ("seed_gp_metrics.csv", gp_rows),
        ("seed_condition_metrics.csv", condition_rows),
        ("seed_selector_metrics.csv", selector_rows),
        ("seed_optimizer_finals.csv", optimizer_rows),
        ("condition_summary.csv", condition_summary),
        ("selector_summary.csv", selector_summary),
        ("optimizer_final_summary.csv", optimizer_summary),
        ("optimizer_trajectory_summary.csv", trajectory_summary),
        ("coverage_summary.csv", coverage_summary),
    )
    for filename, rows in tables:
        if rows:
            write_rows_csv(project_dir / filename, rows, tuple(rows[0]))
    _write_continuous_gp_plots(manifest.spec, results[0], gp_rows, condition_summary, selector_summary, optimizer_summary, trajectory_summary, project_dir / "plots")
    return {
        "project_dir": str(project_dir),
        "n_seed_results": len(results),
        "n_condition_rows": len(condition_rows),
        "n_selector_rows": len(selector_rows),
        "n_optimizer_rows": len(optimizer_rows),
    }


def write_continuous_gp_variable_lcb_experiment_readme(
    manifest: ContinuousGPVariableLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write all representation and confidence decisions beside the results."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    spec = manifest.spec
    certificate = analytic_uniform_certificate(spec.gp, spec.confidence)
    source = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    text = f"""# {manifest.name}

- Manifest: `{source}`
- Domain: continuous `[0, 1]`
- GP: exact finite-rank analytic Fourier process, rank `{spec.gp.rank}`, lengthscale `{spec.gp.lengthscale}`
- Centers `m`: `{list(spec.uncertainty.centers)}`
- Noise/envelope scales `c`: `{list(spec.noise_scales)}`
- Run seeds: `{spec.run_seeds[0]}..{spec.run_seeds[-1]}` ({len(spec.run_seeds)} total)
- Optimizer subset: `{spec.optimizer_run_seeds[0]}..{spec.optimizer_run_seeds[-1]}` ({len(spec.optimizer_run_seeds)} total)
- Dedicated GP / optimizer / reporting seeds: `{spec.master_gp_seed}` / `{spec.master_optimizer_seed}` / `{spec.reporting_seed}`

## Key representation decisions

`G_s(x)` is evaluated directly from its sine/cosine formula at every real-valued query.
It is not sampled on a grid and interpolated. Plotting points are rendered evaluations
only. One seed owns one coefficient vector and reuses it across all `m`, `c`, targets,
starts, and estimators. The point `m` minimizes marginal uncertainty and envelope width;
it need not minimize the realized absolute error.

Optimizer iterates are projected onto `[0,1]`. Finite-difference and Gaussian-Stein
probes use the documented analytic extension of `f`, `sigma_m`, and `G_s` to the real
line before projection of the next iterate.

```text
f(x) = 5*x - 5*x^2
sigma_m(x) = 0.1 + 0.9*h(|x-m|/0.5)
f_hat(x) = f(x) + c*sigma_m(x)*G_s(x)
E(x) = c*q*sigma_m(x)
LCB(x) = f_hat(x) - E(x)
R(x) = f(0.5)-f(x) = 5*(x-0.5)^2
```

## Analytic simultaneous certificate

The envelope multiplier is seed-independent: `q={certificate.quantile:.12g}`.
Bonferroni on `{certificate.net_count}` covering points contributes
`q_net={certificate.net_quantile:.12g}`. The chi-square coefficient event bounds the
between-point movement by `{certificate.remainder:.12g}`. Splitting the failure budget
equally and applying a union bound proves `P(sup_x |G_s(x)| <= q) >= {1-spec.confidence.delta:g}`.
The run seeds verify this statement empirically; they do not calibrate the band.

Global nominal and LCB values are certified to `{spec.global_reference.value_tolerance:g}`
by one-dimensional branch-and-bound. This certifies the reference value, not global
convergence of first-order or zeroth-order methods.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save_plot(fig: Any, path: Path) -> None:
    plt = _load_pyplot()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_continuous_gp_plots(
    spec: ContinuousGPVariableLCBSpec,
    first_result: ContinuousGPSeedResult,
    gp_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    selector_rows: Sequence[Mapping[str, Any]],
    optimizer_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    plot_dir: Path,
) -> None:
    """Render concise validity, geometry, regret, and convergence diagnostics."""
    plt = _load_pyplot()
    plot_dir.mkdir(parents=True, exist_ok=True)
    certificate = analytic_uniform_certificate(spec.gp, spec.confidence)
    draw = FourierGPDraw(spec.gp, first_result.a_coefficients, first_result.b_coefficients)
    center = 0.5 if 0.5 in spec.uncertainty.centers else spec.uncertainty.centers[len(spec.uncertainty.centers) // 2]
    scale = 1.0 if 1.0 in spec.noise_scales else spec.noise_scales[-1]
    x = np.linspace(0.0, 1.0, 1001)
    sigma = smooth_clipped_uncertainty(x, center, spec.uncertainty)
    clean = true_value(x)
    surrogate = clean + scale * sigma * draw.evaluate(x)
    width = scale * certificate.quantile * sigma

    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(x, clean, color="black", linewidth=2, label=r"$f(x)$")
    axis.plot(x, surrogate, color="tab:blue", label=r"$\hat f(x)$")
    axis.fill_between(x, surrogate - width, surrogate + width, color="tab:blue", alpha=0.16, label=r"$\hat f\pm E$")
    axis.plot(x, surrogate - width, color="tab:orange", label=r"LCB $=\hat f-E$")
    axis.axvline(0.5, color="black", linestyle="--", alpha=0.7, label=r"$x^*=0.5$")
    axis.axvline(center, color="tab:green", linestyle=":", label=f"minimum uncertainty m={center:g}")
    axis.set(xlabel="Continuous action x", ylabel="Value", title=f"Analytic GP realization; seed={first_result.run_seed}, c={scale:g}, m={center:g}")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    fig.text(0.5, -0.01, "Curves are rendered evaluations of an analytic Fourier function; no interpolation defines the path.", ha="center", fontsize=9)
    _save_plot(fig, plot_dir / "realized_analytic_landscape.png")

    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    suprema = [float(row["lower_bound"]) for row in gp_rows]
    axis.hist(suprema, bins=min(40, max(5, int(np.sqrt(len(suprema))))), alpha=0.75, color="tab:blue")
    axis.axvline(certificate.quantile, color="tab:red", linewidth=2, label=f"analytic q={certificate.quantile:.3f}")
    coverage = np.mean([bool(row["simultaneous_coverage"]) for row in gp_rows])
    axis.set(xlabel=r"Certified lower bound for $\sup_x|G_s(x)|$", ylabel="Seeds", title=f"Analytic simultaneous band; empirical coverage={coverage:.4f}")
    axis.legend()
    axis.grid(alpha=0.2)
    _save_plot(fig, plot_dir / "simultaneous_coverage.png")

    centers = spec.uncertainty.centers
    columns = min(3, len(centers))
    rows_count = int(np.ceil(len(centers) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(4.4 * columns, 3.5 * rows_count), squeeze=False, sharex=True, sharey=True)
    for axis, current_center in zip(axes.ravel(), centers):
        for target, label, marker in (("nominal", "Nominal", "o"), ("variable_lcb", "Variable LCB", "s")):
            group = sorted(
                [row for row in selector_rows if float(row["uncertainty_center"]) == current_center and row["target"] == target],
                key=lambda row: float(row["noise_scale"]),
            )
            axis.plot([float(row["noise_scale"]) for row in group], [float(row["regret_mean"]) for row in group], marker=marker, label=label)
        axis.set_title(f"m={current_center:g}")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(centers):]:
        axis.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)
    fig.supxlabel("Noise and envelope scale c")
    fig.supylabel(r"Mean true regret $R(x)=5(x-0.5)^2$")
    fig.suptitle("Globally referenced regret by minimum-uncertainty point m", y=0.99)
    fig.subplots_adjust(top=0.82)
    _save_plot(fig, plot_dir / "regret_by_center.png")

    fig, axes = plt.subplots(1, len(spec.noise_scales), figsize=(4.0 * len(spec.noise_scales), 3.8), sharey=True, squeeze=False)
    for axis, current_scale in zip(axes.ravel(), spec.noise_scales):
        for target, label, marker in (("nominal", "Nominal", "o"), ("variable_lcb", "Variable LCB", "s")):
            group = sorted(
                [row for row in selector_rows if float(row["noise_scale"]) == current_scale and row["target"] == target],
                key=lambda row: float(row["uncertainty_center"]),
            )
            axis.plot([float(row["uncertainty_center"]) for row in group], [float(row["selected_x_mean"]) for row in group], marker=marker, label=label)
        axis.plot(centers, centers, color="tab:green", linestyle=":", label="m")
        axis.axhline(0.5, color="black", linestyle="--", label=r"$x^*$")
        axis.set_title(f"c={current_scale:g}")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4)
    fig.supxlabel("Minimum-uncertainty point m")
    fig.supylabel("Mean globally selected x")
    fig.suptitle("Selection follows function, uncertainty, or their LCB compromise", y=0.99)
    fig.subplots_adjust(top=0.73)
    _save_plot(fig, plot_dir / "selected_location_by_center.png")

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for current_center in centers:
        group = sorted(
            [row for row in condition_rows if float(row["uncertainty_center"]) == current_center],
            key=lambda row: float(row["noise_scale"]),
        )
        axis.plot([float(row["noise_scale"]) for row in group], [float(row["average_half_width_mean"]) for row in group], marker="o", label=f"m={current_center:g}")
    axis.set(xlabel="Noise and envelope scale c", ylabel="Mean envelope half-width", title="Analytic envelope tightness")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    _save_plot(fig, plot_dir / "envelope_tightness.png")

    if optimizer_rows:
        methods = tuple(spec.optimizer.enabled_estimators)
        fig, axes = plt.subplots(
            2,
            len(methods),
            figsize=(4.2 * len(methods), 8.2),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, target in enumerate(("nominal", "variable_lcb")):
            for column, method in enumerate(methods):
                axis = axes[row_index, column]
                matrix = np.full((len(centers), len(spec.noise_scales)), np.nan)
                for i, current_center in enumerate(centers):
                    for j, current_scale in enumerate(spec.noise_scales):
                        values = [
                            float(row["optimization_gap_mean"])
                            for row in optimizer_rows
                            if row["target"] == target
                            and row["estimator"] == method
                            and float(row["uncertainty_center"]) == current_center
                            and float(row["noise_scale"]) == current_scale
                        ]
                        if values:
                            matrix[i, j] = float(np.mean(values))
                image = axis.imshow(np.log10(np.maximum(matrix, 1e-12)), origin="lower", aspect="auto", cmap="viridis")
                axis.set_xticks(range(len(spec.noise_scales)), [f"{value:g}" for value in spec.noise_scales])
                axis.set_yticks(range(len(centers)), [f"{value:g}" for value in centers])
                axis.set_title(f"{target}; {method}")
                axis.set_xlabel("c")
                axis.set_ylabel("m")
                fig.colorbar(image, ax=axis, label="log10 mean global gap")
        fig.suptitle("Optimizer gap to certified global reference")
        _save_plot(fig, plot_dir / "optimizer_global_gap_heatmaps.png")

    if trajectory_rows:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
        for axis, target in zip(axes, ("nominal", "variable_lcb")):
            for method in spec.optimizer.enabled_estimators:
                steps = sorted({int(row["step"]) for row in trajectory_rows if row["target"] == target and row["estimator"] == method})
                means = []
                for step in steps:
                    values = [float(row["optimization_gap_mean"]) for row in trajectory_rows if row["target"] == target and row["estimator"] == method and int(row["step"]) == step]
                    means.append(float(np.mean(values)))
                axis.plot(steps, means, label=method)
            axis.set_title(target)
            axis.set_xlabel("Step")
            axis.set_yscale("log")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Mean gap to certified global value")
        axes[1].legend()
        fig.suptitle("Convergence on the predeclared 200-path optimizer subset")
        _save_plot(fig, plot_dir / "optimizer_convergence.png")


__all__ = [
    "AnalyticUniformCertificate",
    "ContinuousGPLandscape",
    "ContinuousGPOptimizerSpec",
    "ContinuousGPSeedResult",
    "ContinuousGPVariableLCBLaunchSpec",
    "ContinuousGPVariableLCBManifest",
    "ContinuousGPVariableLCBSpec",
    "FiniteFourierGPSpec",
    "FourierGPDraw",
    "GPConditionResult",
    "GPOptimizerFinalResult",
    "GPOptimizerTrajectoryRow",
    "GPSelectorResult",
    "GPSupremumResult",
    "GlobalMaximumResult",
    "GlobalReferenceSpec",
    "SmoothClippedUncertaintySpec",
    "UniformConfidenceSpec",
    "analytic_uniform_certificate",
    "certified_global_maximum",
    "certified_gp_supremum",
    "collect_continuous_gp_variable_lcb_outputs",
    "continuous_gp_optimizer_seed",
    "continuous_gp_seed_for_run",
    "continuous_gp_variable_lcb_seed_complete",
    "draw_fourier_gp",
    "evaluate_continuous_gp_variable_lcb_draw",
    "evaluate_continuous_gp_variable_lcb_seed",
    "load_continuous_gp_variable_lcb_manifest",
    "parse_continuous_gp_variable_lcb_manifest",
    "run_continuous_gp_variable_lcb_manifest_seed",
    "run_continuous_gp_variable_lcb_manifest_serial",
    "smooth_clipped_uncertainty",
    "smoothstep",
    "true_regret",
    "true_value",
    "uncertainty_derivative_bound",
    "write_continuous_gp_variable_lcb_experiment_readme",
]
