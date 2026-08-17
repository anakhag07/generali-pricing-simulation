"""Pure analytic machinery shared by continuous finite-Fourier GP experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import heapq
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm


LandscapeTarget = Literal["surrogate", "lower", "violation"]


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
class CertifiedMinimumResult:
    """Certified bracket for a scalar global minimum."""

    x: float
    value: float
    lower_bound: float
    upper_bound: float
    bound_gap: float
    certified: bool
    intervals_created: int
    evaluations: int


@dataclass(frozen=True)
class GPSupremumResult:
    """Certified lower/upper bracket for an absolute function supremum."""

    lower_bound: float
    upper_bound: float
    bound_gap: float
    certified: bool
    maximizing_x: float
    intervals_created: int
    evaluations: int


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


def uncertainty_derivative_bound(
    spec: SmoothClippedUncertaintySpec, derivative: int
) -> float:
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
        self, x: float | Sequence[float] | np.ndarray, derivative: int = 0
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
    gp: FiniteFourierGPSpec, confidence: UniformConfidenceSpec
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


def certified_coverage_probability(
    gp: FiniteFourierGPSpec,
    confidence: UniformConfidenceSpec,
    threshold: float,
) -> float:
    """Invert the original two-sided net certificate at a GP threshold."""
    if np.isnan(threshold) or threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if np.isposinf(threshold):
        return 1.0
    lower_delta = 1e-12
    upper_delta = 1.0 - 1e-12

    def difference(delta: float) -> float:
        return analytic_uniform_certificate(
            gp, UniformConfidenceSpec(delta=delta, net_count=confidence.net_count)
        ).quantile - threshold

    if difference(upper_delta) > 0.0:
        return 0.0
    if difference(lower_delta) < 0.0:
        return 1.0 - lower_delta
    delta = brentq(difference, lower_delta, upper_delta, xtol=1e-13, rtol=1e-13)
    return float(1.0 - delta)


def _product_terms(
    draw: FourierGPDraw,
    uncertainty: SmoothClippedUncertaintySpec,
    center: float,
    x: float | Sequence[float] | np.ndarray,
    derivative: int,
) -> Any:
    sigma0 = smooth_clipped_uncertainty(x, center, uncertainty, 0)
    if derivative == 0:
        return np.asarray(sigma0) * np.asarray(draw.evaluate(x, 0))
    sigma1 = smooth_clipped_uncertainty(x, center, uncertainty, 1)
    if derivative == 1:
        return np.asarray(sigma1) * np.asarray(draw.evaluate(x, 0)) + np.asarray(
            sigma0
        ) * np.asarray(draw.evaluate(x, 1))
    if derivative == 2:
        sigma2 = smooth_clipped_uncertainty(x, center, uncertainty, 2)
        return (
            np.asarray(sigma2) * np.asarray(draw.evaluate(x, 0))
            + 2.0 * np.asarray(sigma1) * np.asarray(draw.evaluate(x, 1))
            + np.asarray(sigma0) * np.asarray(draw.evaluate(x, 2))
        )
    raise ValueError("product derivatives support orders 0, 1, and 2.")


@dataclass(frozen=True)
class DecomposedGPLandscape:
    """Analytic surrogate, lower envelope, or lower-envelope violation."""

    draw: FourierGPDraw
    uncertainty: SmoothClippedUncertaintySpec
    surrogate_center: float
    surrogate_scale: float
    envelope_center: float
    envelope_scale: float
    quantile: float
    target: LandscapeTarget = "lower"

    def __post_init__(self) -> None:
        for name, value in (
            ("surrogate_center", self.surrogate_center),
            ("envelope_center", self.envelope_center),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        for name, value in (
            ("surrogate_scale", self.surrogate_scale),
            ("envelope_scale", self.envelope_scale),
            ("quantile", self.quantile),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def evaluate(
        self, x: float | Sequence[float] | np.ndarray, derivative: int = 0
    ) -> Any:
        """Evaluate the selected analytic target or either derivative."""
        values = np.asarray(x)
        result = (
            np.zeros_like(np.asarray(x, dtype=float))
            if self.target == "violation"
            else np.asarray(true_value(x, derivative), dtype=float)
        )
        if self.surrogate_scale > 0.0:
            result = result + self.surrogate_scale * np.asarray(
                _product_terms(
                    self.draw,
                    self.uncertainty,
                    self.surrogate_center,
                    x,
                    derivative,
                )
            )
        if self.target in {"lower", "violation"} and self.envelope_scale > 0.0:
            result = result - self.envelope_scale * self.quantile * np.asarray(
                smooth_clipped_uncertainty(
                    x, self.envelope_center, self.uncertainty, derivative
                )
            )
        return float(result) if values.ndim == 0 else result

    def second_derivative_bound(self) -> float:
        """Return a global absolute second-derivative bound."""
        sigma0 = uncertainty_derivative_bound(self.uncertainty, 0)
        sigma1 = uncertainty_derivative_bound(self.uncertainty, 1)
        sigma2 = uncertainty_derivative_bound(self.uncertainty, 2)
        bound = 0.0 if self.target == "violation" else 10.0
        if self.surrogate_scale > 0.0:
            bound += self.surrogate_scale * (
                sigma2 * self.draw.derivative_bound(0)
                + 2.0 * sigma1 * self.draw.derivative_bound(1)
                + sigma0 * self.draw.derivative_bound(2)
            )
        if self.target in {"lower", "violation"}:
            bound += self.envelope_scale * self.quantile * sigma2
        return float(bound)

    def breakpoints(self) -> tuple[float, ...]:
        """Return all uncertainty kinks relevant to interval certification."""
        radius = self.uncertainty.ramp_radius
        candidates = [0.0, 1.0]
        for center in (self.surrogate_center, self.envelope_center):
            candidates.extend((center - radius, center, center + radius))
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
    candidates = [(float(x), float(y)) for x, y in zip(points, values)]
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
    draw: FourierGPDraw, reference: GlobalReferenceSpec
) -> GPSupremumResult:
    """Return a certified bracket for sup_[0,1] |G|."""
    return certified_absolute_supremum(
        draw.evaluate,
        second_derivative_bound=draw.derivative_bound(2),
        reference=reference,
    )


def certified_absolute_supremum(
    value_fn: Any,
    *,
    second_derivative_bound: float,
    reference: GlobalReferenceSpec,
    breakpoints: Sequence[float] = (0.0, 1.0),
) -> GPSupremumResult:
    """Certify an absolute supremum for a twice-differentiable scalar function."""
    positive = certified_global_maximum(
        value_fn,
        second_derivative_bound=second_derivative_bound,
        reference=reference,
        breakpoints=breakpoints,
    )
    negative = certified_global_maximum(
        lambda x: -float(value_fn(x)),
        second_derivative_bound=second_derivative_bound,
        reference=reference,
        breakpoints=breakpoints,
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


def certified_weighted_gp_supremum(
    draw: FourierGPDraw,
    uncertainty: SmoothClippedUncertaintySpec,
    center: float,
    reference: GlobalReferenceSpec,
) -> GPSupremumResult:
    """Certify sup_x |sigma_center(x) G(x)|."""
    sigma0 = uncertainty_derivative_bound(uncertainty, 0)
    sigma1 = uncertainty_derivative_bound(uncertainty, 1)
    sigma2 = uncertainty_derivative_bound(uncertainty, 2)
    second = (
        sigma2 * draw.derivative_bound(0)
        + 2.0 * sigma1 * draw.derivative_bound(1)
        + sigma0 * draw.derivative_bound(2)
    )
    radius = uncertainty.ramp_radius
    breakpoints = tuple(
        sorted(
            {
                float(np.clip(value, 0.0, 1.0))
                for value in (0.0, 1.0, center - radius, center, center + radius)
            }
        )
    )
    return certified_absolute_supremum(
        lambda x: float(_product_terms(draw, uncertainty, center, x, 0)),
        second_derivative_bound=second,
        reference=reference,
        breakpoints=breakpoints,
    )


def certified_shape_ratio(
    uncertainty: SmoothClippedUncertaintySpec,
    surrogate_center: float,
    envelope_center: float,
    reference: GlobalReferenceSpec,
) -> CertifiedMinimumResult:
    """Certify inf_x sigma_mE(x)/sigma_mf(x)."""
    if surrogate_center == envelope_center:
        return CertifiedMinimumResult(
            x=0.0,
            value=1.0,
            lower_bound=1.0,
            upper_bound=1.0,
            bound_gap=0.0,
            certified=True,
            intervals_created=0,
            evaluations=0,
        )
    sigma0 = uncertainty_derivative_bound(uncertainty, 0)
    sigma1 = uncertainty_derivative_bound(uncertainty, 1)
    sigma2 = uncertainty_derivative_bound(uncertainty, 2)
    minimum = uncertainty.minimum
    ratio_second_bound = (
        sigma2 / minimum
        + sigma0 * sigma2 / minimum**2
        + 2.0 * sigma1**2 / minimum**2
        + 2.0 * sigma0 * sigma1**2 / minimum**3
    )

    def ratio(x: float) -> float:
        return float(
            smooth_clipped_uncertainty(x, envelope_center, uncertainty)
            / smooth_clipped_uncertainty(x, surrogate_center, uncertainty)
        )

    radius = uncertainty.ramp_radius
    breakpoints = tuple(
        sorted(
            {
                float(np.clip(value, 0.0, 1.0))
                for center in (surrogate_center, envelope_center)
                for value in (0.0, 1.0, center - radius, center, center + radius)
            }
        )
    )
    negative = certified_global_maximum(
        lambda x: -ratio(x),
        second_derivative_bound=ratio_second_bound,
        reference=reference,
        breakpoints=breakpoints,
    )
    return CertifiedMinimumResult(
        x=negative.x,
        value=-negative.value,
        lower_bound=-negative.upper_bound,
        upper_bound=-negative.value,
        bound_gap=negative.bound_gap,
        certified=negative.certified,
        intervals_created=negative.intervals_created,
        evaluations=negative.evaluations,
    )


__all__ = [
    "AnalyticUniformCertificate",
    "CertifiedMinimumResult",
    "DecomposedGPLandscape",
    "FiniteFourierGPSpec",
    "FourierGPDraw",
    "GPSupremumResult",
    "GlobalMaximumResult",
    "GlobalReferenceSpec",
    "SmoothClippedUncertaintySpec",
    "UniformConfidenceSpec",
    "analytic_uniform_certificate",
    "certified_absolute_supremum",
    "certified_coverage_probability",
    "certified_global_maximum",
    "certified_gp_supremum",
    "certified_shape_ratio",
    "certified_weighted_gp_supremum",
    "smooth_clipped_uncertainty",
    "smoothstep",
    "true_regret",
    "true_value",
    "uncertainty_derivative_bound",
]
