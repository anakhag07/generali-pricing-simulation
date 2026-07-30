"""Pure helpers for one-dimensional zeroth-order landscape analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import brentq


ScalarFn = Callable[[float], float]


@lru_cache(maxsize=None)
def _hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


@lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


@dataclass(frozen=True)
class StationaryPoint:
    """A classified stationary point of a scalar objective."""

    x: float
    kind: str
    value: float


def finite_difference_population_gradient(
    value_fn: ScalarFn,
    x: float,
    sigma: float,
) -> float:
    """Return the population central finite-difference gradient."""
    sigma_float = float(sigma)
    if not np.isfinite(sigma_float) or sigma_float <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    return float(
        (value_fn(float(x) + sigma_float) - value_fn(float(x) - sigma_float))
        / (2.0 * sigma_float)
    )


def stein_population_gradient(
    grad_fn: ScalarFn,
    x: float,
    sigma: float,
    *,
    quadrature_order: int = 80,
) -> float:
    """Return the population two-sided Gaussian Stein-difference gradient."""
    sigma_float = float(sigma)
    if not np.isfinite(sigma_float) or sigma_float <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if int(quadrature_order) < 2:
        raise ValueError("quadrature_order must be at least 2.")
    normal_nodes, normal_weights = _hermite_rule(int(quadrature_order))
    return float(
        np.sum(
            normal_weights
            * np.asarray(
                [grad_fn(float(x) + sigma_float * float(node)) for node in normal_nodes],
                dtype=float,
            )
        )
    )


def estimator_population_gradient(
    estimator: str,
    value_fn: ScalarFn,
    grad_fn: ScalarFn,
    x: float,
    sigma: float,
) -> float:
    """Return the population gradient for a supported zeroth-order estimator."""
    if estimator == "finite_difference":
        return finite_difference_population_gradient(value_fn, x, sigma)
    if estimator == "stein_difference":
        return stein_population_gradient(grad_fn, x, sigma)
    raise ValueError(f"Unsupported estimator {estimator!r}.")


def estimator_population_value(
    estimator: str,
    value_fn: ScalarFn,
    x: float,
    sigma: float,
    *,
    quadrature_order: int = 80,
) -> float:
    """Return an antiderivative-compatible population-smoothed value."""
    sigma_float = float(sigma)
    if not np.isfinite(sigma_float) or sigma_float <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if int(quadrature_order) < 2:
        raise ValueError("quadrature_order must be at least 2.")
    if estimator == "finite_difference":
        nodes, weights = _legendre_rule(int(quadrature_order))
        values = np.asarray(
            [value_fn(float(x) + sigma_float * float(node)) for node in nodes],
            dtype=float,
        )
        return float(0.5 * np.sum(weights * values))
    if estimator == "stein_difference":
        normal_nodes, normal_weights = _hermite_rule(int(quadrature_order))
        values = np.asarray(
            [
                value_fn(float(x) + sigma_float * float(node))
                for node in normal_nodes
            ],
            dtype=float,
        )
        return float(np.sum(normal_weights * values))
    raise ValueError(f"Unsupported estimator {estimator!r}.")


def find_stationary_points(
    value_fn: ScalarFn,
    grad_fn: ScalarFn,
    *,
    domain: tuple[float, float],
    kinks: Iterable[float] = (),
    grid_size: int = 12001,
) -> list[StationaryPoint]:
    """Find and classify scalar stationary points on an enclosing domain."""
    low, high = (float(value) for value in domain)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("domain must be a finite increasing pair.")
    if int(grid_size) < 101:
        raise ValueError("grid_size must be at least 101.")

    grid = np.linspace(low, high, int(grid_size))
    gradients = np.asarray([grad_fn(float(x)) for x in grid], dtype=float)
    if not np.all(np.isfinite(gradients)):
        raise ValueError("gradient must be finite across the search domain.")

    roots: list[float] = []
    zero_indices = np.flatnonzero(gradients == 0.0)
    roots.extend(float(grid[index]) for index in zero_indices)
    crossings = np.flatnonzero(gradients[:-1] * gradients[1:] < 0.0)
    for index in crossings:
        left = float(grid[index])
        right = float(grid[index + 1])
        roots.append(
            float(brentq(grad_fn, left, right, xtol=1e-13, rtol=1e-13))
        )

    spacing = float((high - low) / (int(grid_size) - 1))
    probe = max(1e-7, spacing * 0.25)
    for kink in kinks:
        kink_float = float(kink)
        if not low < kink_float < high:
            continue
        left_grad = float(grad_fn(kink_float - probe))
        right_grad = float(grad_fn(kink_float + probe))
        if left_grad <= 0.0 <= right_grad or left_grad >= 0.0 >= right_grad:
            roots.append(kink_float)

    points: list[StationaryPoint] = []
    for root in _deduplicate(roots, tolerance=max(1e-9, spacing * 0.1)):
        left_grad = float(grad_fn(max(low, root - probe)))
        right_grad = float(grad_fn(min(high, root + probe)))
        if left_grad < 0.0 < right_grad:
            kind = "minimum"
        elif left_grad > 0.0 > right_grad:
            kind = "maximum"
        else:
            kind = "stationary"
        points.append(
            StationaryPoint(x=float(root), kind=kind, value=float(value_fn(float(root))))
        )

    if float(grad_fn(low)) >= 0.0 or float(grad_fn(high)) <= 0.0:
        raise ValueError("domain does not enclose the scalar landscape.")
    return sorted(points, key=lambda point: point.x)


def global_minimum(points: Iterable[StationaryPoint]) -> StationaryPoint:
    """Return the lowest classified local minimum."""
    minima = [point for point in points if point.kind == "minimum"]
    if not minima:
        raise ValueError("No classified local minimum was found.")
    return min(minima, key=lambda point: point.value)


def _deduplicate(values: Iterable[float], *, tolerance: float) -> list[float]:
    ordered = sorted(float(value) for value in values)
    output: list[float] = []
    for value in ordered:
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
    return output


__all__ = [
    "StationaryPoint",
    "estimator_population_gradient",
    "estimator_population_value",
    "find_stationary_points",
    "finite_difference_population_gradient",
    "global_minimum",
    "stein_population_gradient",
]
