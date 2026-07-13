"""Strongly convex quadratic objective in parameter space."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from objective.base import Objective


@dataclass(frozen=True)
class QuadraticObjective(Objective):
    """Direct theta-space objective $$J(\\theta)=\\frac{1}{2}\\|\\theta\\|_2^2$$."""

    dimension: int

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, Integral):
            raise TypeError("dimension must be an integer.")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        object.__setattr__(self, "dimension", int(self.dimension))

    def theta_dim(self, state_dim: int | None = None) -> int:
        """Return the required parameter dimension; state dimension is irrelevant."""
        del state_dim
        return self.dimension

    def optimal_theta(self) -> np.ndarray:
        """Return the unique minimizer $$\\theta^*=0$$."""
        return np.zeros(self.dimension, dtype=float)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return $$\\frac{1}{2}\\|\\theta\\|_2^2$$; ``x_batch`` is intentionally ignored."""
        theta_arr = self._validate_inputs(theta, x_batch)
        return 0.5 * float(np.dot(theta_arr, theta_arr))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return the exact theta-gradient $$\\nabla J(\\theta)=\\theta$$."""
        return self._validate_inputs(theta, x_batch).copy()

    def _validate_inputs(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.ndim != 1 or theta_arr.size != self.dimension:
            raise ValueError(f"theta must be a 1D array with dimension {self.dimension}.")
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("theta must contain only finite values.")
        if np.ndim(x_batch) != 2:
            raise ValueError("x_batch must be a 2D array.")
        return theta_arr


__all__ = ["QuadraticObjective"]
