"""One-dimensional strongly convex objective for zeroth-order proof checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from objective.base import Objective


@dataclass(frozen=True)
class ZerothOrderProofObjective(Objective):
    r"""The objective $$f(x)=x^2+\frac12(\sin x-x)$$ with $$x^\star=0$$."""

    mu: ClassVar[float] = 1.5
    smoothness: ClassVar[float] = 2.5
    third_derivative_bound: ClassVar[float] = 0.5

    def theta_dim(self, state_dim: int | None = None) -> int:
        del state_dim
        return 1

    def optimal_theta(self) -> np.ndarray:
        return np.zeros(1, dtype=float)

    def optimal_value(self) -> float:
        return 0.0

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        x = self._x(theta, x_batch)
        return float(x * x + 0.5 * (np.sin(x) - x))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        x = self._x(theta, x_batch)
        return np.asarray([2.0 * x + 0.5 * (np.cos(x) - 1.0)], dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "mu": self.mu,
            "smoothness": self.smoothness,
            "third_derivative_bound": self.third_derivative_bound,
            "x_star": 0.0,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZerothOrderProofObjective":
        if payload.get("type") != cls.__name__:
            raise ValueError(f"Expected objective type {cls.__name__!r}.")
        return cls()

    @staticmethod
    def _x(theta: np.ndarray, x_batch: Any) -> float:
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.shape != (1,) or not np.isfinite(theta_arr).all():
            raise ValueError("theta must be a finite one-dimensional vector with shape (1,).")
        if np.ndim(x_batch) != 2:
            raise ValueError("x_batch must be a 2D array.")
        return float(theta_arr[0])


__all__ = ["ZerothOrderProofObjective"]
