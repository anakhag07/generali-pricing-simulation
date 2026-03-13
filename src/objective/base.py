"""Core objective interfaces and state-vector container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


def default_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class StateVector:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("StateVector values must be a 1D array.")
        if values.size < 1:
            raise ValueError("StateVector must have at least one element.")
        object.__setattr__(self, "values", values)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    @staticmethod
    def sample(rng: np.random.Generator, dim: int) -> "StateVector":
        if dim <= 0:
            raise ValueError("StateVector dim must be positive.")
        return StateVector(values=rng.normal(0.0, 1.0, size=dim).astype(float))


class ActionObjective(Protocol):
    def value(self, x: StateVector, u: float) -> float:
        ...

    def grad_u(self, x: StateVector, u: float) -> float:
        ...

    def value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        ...


class ThetaObjective(Protocol):
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        ...

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        ...
