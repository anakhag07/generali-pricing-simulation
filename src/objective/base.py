"""Core objective interfaces and state-vector container."""

from __future__ import annotations

from typing import Optional

import numpy as np


def default_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


class StateVector:
    def __init__(self, values: np.ndarray) -> None:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim != 1:
            raise ValueError("StateVector values must be a 1D array.")
        if values_arr.size < 1:
            raise ValueError("StateVector must have at least one element.")
        self.values = values_arr

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        if dtype is None:
            return self.values
        return self.values.astype(dtype, copy=False)

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return f"StateVector(values={self.values!r})"

    @staticmethod
    def sample(rng: np.random.Generator, dim: int) -> "StateVector":
        if dim <= 0:
            raise ValueError("StateVector dim must be positive.")
        return StateVector(values=rng.normal(0.0, 1.0, size=dim).astype(float))


class ActionObjective:
    def value(self, x: StateVector, u: float) -> float:
        raise NotImplementedError

    def grad_u(self, x: StateVector, u: float) -> float:
        raise NotImplementedError


class ThetaObjective:
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError
