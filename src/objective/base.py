"""Core objective interfaces and state-vector container."""

from __future__ import annotations

from typing import Optional

import numpy as np


def default_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy random generator, optionally seeded."""
    return np.random.default_rng(seed)


class StateVector:
    """Customer feature vector $$x \\in \\mathbb{R}^d$$ sampled from $$\\mathcal{N}(0, I)$$."""

    def __init__(self, values: np.ndarray) -> None:
        """Create a validated 1D float feature vector."""
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim != 1:
            raise ValueError("StateVector values must be a 1D array.")
        if values_arr.size < 1:
            raise ValueError("StateVector must have at least one element.")
        self.values = values_arr

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Expose ``values`` for ``np.asarray`` interoperability."""
        if dtype is None:
            return self.values
        return self.values.astype(dtype, copy=False)

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return f"StateVector(values={self.values!r})"

    @staticmethod
    def sample(rng: np.random.Generator, dim: int) -> "StateVector":
        """Sample a standard-normal state vector of dimension ``dim``."""
        if dim <= 0:
            raise ValueError("StateVector dim must be positive.")
        return StateVector(values=rng.normal(0.0, 1.0, size=dim).astype(float))


class Policy:
    """Policy interface: $$u = \\pi_\\theta(x)$$ with gradient $$\\partial u / \\partial \\theta$$."""

    def value(self, theta: np.ndarray, x: StateVector) -> float:
        """Return action value ``u`` for ``(theta, x)``."""
        raise NotImplementedError

    def grad(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        """Return policy gradient with respect to ``theta``."""
        raise NotImplementedError


class Objective:
    """Theta-space objective: $$J(\\theta) = \\mathbb{E}_x[f(\\pi_\\theta(x); x)]$$."""

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return mean objective value for ``theta`` on ``x_batch``."""
        raise NotImplementedError

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return theta-gradient for ``theta`` on ``x_batch``."""
        raise NotImplementedError
