"""Core objective interfaces and sampling utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np


def default_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy random generator, optionally seeded."""
    return np.random.default_rng(seed)


def sample_states(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Sample n state vectors from $$\\mathcal{N}(0, I)$$, shape (n, dim)."""
    if n <= 0 or dim <= 0:
        raise ValueError("n and dim must be positive.")
    return rng.normal(0.0, 1.0, size=(n, dim)).astype(float)


class Policy:
    """Policy interface: $$u = \\pi_\\theta(x)$$ with gradient $$\\partial u / \\partial \\theta$$."""

    def theta_dim(self, state_dim: int) -> int:
        """Return required theta dimension for inputs with ``state_dim`` columns."""
        raise NotImplementedError

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return action values for batch, shape (n_samples,)."""
        raise NotImplementedError

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return policy gradients for batch, shape (n_samples, theta_dim)."""
        raise NotImplementedError

    def weighted_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Return ``sum_i weights_i * d pi_theta(x_i) / d theta``."""
        policy_grad = self.grad(theta, x_batch)
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if weights_arr.shape != (policy_grad.shape[0],):
            raise ValueError("weights must have one value per x_batch row.")
        return np.einsum("n,nd->d", weights_arr, policy_grad)


class Objective:
    """Theta-space objective: $$J(\\theta) = \\mathbb{E}_x[f(\\pi_\\theta(x); x)]$$."""

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return mean objective value for ``theta`` on ``x_batch``."""
        raise NotImplementedError

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return theta-gradient for ``theta`` on ``x_batch``."""
        raise NotImplementedError
