"""Established finite-Fourier synthetic representation, independent of study orchestration."""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.stats import norm

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
