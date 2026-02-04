"""Common helpers for optimization routines."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

U_BOUNDS: Tuple[float, float] = (0.5, 1.5)


def clip_u(u: float) -> float:
    return float(np.clip(u, U_BOUNDS[0], U_BOUNDS[1]))


def gaussian_noise(rng: np.random.Generator, shape: Iterable[int] = ()) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=shape)


def constant_step(step_size: float) -> float:
    return float(step_size)
