"""Private math utilities for objective computation."""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arrays."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out
