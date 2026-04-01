"""CSV-backed u-space objective using pre-computed model predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from objective.base import Objective, Policy
from objective.policy import SoftmaxPolicy


@dataclass(frozen=True)
class CSVObjective(Objective):
    """U-space objective from pre-computed CSV predictions.

    Evaluates $$f(u) = \\text{mean}(a \\cdot (\\hat{Y} - u \\cdot p))$$ where
    $$a$$ (``prob_acceptance``), $$\\hat{Y}$$ (``Y_hat``), and $$p$$ (``X_policy_premium``)
    are taken from rows of the CSV near the query u.

    ``value_at_u(u)``: filters rows where ``|U - u| < tol``; falls back to the
    ``k_fallback`` nearest rows if fewer than ``k_fallback`` rows are found.

    ``value(theta, x_batch)``: computes u as the mean policy action over x_batch,
    then delegates to ``value_at_u``.

    ``grad()``: raises ``NotImplementedError`` — use FD/SPSA/Gauss-Stein estimators.
    """

    _df: pd.DataFrame            # must contain: U, prob_acceptance, Y_hat, X_policy_premium
    policy: Policy = field(default_factory=SoftmaxPolicy)
    tol: float = 0.005
    _k_fallback: int = 100

    def value_at_u(self, u: float, k_fallback: int | None = None) -> float:
        """Mean objective at query u using pre-computed CSV predictions.

        Filters rows where ``|U - u| < tol``. If fewer than ``k_fallback`` rows are
        found, uses the k nearest rows by absolute distance instead.
        """
        k = k_fallback if k_fallback is not None else self._k_fallback
        u_col = self._df["U"].to_numpy(dtype=float)
        mask = np.abs(u_col - u) < self.tol
        if mask.sum() < k:
            # Fallback: k nearest rows
            distances = np.abs(u_col - u)
            idx = np.argpartition(distances, min(k, len(distances) - 1))[:k]
            mask = np.zeros(len(u_col), dtype=bool)
            mask[idx] = True

        rows = self._df[mask]
        prob_acc = rows["prob_acceptance"].to_numpy(dtype=float)
        y_hat = rows["Y_hat"].to_numpy(dtype=float)
        premium = rows["X_policy_premium"].to_numpy(dtype=float)
        revenue = float(u) * premium
        values = prob_acc * (y_hat - revenue)
        return float(np.mean(values))

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute objective value by evaluating policy on x_batch to get mean u."""
        theta_arr = np.asarray(theta, dtype=float)
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        u_batch = self.policy.value(theta_arr, x_arr)
        u_scalar = float(np.mean(u_batch))
        return self.value_at_u(u_scalar)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Not implemented: use FD/SPSA/Gauss-Stein estimators instead."""
        raise NotImplementedError(
            "CSVObjective does not support analytical gradients. "
            "Use finite_difference, spsa, or gauss_stein estimators."
        )


__all__ = ["CSVObjective"]
