"""Validated interpolation for customer-specific action-response grids."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from objective.base import Objective
from objective.policy import SoftmaxPolicy


def interpolate_customer_curves(
    values: np.ndarray,
    grid: Sequence[float],
    actions: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate one curve per customer and return local slopes."""
    curves = np.asarray(values, dtype=float)
    action_grid = np.asarray(grid, dtype=float)
    proposed = np.asarray(actions, dtype=float).reshape(-1)
    if action_grid.ndim != 1 or action_grid.size < 2:
        raise ValueError("grid must contain at least two actions.")
    if not np.isfinite(action_grid).all() or not np.all(np.diff(action_grid) > 0.0):
        raise ValueError("grid must be finite and strictly increasing.")
    if not np.isfinite(proposed).all():
        raise ValueError("actions must be finite.")
    if curves.shape != (proposed.size, action_grid.size):
        raise ValueError("values must contain one action-grid curve per customer.")
    if not np.isfinite(curves).all():
        raise ValueError("values must be finite.")

    clipped = np.clip(proposed, action_grid[0], action_grid[-1])
    right = np.searchsorted(action_grid, clipped, side="right")
    left = np.clip(right - 1, 0, action_grid.size - 2)
    right = left + 1
    row = np.arange(proposed.size)
    span = action_grid[right] - action_grid[left]
    fraction = (clipped - action_grid[left]) / span
    low = curves[row, left]
    high = curves[row, right]
    return low + fraction * (high - low), (high - low) / span


class SplineSupportLowerBoundObjective(Objective):
    """Customer-grid cost plus a smooth, action-only support penalty."""

    def __init__(
        self,
        *,
        policy: SoftmaxPolicy,
        cost_grid: np.ndarray,
        acceptance_grid: np.ndarray,
        action_grid: np.ndarray,
        support_width: np.ndarray,
        acceptance_floor: float,
    ) -> None:
        self.policy = policy
        self.cost_grid = np.asarray(cost_grid, dtype=float)
        self.acceptance_grid = np.asarray(acceptance_grid, dtype=float)
        self.action_grid = np.asarray(action_grid, dtype=float)
        self.support_width = np.asarray(support_width, dtype=float)
        self.acceptance_floor = float(acceptance_floor)
        if self.cost_grid.shape != self.acceptance_grid.shape:
            raise ValueError("cost and acceptance grids must have matching shapes.")
        if self.cost_grid.shape[1:] != (self.action_grid.size,):
            raise ValueError("response grids must align to action_grid.")
        if self.support_width.shape != self.action_grid.shape:
            raise ValueError("support_width must align to action_grid.")
        if not np.isfinite(self.cost_grid).all() or not np.isfinite(
            self.acceptance_grid
        ).all():
            raise ValueError("response grids must be finite.")
        if np.any(self.support_width <= 0.0):
            raise ValueError("absolute support width must be positive everywhere.")
        self._support_spline = CubicSpline(
            self.action_grid,
            self.support_width,
            bc_type="natural",
            extrapolate=False,
        )

    def _actions(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        if len(x_batch) != self.cost_grid.shape[0]:
            raise ValueError("This deterministic objective requires its full fixed batch.")
        return np.asarray(self.policy.value(theta, x_batch), dtype=float)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self._actions(theta, x_batch)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions = self._actions(theta, x_batch)
        cost, _ = interpolate_customer_curves(self.cost_grid, self.action_grid, actions)
        width = np.asarray(self._support_spline(actions), dtype=float)
        return float(np.mean(cost + width))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        actions = self._actions(theta, x_batch)
        _, cost_slope = interpolate_customer_curves(
            self.cost_grid, self.action_grid, actions
        )
        width_slope = np.asarray(self._support_spline(actions, 1), dtype=float)
        return self.policy.weighted_grad(
            theta, x_batch, cost_slope + width_slope
        ) / len(actions)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        actions = self._actions(theta, x_batch)
        acceptance, _ = interpolate_customer_curves(
            self.acceptance_grid, self.action_grid, actions
        )
        return float(np.mean(acceptance))

    def mean_acceptance_grad(
        self, theta: np.ndarray, x_batch: np.ndarray
    ) -> np.ndarray:
        actions = self._actions(theta, x_batch)
        _, slope = interpolate_customer_curves(
            self.acceptance_grid, self.action_grid, actions
        )
        return self.policy.weighted_grad(theta, x_batch, slope) / len(actions)

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, float]:
        actions = self._actions(theta, x_batch)
        cost, _ = interpolate_customer_curves(self.cost_grid, self.action_grid, actions)
        width = np.asarray(self._support_spline(actions), dtype=float)
        return {
            "mean_acceptance": self.mean_acceptance(theta, x_batch),
            "projected_loss": float("nan"),
            "projected_revenue": float("nan"),
            "raw_mean_profit": float(-np.mean(cost)),
            "mean_support_width": float(np.mean(width)),
        }

    def summarize(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, Any]:
        actions = self._actions(theta, x_batch)
        cost, _ = interpolate_customer_curves(self.cost_grid, self.action_grid, actions)
        acceptance, _ = interpolate_customer_curves(
            self.acceptance_grid, self.action_grid, actions
        )
        width = np.asarray(self._support_spline(actions), dtype=float)
        quantiles = np.quantile(actions, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            "mean_action": float(np.mean(actions)),
            "population_std_action": float(np.std(actions, ddof=0)),
            "action_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                    quantiles,
                    strict=True,
                )
            },
            "mean_acceptance": float(np.mean(acceptance)),
            "raw_mean_profit": float(-np.mean(cost)),
            "mean_support_half_width": float(np.mean(width)),
            "support_lower_bound_mean_profit": float(np.mean(-cost - width)),
        }


__all__ = ["SplineSupportLowerBoundObjective", "interpolate_customer_curves"]
