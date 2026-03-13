"""Theta-level objective composed from action objective and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from objective.base import ActionObjective, StateVector
from objective.policy import Policy


def _as_state_list(x_batch: np.ndarray | Sequence[StateVector]) -> list[StateVector]:
    if isinstance(x_batch, np.ndarray):
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        return [StateVector(values=row) for row in x_arr]
    x_list = list(x_batch)
    if not x_list:
        raise ValueError("x_batch must contain at least one sample.")
    return x_list


def _as_array(x_batch: np.ndarray | Sequence[StateVector]) -> np.ndarray:
    if isinstance(x_batch, np.ndarray):
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        return x_arr
    x_list = list(x_batch)
    if not x_list:
        raise ValueError("x_batch must contain at least one sample.")
    return np.stack([np.asarray(x, dtype=float) for x in x_list], axis=0).astype(float)


@dataclass(frozen=True)
class PolicyObjective:
    action_objective: ActionObjective
    policy: Policy

    def value(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> float:
        x_arr = _as_array(x_batch)
        u_batch = self.policy.action_batch(theta, x_arr)
        value_batch = getattr(self.action_objective, "value_batch", None)
        if callable(value_batch):
            values = np.asarray(value_batch(x_arr, u_batch), dtype=float)
            return float(np.mean(values))
        x_list = _as_state_list(x_arr)
        return float(
            np.mean(
                [self.action_objective.value(x, u) for x, u in zip(x_list, u_batch)],
            )
        )

    def grad(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> np.ndarray:
        x_arr = _as_array(x_batch)
        x_list = _as_state_list(x_arr)
        u_batch = self.policy.action_batch(theta, x_arr)
        grad_u_batch = getattr(self.action_objective, "grad_u_batch", None)
        if callable(grad_u_batch):
            grad_u_vals = np.asarray(grad_u_batch(x_arr, u_batch), dtype=float)
        else:
            grad_u_vals = np.asarray(
                [self.action_objective.grad_u(x, u) for x, u in zip(x_list, u_batch)],
                dtype=float,
            )

        theta_arr = np.asarray(theta, dtype=float)
        grad = np.zeros_like(theta_arr)
        for x, grad_u in zip(x_list, grad_u_vals):
            grad = grad + float(grad_u) * self.policy.grad_theta(theta_arr, x)
        return grad / float(len(x_list))

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> np.ndarray:
        return self.policy.action_batch(theta, _as_array(x_batch))

    def mean_action(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> float:
        return float(np.mean(self.action_batch(theta, x_batch)))

    def action_value(self, x: StateVector, u: float) -> float:
        return float(self.action_objective.value(x, u))

    def action_grad_u(self, x: StateVector, u: float) -> float:
        return float(self.action_objective.grad_u(x, u))

    def optimal_u(self) -> float | None:
        objective = self.action_objective
        optimal = getattr(objective, "optimal_u", None)
        if callable(optimal):
            return float(optimal())
        u_star = getattr(objective, "u_star", None)
        if u_star is not None:
            return float(u_star)
        return None


__all__ = ["PolicyObjective"]
