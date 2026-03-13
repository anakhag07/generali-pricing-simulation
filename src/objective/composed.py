"""Theta-level objective composed from action objective and policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objective.base import Policy, StateVector
from objective.policy import policy_u_batch


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


def _action_values(
    action_objective: object,
    x_arr: np.ndarray,
    x_list: Sequence[StateVector],
    u_batch: np.ndarray,
) -> np.ndarray:
    objective = action_objective
    value_batch = getattr(objective, "value_batch", None)
    if callable(value_batch):
        return np.asarray(value_batch(x_arr, u_batch), dtype=float)
    value_fn = getattr(objective, "value", None)
    if not callable(value_fn):
        raise ValueError("action_objective must implement value(x, u).")
    return np.asarray([value_fn(x, u) for x, u in zip(x_list, u_batch)], dtype=float)


def _action_grad_u_values(
    action_objective: object,
    x_arr: np.ndarray,
    x_list: Sequence[StateVector],
    u_batch: np.ndarray,
) -> np.ndarray:
    objective = action_objective
    grad_u_batch = getattr(objective, "grad_u_batch", None)
    if callable(grad_u_batch):
        return np.asarray(grad_u_batch(x_arr, u_batch), dtype=float)
    grad_u_fn = getattr(objective, "grad_u", None)
    if not callable(grad_u_fn):
        raise ValueError("action_objective must implement grad_u(x, u).")
    return np.asarray([grad_u_fn(x, u) for x, u in zip(x_list, u_batch)], dtype=float)


def _policy_values(
    policy: Policy,
    theta: np.ndarray,
    x_arr: np.ndarray,
    x_list: Sequence[StateVector],
) -> np.ndarray:
    policy_kind = getattr(policy, "kind", None)
    if isinstance(policy_kind, str):
        return policy_u_batch(theta, x_arr, kind=policy_kind)
    return np.asarray([policy.value(theta, x) for x in x_list], dtype=float)


@dataclass(frozen=True)
class PolicyObjective:
    action_objective: Any
    policy: Policy

    def value(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> float:
        x_arr = _as_array(x_batch)
        x_list = _as_state_list(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = _policy_values(self.policy, theta_arr, x_arr, x_list)
        values = _action_values(self.action_objective, x_arr, x_list, u_batch)
        return float(np.mean(values))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> np.ndarray:
        x_arr = _as_array(x_batch)
        x_list = _as_state_list(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = _policy_values(self.policy, theta_arr, x_arr, x_list)
        grad_u_vals = _action_grad_u_values(self.action_objective, x_arr, x_list, u_batch)

        grad = np.zeros_like(theta_arr)
        for x, grad_u in zip(x_list, grad_u_vals):
            grad = grad + float(grad_u) * self.policy.grad(theta_arr, x)
        return grad / float(len(x_list))

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> np.ndarray:
        x_arr = _as_array(x_batch)
        x_list = _as_state_list(x_arr)
        return _policy_values(self.policy, np.asarray(theta, dtype=float), x_arr, x_list)

    def mean_action(self, theta: np.ndarray, x_batch: np.ndarray | Sequence[StateVector]) -> float:
        return float(np.mean(self.action_batch(theta, x_batch)))

    def action_value(self, x: StateVector, u: float) -> float:
        value_fn = getattr(self.action_objective, "value", None)
        if not callable(value_fn):
            raise ValueError("action_objective must implement value(x, u).")
        return float(value_fn(x, u))

    def action_grad_u(self, x: StateVector, u: float) -> float:
        grad_u_fn = getattr(self.action_objective, "grad_u", None)
        if not callable(grad_u_fn):
            raise ValueError("action_objective must implement grad_u(x, u).")
        return float(grad_u_fn(x, u))

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
