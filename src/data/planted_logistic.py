"""Planted convex logistic objective with known optimum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.models import ObjectiveResult, StateVector


def _logistic(z: float) -> float:
    if z >= 0.0:
        exp_neg = float(np.exp(-z))
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = float(np.exp(z))
    return float(exp_pos / (1.0 + exp_pos))


def _beta_dot_x(beta: np.ndarray, x: StateVector) -> float:
    features = x.as_array().astype(float)
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.size < features.size:
        raise ValueError("beta must have at least as many elements as x.")
    return float(np.dot(beta_arr[: features.size], features))


@dataclass(frozen=True)
class PlantedLogisticObjective:
    """Convex logistic objective with a planted minimum at u_star."""

    alpha: float
    beta: np.ndarray
    bias: float
    u_star: float

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        beta = np.asarray(self.beta, dtype=float)
        bias = float(self.bias)
        u_star = float(self.u_star)
        if alpha == 0.0:
            raise ValueError("alpha must be nonzero for a unique optimum.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "u_star", u_star)

    def optimal_u(self) -> float:
        return float(self.u_star)

    def _logit(self, x: StateVector, u: float) -> float:
        return float(self.alpha * u + _beta_dot_x(self.beta, x) + self.bias)

    def _p_star(self, x: StateVector) -> float:
        return _logistic(self._logit(x, self.u_star))

    def value(self, x: StateVector, u: float) -> float:
        z = self._logit(x, u)
        p_star = self._p_star(x)
        return float(np.logaddexp(0.0, z) - p_star * z)

    def grad_u(self, x: StateVector, u: float) -> float:
        z = self._logit(x, u)
        p = _logistic(z)
        p_star = self._p_star(x)
        return float(self.alpha * (p - p_star))

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        value = self.value(x, u)
        grad_u = self.grad_u(x, u)
        return ObjectiveResult(value=value, grad_u=grad_u)
