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


def _logistic_batch(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


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

    def prepare_batch(self, x_array: np.ndarray) -> "PlantedLogisticBatch":
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        beta = self.beta
        beta_x = x_arr @ beta[: x_arr.shape[1]]
        z_star = self.alpha * self.u_star + beta_x + self.bias
        p_star = _logistic_batch(z_star)
        return PlantedLogisticBatch(
            alpha=self.alpha,
            beta_x=beta_x,
            bias=self.bias,
            p_star=p_star,
        )

    def value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        batch = self.prepare_batch(x_array)
        return batch.value(u_array)

    def grad_u_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        batch = self.prepare_batch(x_array)
        return batch.grad_u(u_array)


@dataclass(frozen=True)
class PlantedLogisticBatch:
    alpha: float
    beta_x: np.ndarray
    bias: float
    p_star: np.ndarray

    def value(self, u_array: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u_array, dtype=float)
        z = self.alpha * u_arr + self.beta_x + self.bias
        return np.logaddexp(0.0, z) - self.p_star * z

    def grad_u(self, u_array: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u_array, dtype=float)
        z = self.alpha * u_arr + self.beta_x + self.bias
        p = _logistic_batch(z)
        return self.alpha * (p - self.p_star)
