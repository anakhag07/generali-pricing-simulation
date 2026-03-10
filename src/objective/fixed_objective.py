"""Fixed regression objective components and implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from objective.base import ObjectiveResult, StateVector


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
class FixedRegressionAcceptance:
    beta_1: np.ndarray
    beta_2: float

    def __post_init__(self) -> None:
        beta_1 = np.asarray(self.beta_1, dtype=float)
        beta_2 = float(self.beta_2)
        if np.any(beta_1 <= 0.0):
            raise ValueError("beta_1 entries must be positive.")
        if beta_2 >= 0.0:
            raise ValueError(
                "beta_2 must be negative; acceptance probability should decrease as policy value increases."
            )
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "beta_2", beta_2)

    def logit(self, x: StateVector, u: float) -> float:
        return _beta_dot_x(self.beta_1, x) + self.beta_2 * u

    def probability(self, x: StateVector, u: float) -> float:
        return _logistic(self.logit(x, u))

    def grad_u(self, x: StateVector, u: float) -> float:
        acceptance = self.probability(x, u)
        return float(acceptance * (1.0 - acceptance) * self.beta_2)


@dataclass(frozen=True)
class FixedRegressionLoss:
    beta_3: np.ndarray

    def __post_init__(self) -> None:
        beta_3 = np.asarray(self.beta_3, dtype=float)
        if np.any(beta_3 <= 0.0):
            raise ValueError("beta_3 entries must be positive.")
        object.__setattr__(self, "beta_3", beta_3)

    def expected_loss(self, x: StateVector) -> float:
        return _beta_dot_x(self.beta_3, x)


@dataclass(frozen=True)
class FixedRegressionRevenue:
    beta_4: float

    def __post_init__(self) -> None:
        beta_4 = float(self.beta_4)
        if beta_4 <= 0.0:
            raise ValueError("beta_4 must be positive.")
        object.__setattr__(self, "beta_4", beta_4)

    def revenue(self, u: float) -> float:
        return self.beta_4 * u

    def grad_u(self, u: float) -> float:
        return self.beta_4


@dataclass(frozen=True)
class FixedRegressionObjective:
    acceptance: FixedRegressionAcceptance
    loss: FixedRegressionLoss
    revenue: FixedRegressionRevenue

    @classmethod
    def from_parameters(
        cls,
        beta_1: np.ndarray,
        beta_2: float,
        beta_3: np.ndarray,
        beta_4: float,
    ) -> "FixedRegressionObjective":
        acceptance = FixedRegressionAcceptance(beta_1=beta_1, beta_2=beta_2)
        loss = FixedRegressionLoss(beta_3=beta_3)
        revenue = FixedRegressionRevenue(beta_4=beta_4)
        return cls(acceptance=acceptance, loss=loss, revenue=revenue)

    def value(self, x: StateVector, u: float) -> float:
        acceptance = self.acceptance.probability(x, u)
        loss = self.loss.expected_loss(x)
        revenue_value = self.revenue.revenue(u)
        return float(acceptance * (loss - revenue_value))

    def grad_u(self, x: StateVector, u: float) -> float:
        acceptance = self.acceptance.probability(x, u)
        d_acceptance_du = self.acceptance.grad_u(x, u)
        loss = self.loss.expected_loss(x)
        revenue_value = self.revenue.revenue(u)
        return float(d_acceptance_du * (loss - revenue_value) - acceptance * self.revenue.grad_u(u))

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        value = self.value(x, u)
        grad_u = self.grad_u(x, u)
        return ObjectiveResult(value=value, grad_u=grad_u)

    def prepare_batch(self, x_array: np.ndarray) -> "FixedRegressionBatch":
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        beta_1 = self.acceptance.beta_1
        beta_3 = self.loss.beta_3
        beta_1_x = x_arr @ beta_1[: x_arr.shape[1]]
        beta_3_x = x_arr @ beta_3[: x_arr.shape[1]]
        return FixedRegressionBatch(
            beta_1_x=beta_1_x,
            beta_3_x=beta_3_x,
            beta_2=self.acceptance.beta_2,
            beta_4=self.revenue.beta_4,
        )

    def value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        batch = self.prepare_batch(x_array)
        return batch.value(u_array)

    def grad_u_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        batch = self.prepare_batch(x_array)
        return batch.grad_u(u_array)


@dataclass(frozen=True)
class FixedRegressionBatch:
    beta_1_x: np.ndarray
    beta_3_x: np.ndarray
    beta_2: float
    beta_4: float

    def value(self, u_array: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u_array, dtype=float)
        logits = self.beta_1_x + self.beta_2 * u_arr
        acceptance = _logistic_batch(logits)
        revenue = self.beta_4 * u_arr
        return acceptance * (self.beta_3_x - revenue)

    def grad_u(self, u_array: np.ndarray) -> np.ndarray:
        u_arr = np.asarray(u_array, dtype=float)
        logits = self.beta_1_x + self.beta_2 * u_arr
        acceptance = _logistic_batch(logits)
        d_acceptance_du = acceptance * (1.0 - acceptance) * self.beta_2
        revenue = self.beta_4 * u_arr
        return d_acceptance_du * (self.beta_3_x - revenue) - acceptance * self.beta_4
