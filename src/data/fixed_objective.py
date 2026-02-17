"""Fixed regression objective components and implementation."""

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
