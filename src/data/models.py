"""Data classes and objective components for pricing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


RNG = np.random.Generator


def default_rng(seed: Optional[int] = None) -> RNG:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class StateVector:
    """Customer state vector x in X."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("StateVector values must be a 1D array.")
        if values.size < 1:
            raise ValueError("StateVector must have at least one element.")
        object.__setattr__(self, "values", values)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    @staticmethod
    def sample(
        rng: RNG,
        dim: int = 3,
        age_range: Tuple[float, float] = (18.0, 90.0),
        gender_categories: int = 2,
        location_range: Tuple[float, float] = (0.0, 1.0),
    ) -> "StateVector":
        if dim <= 0:
            raise ValueError("StateVector dim must be positive.")
        if dim == 3:
            age = rng.uniform(*age_range)
            gender = float(rng.integers(0, gender_categories))
            geographic_location = rng.uniform(*location_range)
            values = np.asarray([age, gender, geographic_location], dtype=float)
        else:
            values = rng.uniform(0.0, 1.0, size=dim).astype(float)
        return StateVector(values=values)


@dataclass(frozen=True)
class Customer:
    """Customer with state vector x in X."""

    x: StateVector
    customer_id: Optional[str] = None

    @staticmethod
    def sample(rng: RNG, state_dim: int = 3) -> "Customer":
        return Customer(x=StateVector.sample(rng=rng, dim=state_dim))


@dataclass(frozen=True)
class Contract:
    """Contract with action u in U = [0.5, 1.5]."""

    u: float

    def __post_init__(self) -> None:
        if not (0.5 <= self.u <= 1.5):
            pass


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    grad_u: float


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


def acceptance_probability(x: StateVector, u: float, beta_1: np.ndarray, beta_2: float) -> float:
    logit = _beta_dot_x(beta_1, x) + float(beta_2) * u
    return _logistic(logit)


def expected_loss(x: StateVector, beta_3: np.ndarray) -> float:
    return _beta_dot_x(beta_3, x)


def revenue(u: float, beta_4: float) -> float:
    return float(beta_4) * u


def fixed_regression_objective(
    x: StateVector,
    u: float,
    beta_1: np.ndarray,
    beta_2: float,
    beta_3: np.ndarray,
    beta_4: float,
) -> float:
    acceptance = acceptance_probability(x, u, beta_1, beta_2)
    loss = expected_loss(x, beta_3)
    revenue_value = revenue(u, beta_4)
    return float(acceptance * (loss - revenue_value))


def fixed_regression_objective_with_grad(
    x: StateVector,
    u: float,
    beta_1: np.ndarray,
    beta_2: float,
    beta_3: np.ndarray,
    beta_4: float,
) -> ObjectiveResult:
    value = fixed_regression_objective(x, u, beta_1, beta_2, beta_3, beta_4)
    logit = _beta_dot_x(beta_1, x) + float(beta_2) * u
    acceptance = _logistic(logit)
    loss = _beta_dot_x(beta_3, x)
    revenue_value = revenue(u, beta_4)
    d_acceptance_du = float(acceptance * (1.0 - acceptance) * float(beta_2))
    grad_u = d_acceptance_du * (loss - revenue_value) - acceptance * float(beta_4)
    return ObjectiveResult(value=value, grad_u=grad_u)
