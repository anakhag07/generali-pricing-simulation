"""Core objective-related dataclasses and protocols for pricing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

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
        dim: int,
    ) -> "StateVector":
        if dim <= 0:
            raise ValueError("StateVector dim must be positive.")
        values = rng.normal(0.0, 1.0, size=dim).astype(float)
        return StateVector(values=values)


@dataclass(frozen=True)
class Customer:
    """Customer with state vector x in X."""

    x: StateVector
    customer_id: Optional[str] = None

    @staticmethod
    def sample(rng: RNG, state_dim: int) -> "Customer":
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


class AcceptanceModel(Protocol):
    def probability(self, x: "StateVector", u: float) -> float:
        ...

    def grad_u(self, x: "StateVector", u: float) -> float:
        ...


class LossModel(Protocol):
    def expected_loss(self, x: "StateVector") -> float:
        ...


class RevenueModel(Protocol):
    def revenue(self, u: float) -> float:
        ...

    def grad_u(self, u: float) -> float:
        ...


class ObjectiveModel(Protocol):
    def value(self, x: "StateVector", u: float) -> float:
        ...

    def grad_u(self, x: "StateVector", u: float) -> float:
        ...

    def evaluate(self, x: "StateVector", u: float) -> ObjectiveResult:
        ...


class ActionObjective(Protocol):
    def value(self, x: "StateVector", u: float) -> float:
        ...

    def grad_u(self, x: "StateVector", u: float) -> float:
        ...

    def value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        ...


class ThetaObjective(Protocol):
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        ...

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        ...
