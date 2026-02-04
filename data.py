"""Data classes and blackbox generators for pricing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


RNG = np.random.Generator


def default_rng(seed: Optional[int] = None) -> RNG:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class StateVector:
    """Customer state vector x in X with named features.

    Features: age, gender, geographic_location.
    """

    age: float
    gender: float
    geographic_location: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.age, self.gender, self.geographic_location], dtype=float)

    @staticmethod
    def sample(
        rng: RNG,
        age_range: Tuple[float, float] = (18.0, 90.0),
        gender_categories: int = 2,
        location_range: Tuple[float, float] = (0.0, 1.0),
    ) -> "StateVector":
        age = rng.uniform(*age_range)
        gender = float(rng.integers(0, gender_categories))
        geographic_location = rng.uniform(*location_range)
        return StateVector(age=age, gender=gender, geographic_location=geographic_location)


@dataclass(frozen=True)
class Customer:
    """Customer with state vector x in X."""

    x: StateVector
    customer_id: Optional[str] = None

    @staticmethod
    def sample(rng: RNG) -> "Customer":
        return Customer(x=StateVector.sample(rng=rng))


@dataclass(frozen=True)
class Contract:
    """Contract with action u in U = [0.5, 1.5]."""

    u: float

    def __post_init__(self) -> None:
        if not (0.5 <= self.u <= 1.5):
            raise ValueError("Contract u must be in [0.5, 1.5].")


@dataclass(frozen=True)
class AcceptanceProbability:
    """Z in {0, 1} with blackbox acceptance probability."""

    p: float
    z: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("Acceptance probability p must be in [0, 1].")
        if self.z not in (0, 1):
            raise ValueError("Acceptance indicator z must be 0 or 1.")

    @staticmethod
    def _blackbox_probability(x: np.ndarray, u: float, rng: RNG) -> float:
        weights = rng.normal(0.0, 0.5, size=3)
        bias = rng.normal(0.0, 0.2)
        x_arr = np.asarray(x, dtype=float)
        logit = float(np.dot(weights, x_arr) + bias + (u - 1.0))
        return 1.0 / (1.0 + np.exp(-logit))

    @classmethod
    def sample(cls, customer: Customer, contract: Contract, rng: RNG) -> "AcceptanceProbability":
        p = cls._blackbox_probability(customer.x.as_array(), contract.u, rng)
        z = int(rng.binomial(1, p))
        return cls(p=p, z=z)


@dataclass(frozen=True)
class ExpectedFinancialLoss:
    """Blackbox expected loss E[Y|x] for a customer."""

    value: float

    def __post_init__(self) -> None:
        if self.value < 0.0:
            raise ValueError("Expected financial loss must be nonnegative.")

    @staticmethod
    def _blackbox_expected_loss(x: np.ndarray, rng: RNG) -> float:
        x_arr = np.asarray(x, dtype=float)
        scale = 1000.0 + 50.0 * x_arr[0] + 200.0 * x_arr[2]
        noise = rng.lognormal(mean=0.0, sigma=0.6)
        return float(max(0.0, scale * noise))

    @classmethod
    def sample(cls, customer: Customer, rng: RNG) -> "ExpectedFinancialLoss":
        value = cls._blackbox_expected_loss(customer.x.as_array(), rng)
        return cls(value=value)


def example_usage(seed: int = 42) -> None:
    rng = default_rng(seed)
    customer = Customer.sample(rng)
    contract = Contract(u=1.1)
    acceptance = AcceptanceProbability.sample(customer, contract, rng)
    expected_loss = ExpectedFinancialLoss.sample(customer, rng)

    print("Customer x:", customer.x.as_array())
    print("Contract u:", contract.u)
    print("Acceptance p:", acceptance.p, "z:", acceptance.z)
    print("Expected loss:", expected_loss.value)


if __name__ == "__main__":
    example_usage()
