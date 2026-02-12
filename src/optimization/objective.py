"""Objective and oracle gradient API for pricing optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.models import AcceptanceProbability, Contract, Customer, ExpectedFinancialLoss, StateVector
from optimization.common import clip_u
from optimization.policy import phi


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    grad_u: float


def revenue_h(price: float, u: float) -> float:
    return float(price * u)


def objective(customer: Customer, u: float, price: float, rng: np.random.Generator) -> float:
    u_clipped = clip_u(u)
    contract = Contract(u=u_clipped)
    acceptance = AcceptanceProbability.sample(customer, contract, rng)
    expected_loss = ExpectedFinancialLoss.sample(customer, rng)
    revenue = revenue_h(price, contract.u)
    return float(acceptance.p * (expected_loss.value - revenue))


def objective_with_oracle_grad(
    customer: Customer,
    u: float,
    price: float,
    rng: np.random.Generator,
) -> ObjectiveResult:
    value = objective(customer, u, price, rng)

    # Oracle gradient API: treated as directly observable.
    # This is a placeholder; replace with true gradients if available.
    grad_u = float(rng.normal(loc=0.0, scale=1.0))
    return ObjectiveResult(value=value, grad_u=grad_u)


def fixed_regression_objective(x: StateVector, u: float, w: np.ndarray, c: float) -> float:
    if c == 0.0:
        raise ValueError("c must be nonzero for fixed regression objective.")
    features = phi(x)
    if w.size < features.size:
        raise ValueError("w must have at least phi(x) elements.")
    prediction = float(np.dot(w[: features.size], features))
    residual = prediction - c * u
    return float(residual**2)


def fixed_regression_objective_with_grad(
    x: StateVector, u: float, w: np.ndarray, c: float
) -> ObjectiveResult:
    value = fixed_regression_objective(x, u, w, c)
    features = phi(x)
    prediction = float(np.dot(w[: features.size], features))
    grad_u = float(-2.0 * c * (prediction - c * u))
    return ObjectiveResult(value=value, grad_u=grad_u)
