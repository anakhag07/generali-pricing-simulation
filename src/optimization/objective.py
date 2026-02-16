"""Objective and oracle gradient API for pricing optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.models import AcceptanceProbability, Contract, Customer, ExpectedFinancialLoss, StateVector


@dataclass(frozen=True)
class ObjectiveResult:
    value: float
    grad_u: float


def revenue_h(price: float, u: float) -> float:
    return float(price * u)


def objective(customer: Customer, u: float, price: float, rng: np.random.Generator) -> float:
    # u_clipped = clip_u(u)
    u_clipped = u
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


def _logistic(z: float) -> float:
    if z >= 0.0:
        exp_neg = float(np.exp(-z))
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = float(np.exp(z))
    return float(exp_pos / (1.0 + exp_pos))


def _beta_dot_x(beta: np.ndarray, x: StateVector) -> float:
    features = x.as_array().astype(float)
    if beta.size < features.size:
        raise ValueError("beta must have at least as many elements as x.")
    return float(np.dot(beta[: features.size], features))


def fixed_regression_objective(
    x: StateVector,
    u: float,
    beta_1: np.ndarray,
    beta_2: float,
    beta_3: np.ndarray,
    beta_4: float,
) -> float:
    logit = _beta_dot_x(beta_1, x) + float(beta_2) * u
    acceptance = _logistic(logit)
    loss = _beta_dot_x(beta_3, x)
    revenue = float(beta_4) * u
    return float(acceptance * (loss - revenue))


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
    revenue = float(beta_4) * u
    d_acceptance_du = float(acceptance * (1.0 - acceptance) * float(beta_2))
    grad_u = d_acceptance_du * (loss - revenue) - acceptance * float(beta_4)
    return ObjectiveResult(value=value, grad_u=grad_u)
