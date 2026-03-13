from __future__ import annotations

import math

from objective.fixed_objective import (
    FixedRegressionAcceptance,
    FixedRegressionLoss,
    FixedRegressionObjective,
    FixedRegressionRevenue,
)
from objective.base import StateVector


def test_fixed_regression_objective_matches_components() -> None:
    x = StateVector(values=[1.0, 2.0])
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    acceptance = FixedRegressionAcceptance(beta_1=beta_1, beta_2=beta_2)
    loss = FixedRegressionLoss(beta_3=beta_3)
    revenue = FixedRegressionRevenue(beta_4=beta_4)
    objective = FixedRegressionObjective(acceptance=acceptance, loss=loss, revenue=revenue)
    acceptance_value = acceptance.probability(x, u)
    loss_value = loss.expected_loss(x)
    revenue_value = revenue.revenue(u)
    expected_value = acceptance_value * (loss_value - revenue_value)

    value = objective.value(x, u)
    assert math.isclose(value, expected_value, rel_tol=1e-9)


def test_fixed_regression_objective_grad() -> None:
    x = StateVector(values=[1.0, 2.0])
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    objective = FixedRegressionObjective.from_parameters(beta_1, beta_2, beta_3, beta_4)
    grad_u = objective.grad_u(x, u)
    logit = 0.1 * 1.0 + 0.2 * 2.0 + beta_2 * u
    acceptance = 1.0 / (1.0 + math.exp(-logit))
    loss = 0.3 * 1.0 + 0.4 * 2.0
    revenue_value = beta_4 * u
    d_acceptance_du = acceptance * (1.0 - acceptance) * beta_2
    expected_grad = d_acceptance_du * (loss - revenue_value) - acceptance * beta_4

    assert math.isclose(grad_u, expected_grad, rel_tol=1e-9)
