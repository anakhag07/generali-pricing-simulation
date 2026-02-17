from __future__ import annotations

import math

from data.models import (
    StateVector,
    acceptance_probability,
    expected_loss,
    fixed_regression_objective,
    fixed_regression_objective_with_grad,
    revenue,
)


def test_fixed_regression_objective_matches_components() -> None:
    x = StateVector(values=[1.0, 2.0])
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    acceptance = acceptance_probability(x, u, beta_1, beta_2)
    loss = expected_loss(x, beta_3)
    revenue_value = revenue(u, beta_4)
    expected_value = acceptance * (loss - revenue_value)

    value = fixed_regression_objective(x, u, beta_1, beta_2, beta_3, beta_4)
    assert math.isclose(value, expected_value, rel_tol=1e-9)


def test_fixed_regression_objective_grad() -> None:
    x = StateVector(values=[1.0, 2.0])
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    result = fixed_regression_objective_with_grad(x, u, beta_1, beta_2, beta_3, beta_4)
    logit = 0.1 * 1.0 + 0.2 * 2.0 + beta_2 * u
    acceptance = 1.0 / (1.0 + math.exp(-logit))
    loss = 0.3 * 1.0 + 0.4 * 2.0
    revenue_value = beta_4 * u
    d_acceptance_du = acceptance * (1.0 - acceptance) * beta_2
    expected_grad = d_acceptance_du * (loss - revenue_value) - acceptance * beta_4

    assert math.isclose(result.grad_u, expected_grad, rel_tol=1e-9)
