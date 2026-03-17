from __future__ import annotations

import math
import numpy as np

from objective import (
    FixedRegressionObjective,
    SoftmaxPolicy,
)
from objective.base import StateVector


def test_fixed_regression_objective_value_scalar() -> None:
    """Test that value_scalar matches the expected formula."""
    x = StateVector(values=np.asarray([1.0, 2.0], dtype=float))
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    objective = FixedRegressionObjective.from_parameters(
        policy=SoftmaxPolicy(),
        beta_1=beta_1,
        beta_2=beta_2,
        beta_3=beta_3,
        beta_4=beta_4,
    )

    # Compute expected value
    logit = 0.1 * 1.0 + 0.2 * 2.0 + beta_2 * u
    acceptance = 1.0 / (1.0 + math.exp(-logit))
    loss = 0.3 * 1.0 + 0.4 * 2.0
    revenue = beta_4 * u
    expected_value = acceptance * (loss - revenue)

    value = objective.value_scalar(x, u)
    assert math.isclose(value, expected_value, rel_tol=1e-9)


def test_fixed_regression_objective_grad_u_scalar() -> None:
    """Test that grad_u_scalar matches the expected formula."""
    x = StateVector(values=np.asarray([1.0, 2.0], dtype=float))
    u = 1.0
    beta_1 = [0.1, 0.2]
    beta_2 = -0.5
    beta_3 = [0.3, 0.4]
    beta_4 = 0.5

    objective = FixedRegressionObjective.from_parameters(
        policy=SoftmaxPolicy(),
        beta_1=beta_1,
        beta_2=beta_2,
        beta_3=beta_3,
        beta_4=beta_4,
    )

    # Compute expected gradient
    logit = 0.1 * 1.0 + 0.2 * 2.0 + beta_2 * u
    acceptance = 1.0 / (1.0 + math.exp(-logit))
    loss = 0.3 * 1.0 + 0.4 * 2.0
    revenue_value = beta_4 * u
    d_acceptance_du = acceptance * (1.0 - acceptance) * beta_2
    expected_grad = d_acceptance_du * (loss - revenue_value) - acceptance * beta_4

    grad_u = objective.grad_u_scalar(x, u)
    assert math.isclose(grad_u, expected_grad, rel_tol=1e-9)


def test_fixed_regression_objective_batch_matches_scalar() -> None:
    """Test that batch methods match scalar methods."""
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.4],
        beta_4=0.5,
    )

    x1 = np.asarray([1.0, 2.0], dtype=float)
    x2 = np.asarray([0.5, -0.5], dtype=float)
    x_batch = np.stack([x1, x2], axis=0)
    u_vals = np.asarray([0.8, 1.2], dtype=float)

    # Scalar evaluation
    v1 = objective.value_scalar(StateVector(values=x1), u_vals[0])
    v2 = objective.value_scalar(StateVector(values=x2), u_vals[1])
    expected_values = np.asarray([v1, v2], dtype=float)

    batch_values = objective._value_batch(x_batch, u_vals)
    assert np.allclose(batch_values, expected_values, rtol=1e-9)
