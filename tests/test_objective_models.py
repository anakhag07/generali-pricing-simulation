from __future__ import annotations

import math
import numpy as np

from objective import (
    FixedRegressionObjective,
    SoftmaxPolicy,
)


def test_fixed_regression_objective_value() -> None:
    """Test that value method computes correct mean over batch."""
    x_batch = np.array([[1.0, 2.0]], dtype=float)  # Single sample batch
    theta = np.array([0.1, 0.0, 0.0], dtype=float)  # Softmax policy params
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

    # Compute expected value manually
    # Policy outputs u = 0.5 + sigmoid(theta.T @ [1, x])
    # With theta = [0.1, 0, 0] and x = [1, 2], phi = [1, 1, 2]
    # z = 0.1*1 + 0*1 + 0*2 = 0.1
    # u = 0.5 + sigmoid(0.1)
    z = 0.1
    u = 0.5 + 1.0 / (1.0 + math.exp(-z))
    
    logit = 0.1 * 1.0 + 0.2 * 2.0 + beta_2 * u
    acceptance = 1.0 / (1.0 + math.exp(-logit))
    loss = 0.3 * 1.0 + 0.4 * 2.0
    revenue = beta_4 * u
    expected_value = acceptance * (loss - revenue)

    value = objective.value(theta, x_batch)
    assert math.isclose(value, expected_value, rel_tol=1e-9)


def test_fixed_regression_objective_grad() -> None:
    """Test that grad method returns finite gradients."""
    x_batch = np.array([[1.0, 2.0], [0.5, -0.5]], dtype=float)
    theta = np.array([0.1, 0.0, 0.0], dtype=float)

    objective = FixedRegressionObjective.from_parameters(
        policy=SoftmaxPolicy(),
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.4],
        beta_4=0.5,
    )

    grad = objective.grad(theta, x_batch)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))


def test_fixed_regression_value_batch_consistency() -> None:
    """Test that _value_batch results are used consistently in value."""
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.4],
        beta_4=0.5,
    )

    x_batch = np.array([[1.0, 2.0], [0.5, -0.5]], dtype=float)
    theta = np.array([0.1, 0.0, 0.0], dtype=float)
    
    # Get u values from policy
    u_batch = policy.value(theta, x_batch)
    
    # Internal _value_batch should match
    internal_values = objective._value_batch(x_batch, u_batch)
    external_value = objective.value(theta, x_batch)
    
    assert math.isclose(external_value, float(np.mean(internal_values)), rel_tol=1e-9)
