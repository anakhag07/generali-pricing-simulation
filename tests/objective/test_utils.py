"""Tests for src/objective/utils.py — chain rule, optimal_u, action_value_at_u."""

import numpy as np
import pytest

from objective.policy import ConstantPolicy, LinearPolicy
from objective.objectives.fixed_regression import FixedRegressionObjective
from objective.objectives.planted_logistic import PlantedLogisticObjective
from objective.utils import (
    _theta_grad_from_u_grad,
    _action_value_at_u,
    mean_acceptance_at_constant_u,
    optimal_u,
    value_at_constant_u,
    value_for_reporting,
)


def _make_fixed_regression(policy, state_dim=3):
    return FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.ones(state_dim),
        beta_2=-1.0,
        beta_3=np.ones(state_dim),
        beta_4=1.0,
    )


# -- _theta_grad_from_u_grad --------------------------------------------------


def test_theta_grad_constant_policy():
    """With ConstantPolicy, theta_grad == mean(grad_u) * [1, 0, ...]."""
    rng = np.random.default_rng(7)
    state_dim = 3
    policy = ConstantPolicy()
    theta = np.array([0.1, 0.0, 0.0, 0.0])
    x_batch = rng.normal(size=(50, state_dim))
    grad_u = rng.normal(size=50)

    result = _theta_grad_from_u_grad(policy, theta, x_batch, grad_u)
    expected = np.zeros(4)
    expected[0] = np.mean(grad_u)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_theta_grad_linear_policy_vs_fd():
    """Chain-rule result for LinearPolicy matches FD within 1e-5."""
    rng = np.random.default_rng(42)
    state_dim = 3
    policy = LinearPolicy()
    objective = _make_fixed_regression(policy, state_dim)
    theta = rng.normal(size=state_dim + 1) * 0.1
    x_batch = rng.normal(size=(100, state_dim))

    # Analytical gradient via chain rule (what the objective computes internally)
    analytical = objective.grad(theta, x_batch)

    # FD gradient
    h = 1e-5
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = 1.0
        fd[i] = (objective.value(theta + h * e, x_batch) - objective.value(theta - h * e, x_batch)) / (2.0 * h)

    np.testing.assert_allclose(analytical, fd, atol=1e-4)


def test_theta_grad_shape():
    """Output shape matches theta dimension."""
    policy = LinearPolicy()
    theta = np.zeros(4)
    x_batch = np.random.default_rng(7).normal(size=(20, 3))
    grad_u = np.ones(20)
    result = _theta_grad_from_u_grad(policy, theta, x_batch, grad_u)
    assert result.shape == (4,)


def test_theta_grad_zero_input():
    """Zero grad_u produces zero theta_grad."""
    policy = LinearPolicy()
    theta = np.array([0.5, 0.1, -0.2, 0.3])
    x_batch = np.random.default_rng(7).normal(size=(20, 3))
    grad_u = np.zeros(20)
    result = _theta_grad_from_u_grad(policy, theta, x_batch, grad_u)
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


# -- optimal_u -----------------------------------------------------------------


def test_optimal_u_planted():
    """Returns u_star from PlantedLogisticObjective."""
    policy = ConstantPolicy()
    obj = PlantedLogisticObjective.from_parameters(
        policy=policy, alpha=1.0, beta=np.array([0.5, -0.3, 0.2]), bias=0.0, u_star=1.1,
    )
    assert optimal_u(obj) == pytest.approx(1.1)


def test_optimal_u_returns_none():
    """Returns None for objectives without optimal_u."""
    policy = ConstantPolicy()
    obj = _make_fixed_regression(policy)
    assert optimal_u(obj) is None


# -- _action_value_at_u --------------------------------------------------------


def test_action_value_at_u_delegates():
    """Correctly delegates to objective.value_at_u."""
    rng = np.random.default_rng(7)
    policy = ConstantPolicy()
    obj = _make_fixed_regression(policy)
    x_batch = rng.normal(size=(50, 3))
    u = 0.1
    result = _action_value_at_u(obj, x_batch, u)
    expected = obj.value_at_u(x_batch, u)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("u", [-0.3, 0.0, 0.2])
def test_value_at_constant_u_matches_objective_value_at_u(u: float) -> None:
    rng = np.random.default_rng(11)
    policy = ConstantPolicy()
    obj = _make_fixed_regression(policy)
    x_batch = rng.normal(size=(40, 3))

    assert value_at_constant_u(obj, x_batch, u) == pytest.approx(obj.value_at_u(x_batch, u))


@pytest.mark.parametrize("u", [-0.3, 0.0, 0.2])
def test_mean_acceptance_at_constant_u_matches_fixed_regression_formula(u: float) -> None:
    x_batch = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]], dtype=float)
    obj = FixedRegressionObjective.from_parameters(
        policy=ConstantPolicy(),
        beta_1=np.array([0.2, 0.1, 0.3], dtype=float),
        beta_2=-0.7,
        beta_3=np.ones(3, dtype=float),
        beta_4=0.4,
    )
    logits = x_batch @ obj.beta_1 + obj.beta_2 * u
    expected = 1.0 / (1.0 + np.exp(-logits))

    assert mean_acceptance_at_constant_u(obj, x_batch, u) == pytest.approx(float(np.mean(expected)))


def test_value_for_reporting_prefers_base_value() -> None:
    class ObjectiveWithBaseValue:
        def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
            del theta, x_batch
            return 10.0

        def base_value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
            del theta, x_batch
            return 3.0

    objective = ObjectiveWithBaseValue()
    theta = np.array([0.0], dtype=float)
    x_batch = np.zeros((2, 1), dtype=float)

    assert value_for_reporting(objective, theta, x_batch) == pytest.approx(3.0)


def test_value_at_constant_u_prefers_base_value_at_u() -> None:
    class ObjectiveWithBaseValueAtU:
        def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
            del x_batch, u
            return 10.0

        def base_value_at_u(self, x_batch: np.ndarray, u: float) -> float:
            del x_batch, u
            return 3.0

    objective = ObjectiveWithBaseValueAtU()
    x_batch = np.zeros((2, 1), dtype=float)

    assert value_at_constant_u(objective, x_batch, 0.2) == pytest.approx(3.0)
