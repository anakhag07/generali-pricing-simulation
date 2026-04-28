import numpy as np
import pytest

from objective.policy import (
    ConstantPolicy,
    IdentityFeatureMap,
    LinearPolicy,
    QuadraticFeatureMap,
    SoftmaxPolicy,
)


def test_constant_policy_value_shape() -> None:
    """Test that ConstantPolicy.value returns correct shape."""
    x_array = np.random.default_rng(0).normal(size=(5, 3))
    theta = np.array([1.1], dtype=float)
    policy = ConstantPolicy()
    result = policy.value(theta, x_array)
    assert result.shape == (5,)
    assert np.allclose(result, 1.1)


def test_linear_policy_value_shape() -> None:
    """Test that LinearPolicy.value returns correct shape."""
    x_array = np.random.default_rng(1).normal(size=(5, 3))
    theta = np.array([0.2, -0.1, 0.3, 0.4], dtype=float)
    policy = LinearPolicy()
    result = policy.value(theta, x_array)
    assert result.shape == (5,)


def test_softmax_policy_value_range() -> None:
    """Test that SoftmaxPolicy.value returns values in (-0.5, 0.5)."""
    x_array = np.random.default_rng(2).normal(size=(10, 3))
    theta = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    policy = SoftmaxPolicy()
    result = policy.value(theta, x_array)
    assert result.shape == (10,)
    assert np.all(result > -0.5)
    assert np.all(result < 0.5)


def test_constant_policy_grad_shape() -> None:
    """Test that ConstantPolicy.grad returns correct shape."""
    x_array = np.random.default_rng(3).normal(size=(5, 3))
    theta = np.array([1.1], dtype=float)
    policy = ConstantPolicy()
    result = policy.grad(theta, x_array)
    assert result.shape == (5, 1)
    assert np.allclose(result[:, 0], 1.0)


def test_linear_policy_grad_shape() -> None:
    """Test that LinearPolicy.grad returns correct shape."""
    x_array = np.random.default_rng(4).normal(size=(5, 3))
    theta = np.array([0.2, -0.1, 0.3, 0.4], dtype=float)
    policy = LinearPolicy()
    result = policy.grad(theta, x_array)
    assert result.shape == (5, 4)


def test_softmax_policy_grad_shape() -> None:
    """Test that SoftmaxPolicy.grad returns correct shape."""
    x_array = np.random.default_rng(5).normal(size=(5, 3))
    theta = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    policy = SoftmaxPolicy()
    result = policy.grad(theta, x_array)
    assert result.shape == (5, 4)


def test_softmax_policy_grad_matches_closed_form() -> None:
    """Test that SoftmaxPolicy.grad matches the exact Jacobian at a known point."""
    x_array = np.array([[1.0, -2.0]], dtype=float)
    theta = np.zeros(3, dtype=float)

    result = SoftmaxPolicy().grad(theta, x_array)
    expected = np.array([[-0.25, -0.25, 0.5]], dtype=float)

    assert np.allclose(result, expected)


def test_default_feature_map_policy_theta_dim() -> None:
    assert LinearPolicy().theta_dim(3) == 4
    assert SoftmaxPolicy().theta_dim(3) == 4
    assert LinearPolicy().feature_map == IdentityFeatureMap()
    assert SoftmaxPolicy().feature_map == IdentityFeatureMap()


def test_linear_policy_with_quadratic_feature_map() -> None:
    x_array = np.array([[2.0, 3.0]], dtype=float)
    theta = np.array([0.5, 1.0, -1.0, 0.25, 0.1, -0.2], dtype=float)
    policy = LinearPolicy(feature_map=QuadraticFeatureMap())

    result = policy.value(theta, x_array)
    grad = policy.grad(theta, x_array)
    expected_phi = np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]], dtype=float)

    assert policy.theta_dim(2) == 6
    np.testing.assert_allclose(result, expected_phi @ theta)
    np.testing.assert_allclose(grad, expected_phi)


def test_softmax_policy_with_quadratic_feature_map_grad_matches_closed_form() -> None:
    x_array = np.array([[2.0, -1.0]], dtype=float)
    theta = np.array([0.1, 0.2, -0.3, 0.05, 0.01, -0.02], dtype=float)
    policy = SoftmaxPolicy(feature_map=QuadraticFeatureMap())

    features = np.array([[1.0, 2.0, -1.0, 4.0, -2.0, 1.0]], dtype=float)
    z = float((features @ theta)[0])
    sigma = 1.0 / (1.0 + np.exp(-z))
    expected = -sigma * (1.0 - sigma) * features

    result = policy.grad(theta, x_array)

    np.testing.assert_allclose(result, expected)


def test_softmax_policy_with_quadratic_feature_map_grad_matches_fd() -> None:
    x_array = np.array([[0.4, -0.7]], dtype=float)
    policy = SoftmaxPolicy(feature_map=QuadraticFeatureMap())
    theta = np.array([0.1, -0.2, 0.3, 0.05, -0.04, 0.02], dtype=float)
    step = 1e-6

    analytical = policy.grad(theta, x_array)[0]
    fd = np.zeros_like(theta)
    for idx in range(theta.size):
        direction = np.zeros_like(theta)
        direction[idx] = 1.0
        upper = policy.value(theta + step * direction, x_array)[0]
        lower = policy.value(theta - step * direction, x_array)[0]
        fd[idx] = (upper - lower) / (2.0 * step)

    np.testing.assert_allclose(analytical, fd, atol=1e-8)


def test_policy_rejects_inconsistent_theta_dim() -> None:
    x_array = np.zeros((2, 3), dtype=float)

    with pytest.raises(ValueError, match="exactly 4"):
        LinearPolicy().value(np.zeros(3), x_array)


def test_policy_kind_constants() -> None:
    """Test that policy kind attributes are correct."""
    assert ConstantPolicy().kind == "constant"
    assert LinearPolicy().kind == "linear"
    assert SoftmaxPolicy().kind == "softmax"
