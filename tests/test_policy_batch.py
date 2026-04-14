import numpy as np

from objective.policy import (
    ConstantPolicy,
    LinearPolicy,
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


def test_policy_kind_constants() -> None:
    """Test that policy kind attributes are correct."""
    assert ConstantPolicy().kind == "constant"
    assert LinearPolicy().kind == "linear"
    assert SoftmaxPolicy().kind == "softmax"
