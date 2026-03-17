import numpy as np

from objective.base import StateVector
from objective.policy import (
    ConstantPolicy,
    LinearPolicy,
    SoftmaxPolicy,
    policy_constant,
    policy_linear,
    policy_softmax,
)


def test_policy_value_batch_matches_scalar() -> None:
    """Test that policy.value_batch matches per-sample policy.value calls."""
    rng = np.random.default_rng(0)
    x_array = rng.normal(size=(5, 3))
    x_samples = [StateVector(values=row) for row in x_array]

    # Constant policy
    theta_const = np.array([1.1], dtype=float)
    const_policy = ConstantPolicy()
    batch_const = const_policy.value_batch(theta_const, x_array)
    scalar_const = np.array([const_policy.value(theta_const, x) for x in x_samples])
    assert np.allclose(batch_const, scalar_const)

    # Linear policy
    theta_linear = np.array([0.2, -0.1, 0.3, 0.4], dtype=float)
    linear_policy = LinearPolicy()
    batch_linear = linear_policy.value_batch(theta_linear, x_array)
    scalar_linear = np.array([linear_policy.value(theta_linear, x) for x in x_samples])
    assert np.allclose(batch_linear, scalar_linear)

    # Softmax policy
    theta_softmax = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    softmax_policy = SoftmaxPolicy()
    batch_softmax = softmax_policy.value_batch(theta_softmax, x_array)
    scalar_softmax = np.array([softmax_policy.value(theta_softmax, x) for x in x_samples])
    assert np.allclose(batch_softmax, scalar_softmax)


def test_policy_grad_batch_matches_scalar() -> None:
    """Test that policy.grad_batch matches per-sample policy.grad calls."""
    rng = np.random.default_rng(1)
    x_array = rng.normal(size=(5, 3))
    x_samples = [StateVector(values=row) for row in x_array]

    # Constant policy
    theta_const = np.array([1.1], dtype=float)
    const_policy = ConstantPolicy()
    batch_grad_const = const_policy.grad_batch(theta_const, x_array)
    scalar_grad_const = np.array([const_policy.grad(theta_const, x) for x in x_samples])
    assert np.allclose(batch_grad_const, scalar_grad_const)

    # Linear policy
    theta_linear = np.array([0.2, -0.1, 0.3, 0.4], dtype=float)
    linear_policy = LinearPolicy()
    batch_grad_linear = linear_policy.grad_batch(theta_linear, x_array)
    scalar_grad_linear = np.array([linear_policy.grad(theta_linear, x) for x in x_samples])
    assert np.allclose(batch_grad_linear, scalar_grad_linear)

    # Softmax policy
    theta_softmax = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    softmax_policy = SoftmaxPolicy()
    batch_grad_softmax = softmax_policy.grad_batch(theta_softmax, x_array)
    scalar_grad_softmax = np.array([softmax_policy.grad(theta_softmax, x) for x in x_samples])
    assert np.allclose(batch_grad_softmax, scalar_grad_softmax)


def test_policy_kind_constants() -> None:
    """Test that policy kind constants are correct."""
    assert policy_constant == "constant"
    assert policy_linear == "linear"
    assert policy_softmax == "softmax"
    assert ConstantPolicy().kind == policy_constant
    assert LinearPolicy().kind == policy_linear
    assert SoftmaxPolicy().kind == policy_softmax
