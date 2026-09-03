import numpy as np
import pytest

from objective.policy import (
    AdditiveChebyshevFeatureMap,
    ConstantPolicy,
    CubicFeatureMap,
    IdentityFeatureMap,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    SoftmaxPolicy,
    mlp_init_theta,
    policy_from_kind,
)


def test_additive_chebyshev_feature_map_is_nested_and_degree_major() -> None:
    x_array = np.array([[1.5, -3.0], [6.0, 0.0]], dtype=float)
    degree_two = AdditiveChebyshevFeatureMap(max_degree=2)
    degree_three = AdditiveChebyshevFeatureMap(max_degree=3)

    mapped_two = degree_two.transform(x_array)
    mapped_three = degree_three.transform(x_array)

    expected_t = np.array([[0.5, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(mapped_two[:, :2], expected_t)
    np.testing.assert_allclose(mapped_two[:, 2:], 2.0 * expected_t**2 - 1.0)
    np.testing.assert_allclose(mapped_three[:, :4], mapped_two)
    assert degree_two.output_dim(2) == 4
    assert SoftmaxPolicy(feature_map=degree_two).theta_dim(2) == 5


def test_additive_chebyshev_degree_zero_is_intercept_only() -> None:
    x_array = np.ones((3, 19), dtype=float)
    feature_map = AdditiveChebyshevFeatureMap(max_degree=0)

    assert feature_map.transform(x_array).shape == (3, 0)
    assert SoftmaxPolicy(feature_map=feature_map).theta_dim(19) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_degree": -1}, "non-negative"),
        ({"max_degree": 1, "clip_scale": 0.0}, "positive"),
    ],
)
def test_additive_chebyshev_rejects_invalid_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        AdditiveChebyshevFeatureMap(**kwargs)


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


def test_softmax_policy_custom_action_bounds() -> None:
    """Test that SoftmaxPolicy supports custom open action bounds."""
    x_array = np.random.default_rng(20).normal(size=(10, 3))
    theta = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    policy = SoftmaxPolicy(action_low=-0.1, action_high=0.2)

    result = policy.value(theta, x_array)
    midpoint = policy.value(np.zeros_like(theta), x_array)

    assert result.shape == (10,)
    assert np.all(result > -0.1)
    assert np.all(result < 0.2)
    np.testing.assert_allclose(midpoint, 0.05)


def test_softmax_policy_rejects_invalid_action_bounds() -> None:
    with pytest.raises(ValueError, match="action_low < action_high"):
        SoftmaxPolicy(action_low=0.2, action_high=-0.1)

    with pytest.raises(ValueError, match="finite"):
        SoftmaxPolicy(action_low=-0.1, action_high=np.inf)


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
    expected = np.array([[0.25, 0.25, -0.5]], dtype=float)

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
    expected = sigma * (1.0 - sigma) * features

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


def test_softmax_policy_custom_bounds_grad_matches_fd() -> None:
    x_array = np.array([[0.4, -0.7]], dtype=float)
    policy = SoftmaxPolicy(
        feature_map=QuadraticFeatureMap(),
        action_low=-0.1,
        action_high=0.2,
    )
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


@pytest.mark.parametrize(
    ("feature_map", "expected_dim"),
    [(CubicFeatureMap(), 7), (QuarticFeatureMap(), 8)],
)
def test_linear_policy_with_higher_order_feature_map(feature_map, expected_dim) -> None:
    x_array = np.array([[2.0, 3.0]], dtype=float)
    theta = np.arange(expected_dim, dtype=float) / 10.0
    policy = LinearPolicy(feature_map=feature_map)

    result = policy.value(theta, x_array)
    grad = policy.grad(theta, x_array)

    assert policy.theta_dim(2) == expected_dim
    assert result.shape == (1,)
    assert grad.shape == (1, expected_dim)
    np.testing.assert_allclose(result, grad @ theta)


def test_policy_rejects_inconsistent_theta_dim() -> None:
    x_array = np.zeros((2, 3), dtype=float)

    with pytest.raises(ValueError, match="exactly 4"):
        LinearPolicy().value(np.zeros(3), x_array)


def test_policy_kind_constants() -> None:
    """Test that policy kind attributes are correct."""
    assert ConstantPolicy().kind == "constant"
    assert LinearPolicy().kind == "linear"
    assert SoftmaxPolicy().kind == "softmax"
    assert MLPPolicy().kind == "mlp"


def test_mlp_policy_theta_dim_formula() -> None:
    policy = MLPPolicy(hidden=8)
    state_dim = 5
    expected = state_dim * 8 + 8 + 8 * 8 + 8 + 8 + 1
    assert policy.theta_dim(state_dim) == expected


def test_mlp_policy_theta_dim_with_quadratic_feature_map() -> None:
    policy = MLPPolicy(feature_map=QuadraticFeatureMap(), hidden=4)
    state_dim = 3
    d_in = QuadraticFeatureMap().output_dim(state_dim)  # 3 + 6 = 9
    expected = d_in * 4 + 4 + 4 * 4 + 4 + 4 + 1
    assert policy.theta_dim(state_dim) == expected


def test_mlp_policy_value_shape_and_range() -> None:
    rng = np.random.default_rng(11)
    x_array = rng.normal(size=(7, 4))
    policy = MLPPolicy(hidden=8)
    theta = mlp_init_theta(rng, d_in=4, hidden=8)
    result = policy.value(theta, x_array)
    assert result.shape == (7,)
    assert np.all(result > -0.5)
    assert np.all(result < 0.5)


def test_mlp_policy_grad_shape() -> None:
    rng = np.random.default_rng(12)
    x_array = rng.normal(size=(7, 4))
    policy = MLPPolicy(hidden=8)
    theta = mlp_init_theta(rng, d_in=4, hidden=8)
    grad = policy.grad(theta, x_array)
    assert grad.shape == (7, policy.theta_dim(4))


def test_policy_from_kind_returns_mlp() -> None:
    policy = policy_from_kind("mlp")
    assert isinstance(policy, MLPPolicy)


def test_mlp_policy_rejects_zero_hidden() -> None:
    with pytest.raises(ValueError, match="hidden"):
        MLPPolicy(hidden=0)
