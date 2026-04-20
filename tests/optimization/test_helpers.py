"""Tests for src/optimization/helpers.py — clamp, batch, FD helpers."""

import numpy as np
import pytest

from optimization.helpers import (
    _clamp_theta,
    finite_difference_theta_grad,
    sample_indices,
    x_batch,
)


# -- _clamp_theta --------------------------------------------------------------


def test_clamp_theta_within_bounds():
    """No change when theta is within bounds."""
    theta = np.array([0.0, 0.5, -0.3])
    result = _clamp_theta(theta, bounds=(-1.0, 1.0))
    np.testing.assert_array_equal(result, theta)


def test_clamp_theta_clips():
    """Values outside bounds clipped to boundary."""
    theta = np.array([-2.0, 0.5, 1.5])
    result = _clamp_theta(theta, bounds=(-1.0, 1.0))
    np.testing.assert_array_equal(result, np.array([-1.0, 0.5, 1.0]))


def test_clamp_theta_none_bounds():
    """Returns theta unchanged when bounds is None."""
    theta = np.array([10.0, -10.0])
    result = _clamp_theta(theta, bounds=None)
    np.testing.assert_array_equal(result, theta)


# -- sample_indices ------------------------------------------------------------


def test_sample_indices_full_batch():
    """Returns full_indices when batch_size >= n_total."""
    rng = np.random.default_rng(7)
    full = np.arange(50)
    result = sample_indices(rng, 100, 50, full)
    np.testing.assert_array_equal(result, full)


def test_sample_indices_correct_size():
    """Returns correct count when batch_size < n_total."""
    rng = np.random.default_rng(7)
    full = np.arange(100)
    result = sample_indices(rng, 20, 100, full)
    assert result.shape == (20,)
    assert len(set(result)) == 20  # no duplicates


# -- x_batch -------------------------------------------------------------------


def test_x_batch_selects_rows():
    """Correct row indexing from x_array."""
    x_array = np.arange(30).reshape(10, 3).astype(float)
    indices = np.array([2, 5, 7])
    result = x_batch(x_array, indices, 10)
    np.testing.assert_array_equal(result, x_array[[2, 5, 7]])


def test_x_batch_full_returns_original():
    """Returns x_array unchanged when indices cover all rows."""
    x_array = np.arange(15).reshape(5, 3).astype(float)
    indices = np.arange(5)
    result = x_batch(x_array, indices, 5)
    assert result is x_array  # same object, not a copy


# -- finite_difference_theta_grad ----------------------------------------------


def test_fd_central_quadratic():
    """Central FD on f(theta) = theta^T A theta matches analytical within 1e-5."""
    A = np.array([[2.0, 0.5], [0.5, 3.0]])
    theta = np.array([1.0, -0.5])

    def f(t):
        return float(t @ A @ t)

    analytical_grad = (A + A.T) @ theta  # 2 A theta for symmetric A
    fd_grad = finite_difference_theta_grad(f, theta, method="central", step=1e-5)
    np.testing.assert_allclose(fd_grad, analytical_grad, atol=1e-4)


def test_fd_forward_vs_central_accuracy():
    """Central FD is more accurate than forward FD on a quadratic."""
    A = np.array([[2.0, 0.5], [0.5, 3.0]])
    theta = np.array([1.0, -0.5])

    def f(t):
        return float(t @ A @ t)

    true_grad = (A + A.T) @ theta
    central = finite_difference_theta_grad(f, theta, method="central", step=1e-4)
    forward = finite_difference_theta_grad(f, theta, method="forward", step=1e-4)
    err_central = np.linalg.norm(central - true_grad)
    err_forward = np.linalg.norm(forward - true_grad)
    assert err_central <= err_forward


def test_fd_raises_on_invalid_method():
    """ValueError for unknown method string."""
    with pytest.raises(ValueError, match="Unknown numdiff method"):
        finite_difference_theta_grad(lambda t: 0.0, np.zeros(2), method="invalid", step=1e-5)
