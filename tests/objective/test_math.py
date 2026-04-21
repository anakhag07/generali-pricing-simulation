"""Tests for src/objective/_math.py :: _sigmoid."""

import numpy as np
import pytest

from objective._math import _sigmoid


def test_sigmoid_at_zero():
    """sigma(0) == 0.5 exactly."""
    assert _sigmoid(np.array(0.0)) == pytest.approx(0.5, abs=1e-15)


def test_sigmoid_symmetry():
    """sigma(z) + sigma(-z) == 1.0 for various z."""
    z_values = np.array([-10.0, -1.0, -0.5, 0.1, 2.0, 8.0])
    np.testing.assert_allclose(
        _sigmoid(z_values) + _sigmoid(-z_values), 1.0, atol=1e-14
    )


def test_sigmoid_large_positive():
    """sigma(500) approx 1.0, no overflow."""
    result = _sigmoid(np.array(500.0))
    assert np.isfinite(result)
    assert result == pytest.approx(1.0, abs=1e-10)


def test_sigmoid_large_negative():
    """sigma(-500) approx 0.0, no overflow."""
    result = _sigmoid(np.array(-500.0))
    assert np.isfinite(result)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_sigmoid_monotonic():
    """sigma is non-decreasing on a sorted array."""
    z = np.linspace(-10, 10, 200)
    out = _sigmoid(z)
    assert np.all(np.diff(out) >= 0.0)


def test_sigmoid_derivative_fd():
    """sigma'(z) = sigma(z)(1 - sigma(z)) matches FD within 1e-7."""
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    s = _sigmoid(z)
    analytical = s * (1.0 - s)
    h = 1e-5
    fd = (_sigmoid(z + h) - _sigmoid(z - h)) / (2.0 * h)
    np.testing.assert_allclose(analytical, fd, atol=1e-7)


def test_sigmoid_output_range():
    """All outputs in (0, 1) for random inputs."""
    rng = np.random.default_rng(42)
    z = rng.normal(0, 10, size=1000)
    out = _sigmoid(z)
    assert np.all(out > 0.0)
    assert np.all(out < 1.0)
