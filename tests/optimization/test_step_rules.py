from __future__ import annotations

import numpy as np
import pytest

from optimization.steps import armijo_backtracking_step_size


def test_armijo_backtracking_reduces_objective() -> None:
    def objective(theta: np.ndarray) -> float:
        return float(theta[0] ** 2)

    theta = np.asarray([1.0], dtype=float)
    grad = np.asarray([2.0], dtype=float)
    step = armijo_backtracking_step_size(
        theta,
        grad,
        objective_fn=objective,
        initial_step=10.0,
    )

    new_theta = theta - step * grad
    assert step <= 10.0
    assert step >= 1e-6
    assert objective(new_theta) <= objective(theta) + 1e-12


def test_armijo_sufficient_decrease_holds() -> None:
    """Returned step satisfies the Armijo condition: f(new) <= f(old) + c*alpha*g^T*d."""
    A = np.array([[3.0, 0.5], [0.5, 2.0]])

    def objective(theta: np.ndarray) -> float:
        return float(theta @ A @ theta)

    theta = np.array([1.0, -0.5])
    grad = (A + A.T) @ theta  # analytical gradient of x^T A x
    c = 1e-4
    step = armijo_backtracking_step_size(theta, grad, objective, initial_step=1.0, c=c)
    direction = -grad
    new_theta = theta + step * direction
    assert objective(new_theta) <= objective(theta) + c * step * float(np.dot(grad, direction)) + 1e-12


def test_armijo_hits_min_step() -> None:
    """Returns min_step when objective never satisfies condition."""
    # Objective that always returns the same value — Armijo never satisfied
    def flat_objective(theta: np.ndarray) -> float:
        return 1.0

    theta = np.array([1.0])
    grad = np.array([1.0])
    step = armijo_backtracking_step_size(
        theta, grad, flat_objective, initial_step=1.0, min_step=0.01,
    )
    assert step == pytest.approx(0.01)


def test_armijo_zero_gradient() -> None:
    """Returns max(min_step, initial_step) for zero gradient."""
    def objective(theta: np.ndarray) -> float:
        return float(theta[0] ** 2)

    theta = np.array([1.0])
    grad = np.array([0.0])
    step = armijo_backtracking_step_size(theta, grad, objective, initial_step=0.5)
    assert step == pytest.approx(0.5)


def test_armijo_multidimensional() -> None:
    """Works correctly for theta with dim > 1."""
    def objective(theta: np.ndarray) -> float:
        return float(np.sum(theta ** 2))

    theta = np.array([1.0, 2.0, -1.5])
    grad = 2.0 * theta
    step = armijo_backtracking_step_size(theta, grad, objective, initial_step=1.0)
    new_theta = theta - step * grad
    assert objective(new_theta) < objective(theta)


def test_armijo_validates_inputs() -> None:
    """Raises ValueError for invalid parameters."""
    def f(t):
        return 0.0

    theta = np.array([1.0])
    grad = np.array([1.0])

    with pytest.raises(ValueError, match="initial_step must be positive"):
        armijo_backtracking_step_size(theta, grad, f, initial_step=-1.0)

    with pytest.raises(ValueError, match="shrink must be in"):
        armijo_backtracking_step_size(theta, grad, f, initial_step=1.0, shrink=1.5)

    with pytest.raises(ValueError, match="c must be positive"):
        armijo_backtracking_step_size(theta, grad, f, initial_step=1.0, c=-0.1)
