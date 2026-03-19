"""Tests for optimization solver wrappers."""

import numpy as np
from types import SimpleNamespace

from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization.solvers import (
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_spsa_minimize,
    run_stein_difference_minimize,
)
import optimization.solvers as solvers


def _build_theta_objective() -> FixedRegressionObjective:
    """Build a simple theta-space objective for testing."""
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def test_first_order_minimize_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    theta_final, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=25,
        n_grad_samples=4,
        sigma=0.1,
    )
    assert trace.objective_values
    assert trace.objective_values[-1] <= trace.objective_values[0] + 1e-10
    assert theta_final.shape == theta_start.shape


def test_gauss_stein_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    theta_a, trace_a = run_gauss_stein_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_gauss_stein_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_spsa_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    theta_a, trace_a = run_spsa_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(17),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_spsa_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(17),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_stein_difference_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    theta_a, trace_a = run_stein_difference_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(23),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_stein_difference_minimize(
        theta_start,
        x_samples,
        objective,
        np.random.default_rng(23),
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_first_order_passes_ftol_to_minimize(monkeypatch) -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()
    captured_options: dict[str, float | int] = {}

    def fake_minimize(fun, x0, jac, method, options, callback):  # type: ignore[no-untyped-def]
        del fun, jac
        assert method == "L-BFGS-B"
        captured_options.update(options)
        callback(x0)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            status=0,
            message="stub",
        )

    monkeypatch.setattr(solvers, "minimize", fake_minimize)

    _, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=25,
        n_grad_samples=4,
        sigma=0.1,
        grad_norm_tol=1e-6,
        ftol=1e-9,
    )

    assert captured_options["maxiter"] == 25
    assert captured_options["gtol"] == 1e-6
    assert captured_options["ftol"] == 1e-9
    assert trace.optimizer_status == 0
    assert trace.optimizer_message == "stub"
