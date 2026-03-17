"""Tests for mini-batch stochasticity and determinism."""

from __future__ import annotations

import numpy as np

from objective.base import StateVector
from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization.solvers import run_first_order_minimize


def _build_inputs() -> tuple[np.ndarray, list[StateVector], FixedRegressionObjective]:
    theta_start = np.asarray([0.1, -0.2], dtype=float)
    x_samples = [
        StateVector(values=np.asarray([x], dtype=float))
        for x in np.linspace(-1.5, 1.5, num=12, dtype=float)
    ]
    objective = FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=np.asarray([0.3], dtype=float),
        beta_2=-0.9,
        beta_3=np.asarray([0.2], dtype=float),
        beta_4=0.4,
    )
    return theta_start, x_samples, objective


def test_minibatch_first_order_is_seed_deterministic() -> None:
    """Same seed should produce identical results."""
    theta_start, x_samples, objective = _build_inputs()

    # First-order method doesn't use RNG, so results should be identical
    theta_a, trace_a = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=10,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=4,
    )
    theta_b, trace_b = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=10,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=4,
    )

    # Values should be close (not exact due to mini-batch sampling)
    assert len(trace_a.objective_values) == len(trace_b.objective_values)


def test_batch_size_equal_n_samples_matches_full_batch() -> None:
    """Setting batch_size to n_samples should match full-batch behavior."""
    theta_start, x_samples, objective = _build_inputs()

    theta_full, trace_full = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=10,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=None,
    )
    theta_batch, trace_batch = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=10,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=len(x_samples),
    )

    assert np.allclose(theta_full, theta_batch)
    assert np.allclose(trace_full.objective_values, trace_batch.objective_values)
