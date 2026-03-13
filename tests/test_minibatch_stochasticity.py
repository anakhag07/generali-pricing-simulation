from __future__ import annotations

import numpy as np

from experiments.helpers import run_first_order
from objective.base import StateVector
from objective.fixed_objective import FixedRegressionObjective
from objective.policy import POLICY_LINEAR


def _build_inputs() -> tuple[np.ndarray, list[StateVector], FixedRegressionObjective]:
    theta_start = np.asarray([0.1, -0.2], dtype=float)
    x_samples = [
        StateVector(values=np.asarray([x], dtype=float))
        for x in np.linspace(-1.5, 1.5, num=12, dtype=float)
    ]
    objective = FixedRegressionObjective.from_parameters(
        beta_1=np.asarray([0.3], dtype=float),
        beta_2=-0.9,
        beta_3=np.asarray([0.2], dtype=float),
        beta_4=0.4,
    )
    return theta_start, x_samples, objective


def test_minibatch_first_order_is_seed_deterministic() -> None:
    theta_start, x_samples, objective = _build_inputs()

    theta_a, trace_a = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(21),
        t_steps=15,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=4,
    )
    theta_b, trace_b = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(21),
        t_steps=15,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=4,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_batch_size_equal_n_samples_matches_full_batch() -> None:
    theta_start, x_samples, objective = _build_inputs()

    theta_full, trace_full = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(12),
        t_steps=10,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=None,
    )
    theta_batch, trace_batch = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(12),
        t_steps=10,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
        batch_size=len(x_samples),
    )

    assert np.allclose(theta_full, theta_batch)
    assert np.allclose(trace_full.objective_values, trace_batch.objective_values)
