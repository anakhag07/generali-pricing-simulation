from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from objective.base import Objective, Policy
from objective.policy import LinearPolicy
from objective.utils import _theta_grad_from_u_grad
from scripts import diagnose_stein_backend_divergence as script


@dataclass(frozen=True)
class _QuadraticObjective(Objective):
    policy: Policy
    c: float = 0.3

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        u = self.policy.value(theta, x_batch)
        return float(np.mean((u - self.c) ** 2))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        u = self.policy.value(theta, x_batch)
        return _theta_grad_from_u_grad(self.policy, theta, x_batch, 2.0 * (u - self.c))

    def _value_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        del x_batch
        return (np.asarray(u_arr, dtype=float) - self.c) ** 2


def test_fixed_sample_stein_probe_is_identical_for_same_objective() -> None:
    objective = _QuadraticObjective(policy=LinearPolicy())
    x = np.array([[1.0, -0.5], [0.2, 0.3], [-0.1, 0.8]], dtype=float)
    theta = np.array([0.1, -0.2, 0.4], dtype=float)
    w = np.array([0.5, -1.2, 0.7], dtype=float)

    first = script.stein_gradient_with_samples(objective, x, theta, w, sigma=0.05)
    second = script.stein_gradient_with_samples(objective, x, theta, w, sigma=0.05)
    metrics = script.probe_difference_metrics(first, second)

    assert metrics["grad_linf_diff"] == 0.0
    assert metrics["grad_u_linf_diff"] == 0.0
    assert metrics["u_linf_diff"] == 0.0
    assert metrics["grad_cosine"] == pytest.approx(1.0)


def test_trace_comparison_reports_first_rng_block_difference() -> None:
    numpy_events = [
        {
            "event": "gradient",
            "theta_hash": "a",
            "source": "jac",
            "w_hash": "same",
            "rng_state_before": "s0",
            "peer_grad_linf_diff": 0.0,
        },
        {
            "event": "gradient",
            "theta_hash": "b",
            "source": "jac",
            "w_hash": "left",
            "rng_state_before": "s1",
            "peer_grad_linf_diff": 0.0,
        },
    ]
    jax_events = [
        {
            "event": "gradient",
            "theta_hash": "a",
            "source": "jac",
            "w_hash": "same",
            "rng_state_before": "s0",
            "peer_grad_linf_diff": 0.0,
        },
        {
            "event": "gradient",
            "theta_hash": "b",
            "source": "jac",
            "w_hash": "right",
            "rng_state_before": "s1",
            "peer_grad_linf_diff": 0.0,
        },
    ]

    diff = script.compare_event_traces(
        numpy_events,
        jax_events,
        theta_tol=1e-8,
        grad_tol=1e-8,
        value_tol=1e-8,
    )

    assert diff["gradient"]["first_difference_index"] == 1
    assert diff["gradient"]["first_difference_reason"] == "w_hash"
