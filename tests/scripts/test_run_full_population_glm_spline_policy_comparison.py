"""Tests for the full-customer GLM/spline policy comparison."""

from __future__ import annotations

import numpy as np

from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from experiments.policy_utils import constant_softmax_theta
from scripts.run_full_population_glm_spline_policy_comparison import (
    FullCacheSplineProfitObjective,
    _histogram,
)


class _LogisticCache:
    def pairwise_acceptance_and_derivative(self, rows, actions):
        del rows
        acceptance = 1.0 / (1.0 + np.exp(4.0 * np.asarray(actions)))
        derivative = -4.0 * acceptance * (1.0 - acceptance)
        return acceptance, derivative


def _objective() -> tuple[FullCacheSplineProfitObjective, np.ndarray, np.ndarray]:
    x = np.asarray([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5], [0.1, -0.3]])
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(), action_low=-0.1, action_high=0.2
    )
    objective = FullCacheSplineProfitObjective(
        cache=_LogisticCache(),
        row_indices=np.arange(len(x)),
        loss=np.asarray([100.0, 120.0, 90.0, 130.0]),
        premium=np.asarray([220.0, 240.0, 210.0, 260.0]),
        policy=policy,
        acceptance_floor=0.7,
    )
    theta = constant_softmax_theta(policy, x.shape[1], 0.0)
    theta[1:] = [0.2, -0.1]
    return objective, x, theta


def test_objective_and_acceptance_gradients_match_central_differences() -> None:
    objective, x, theta = _objective()
    epsilon = 1e-6
    numerical_value = np.empty(theta.shape)
    numerical_acceptance = np.empty(theta.shape)
    for index in range(theta.size):
        step = np.zeros(theta.shape)
        step[index] = epsilon
        numerical_value[index] = (
            objective.value(theta + step, x) - objective.value(theta - step, x)
        ) / (2.0 * epsilon)
        numerical_acceptance[index] = (
            objective.mean_acceptance(theta + step, x)
            - objective.mean_acceptance(theta - step, x)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(objective.grad(theta, x), numerical_value, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        objective.mean_acceptance_grad(theta, x),
        numerical_acceptance,
        rtol=1e-6,
        atol=1e-7,
    )


def test_histogram_uses_fixed_decimal_bins_and_unit_density() -> None:
    frame = _histogram(
        np.asarray([-0.099, -0.05, 0.0, 0.1, 0.199]),
        series="Spline",
        population="all",
    )

    assert len(frame) == 30
    assert frame.iloc[0]["bin_left"] == -0.1
    assert np.isclose(frame.iloc[-1]["bin_right"], 0.2)
    assert np.isclose(np.sum(frame["density"] * (frame["bin_right"] - frame["bin_left"])), 1.0)
