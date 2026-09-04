"""Focused tests for the spline support-lower-bound policy analysis."""

from __future__ import annotations

import numpy as np

from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from scripts import run_spline_lower_bound_policy as script


def _objective() -> tuple[script.SplineSupportLowerBoundObjective, np.ndarray]:
    grid = np.linspace(-0.1, 0.2, 31)
    x = np.asarray(
        [
            [-1.0, 0.5],
            [-0.2, -0.5],
            [0.3, 0.2],
            [1.1, -0.1],
        ]
    )
    acceptance = np.vstack(
        [0.95 - (0.2 + 0.02 * index) * (grid + 0.1) for index in range(len(x))]
    )
    cost = np.vstack(
        [
            -100.0 - (8.0 + index) * grid + 20.0 * np.square(grid - 0.05)
            for index in range(len(x))
        ]
    )
    width = 1.0 + 20.0 * np.square(grid - 0.04)
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=-0.1,
        action_high=0.2,
    )
    objective = script.SplineSupportLowerBoundObjective(
        policy=policy,
        cost_grid=cost,
        acceptance_grid=acceptance,
        action_grid=grid,
        support_width=width,
        acceptance_floor=0.85,
    )
    return objective, x


def test_lower_bound_objective_gradient_matches_finite_difference() -> None:
    objective, x = _objective()
    theta = np.asarray([0.2, -0.1, 0.15])
    analytical = objective.grad(theta, x)
    numerical = np.empty_like(theta)
    step = 1e-6
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = step
        numerical[index] = (
            objective.value(theta + direction, x)
            - objective.value(theta - direction, x)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-4, atol=2e-5)


def test_acceptance_gradient_matches_finite_difference() -> None:
    objective, x = _objective()
    theta = np.asarray([0.15, 0.05, -0.08])
    analytical = objective.mean_acceptance_grad(theta, x)
    numerical = np.empty_like(theta)
    step = 1e-6
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = step
        numerical[index] = (
            objective.mean_acceptance(theta + direction, x)
            - objective.mean_acceptance(theta - direction, x)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-4, atol=2e-6)


def test_constant_policy_theta_reproduces_requested_action() -> None:
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=-0.1,
        action_high=0.2,
    )
    theta = script._constant_policy_theta(policy, feature_dim=3, initial_u=0.08)
    actions = policy.value(theta, np.zeros((5, 3)))

    np.testing.assert_allclose(actions, 0.08, rtol=0.0, atol=1e-12)


def test_full_population_histogram_uses_all_actions() -> None:
    actions = np.linspace(-0.1, 0.2, 715_023)

    histogram = script._optimized_histogram(actions)

    assert len(histogram) == 30
    assert int(histogram["count"].sum()) == len(actions)
    integrated_density = np.sum(
        histogram["density"] * (histogram["bin_right"] - histogram["bin_left"])
    )
    assert np.isclose(integrated_density, 1.0)
