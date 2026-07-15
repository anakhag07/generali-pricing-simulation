"""Tests for action-space objective regularizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from objective.action_regularizers import (
    ActionRegularizedObjective,
    HeteroskedasticNoiseScaleProvider,
)
from objective.base import Objective, Policy
from objective.noise import HeteroskedasticGaussianNoise, NoisyObjective
from objective.objectives import FixedRegressionObjective
from objective.policy import LinearPolicy, MLPPolicy, mlp_init_theta
from optimization.helpers import finite_difference_theta_grad, objective_grad_on_indices


def _x_batch() -> np.ndarray:
    return np.asarray(
        [
            [0.2, -0.4],
            [1.1, 0.3],
            [-0.7, 0.8],
            [0.5, 1.3],
        ],
        dtype=float,
    )


def _base_objective(policy: Policy) -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.7, 0.4],
        beta_2=-0.8,
        beta_3=[0.5, 0.3],
        beta_4=0.6,
    )


def _penalty_value(objective: ActionRegularizedObjective, theta: np.ndarray, x_batch: np.ndarray) -> float:
    return float(objective.value(theta, x_batch) - objective.base_objective.value(theta, x_batch))


def _penalty_grad(objective: ActionRegularizedObjective, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
    return np.asarray(objective.grad(theta, x_batch), dtype=float) - np.asarray(
        objective.base_objective.grad(theta, x_batch),
        dtype=float,
    )


@pytest.mark.parametrize("policy_kind", ["linear", "mlp"])
def test_action_regularizer_gradient_matches_penalty_finite_difference(policy_kind: str) -> None:
    x_batch = _x_batch()
    if policy_kind == "linear":
        policy = LinearPolicy()
        theta = np.asarray([0.12, -0.27, 0.19], dtype=float)
    else:
        policy = MLPPolicy(hidden=4)
        theta = mlp_init_theta(np.random.default_rng(5), d_in=x_batch.shape[1], hidden=policy.hidden)
    base = _base_objective(policy)
    objective = ActionRegularizedObjective(
        base_objective=base,
        proximal_weight=0.7,
        u_reference=np.asarray([-0.1, 0.05, 0.2, -0.25], dtype=float),
        support_weight=0.3,
        sigma_provider=HeteroskedasticNoiseScaleProvider(
            base_std=0.2,
            growth=0.4,
            u_center=1.25,
        ),
    )

    analytical = _penalty_grad(objective, theta, x_batch)
    numerical = finite_difference_theta_grad(
        lambda theta_eval: _penalty_value(objective, theta_eval, x_batch),
        theta,
        method="central",
        step=1e-6,
    )

    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-6)


def test_disabled_action_regularizers_are_exact_noop() -> None:
    x_batch = _x_batch()
    theta = np.asarray([0.12, -0.27, 0.19], dtype=float)
    base = _base_objective(LinearPolicy())
    objective = ActionRegularizedObjective(base_objective=base)

    assert objective.value(theta, x_batch) == base.value(theta, x_batch)
    np.testing.assert_array_equal(objective.grad(theta, x_batch), base.grad(theta, x_batch))


@dataclass(frozen=True)
class _ZeroClippedObjective(Objective):
    policy: Policy
    u_bounds: tuple[float, float]

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del theta, x_batch
        return 0.0

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del x_batch
        return np.zeros_like(np.asarray(theta, dtype=float), dtype=float)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=float), *self.u_bounds)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self.policy.value(theta, x_batch)

    def policy_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self.policy.grad(theta, x_batch)

    def policy_weighted_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        return self.policy.weighted_grad(theta, x_batch, weights)

    def _value_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        del x_batch
        return np.zeros_like(np.asarray(u_arr, dtype=float), dtype=float)


def test_clipped_samples_contribute_zero_regularizer_gradient() -> None:
    x_batch = np.asarray([[-1.0], [0.2], [1.0]], dtype=float)
    theta = np.asarray([0.0, 1.0], dtype=float)
    base = _ZeroClippedObjective(policy=LinearPolicy(), u_bounds=(-0.5, 0.5))
    objective = ActionRegularizedObjective(
        base_objective=base,
        proximal_weight=2.0,
        u_reference=np.zeros(x_batch.shape[0], dtype=float),
    )

    gradient = objective.grad(theta, x_batch)
    expected_weight = (2.0 * 2.0 / x_batch.shape[0]) * 0.2
    expected = expected_weight * np.asarray([1.0, 0.2], dtype=float)

    np.testing.assert_allclose(gradient, expected, atol=1e-12)


def test_proximal_reference_uses_optimizer_minibatch_indices() -> None:
    x_full = np.asarray(
        [[0.0, 0.0], [1.0, -0.5], [0.25, 0.75], [-0.4, 0.6], [0.9, 0.2]],
        dtype=float,
    )
    theta = np.asarray([0.2, -0.1, 0.3], dtype=float)
    reference = np.asarray([-0.5, -0.1, 0.2, 0.4, 0.8], dtype=float)
    indices = np.asarray([4, 1, 2], dtype=int)
    base = _base_objective(LinearPolicy())
    objective = ActionRegularizedObjective(
        base_objective=base,
        proximal_weight=0.6,
        u_reference=reference,
    )
    direct = ActionRegularizedObjective(
        base_objective=base,
        proximal_weight=0.6,
        u_reference=reference[indices],
    )

    indexed_grad = objective_grad_on_indices(
        objective,
        x_full,
        x_full.shape[0],
        theta,
        indices,
    )
    np.testing.assert_allclose(indexed_grad, direct.grad(theta, x_full[indices]), atol=1e-12)


def test_proximal_reference_length_mismatch_fails_loudly() -> None:
    x_batch = _x_batch()
    theta = np.asarray([0.12, -0.27, 0.19], dtype=float)
    objective = ActionRegularizedObjective(
        base_objective=_base_objective(LinearPolicy()),
        proximal_weight=0.7,
        u_reference=np.asarray([0.0, 0.1], dtype=float),
    )

    with pytest.raises(ValueError, match="u_reference must be aligned"):
        objective.value(theta, x_batch)

    with pytest.raises(ValueError, match="does not cover mini-batch"):
        objective.value_on_indices(theta, x_batch, np.asarray([0, 2, 3], dtype=int))


def test_support_provider_gradient_sign_and_zero_growth() -> None:
    provider = HeteroskedasticNoiseScaleProvider(base_std=0.2, growth=1.5, u_center=0.1)
    values, du_grad = provider.values_and_du_grad(
        np.zeros((3, 1), dtype=float),
        np.asarray([-0.2, 0.1, 0.4], dtype=float),
    )

    np.testing.assert_allclose(values, np.asarray([0.65, 0.2, 0.65], dtype=float))
    np.testing.assert_array_equal(du_grad, np.asarray([-1.5, 0.0, 1.5], dtype=float))

    x_batch = _x_batch()
    theta = np.asarray([0.12, -0.27, 0.19], dtype=float)
    base = _base_objective(LinearPolicy())
    constant_provider = HeteroskedasticNoiseScaleProvider(base_std=0.8, growth=0.0, u_center=-0.3)
    objective = ActionRegularizedObjective(
        base_objective=base,
        support_weight=0.25,
        sigma_provider=constant_provider,
    )

    assert objective.value(theta, x_batch) - base.value(theta, x_batch) == pytest.approx(0.2)
    np.testing.assert_allclose(_penalty_grad(objective, theta, x_batch), 0.0, atol=1e-12)


def test_support_provider_defaults_to_wrapped_noise_scale() -> None:
    base = _base_objective(LinearPolicy())
    noisy = NoisyObjective(
        base,
        HeteroskedasticGaussianNoise(base_std=0.1, growth=0.7, u_center=-0.2, seed=12),
    )
    objective = ActionRegularizedObjective(
        base_objective=noisy,
        support_weight=0.4,
    )

    assert isinstance(objective.sigma_provider, HeteroskedasticNoiseScaleProvider)
    assert objective.sigma_provider.base_std == pytest.approx(0.1)
    assert objective.sigma_provider.growth == pytest.approx(0.7)
    assert objective.sigma_provider.u_center == pytest.approx(-0.2)


def test_action_regularizers_are_added_to_action_value_oracle_with_indices() -> None:
    x_full = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    indices = np.asarray([3, 1], dtype=int)
    u_matrix = np.asarray([[0.3, -0.2], [0.5, 0.1]], dtype=float)
    base = _ZeroClippedObjective(policy=LinearPolicy(), u_bounds=(-10.0, 10.0))
    objective = ActionRegularizedObjective(
        base_objective=base,
        proximal_weight=2.0,
        u_reference=np.asarray([0.0, -0.1, 0.2, 0.4], dtype=float),
        support_weight=0.5,
        sigma_provider=HeteroskedasticNoiseScaleProvider(
            base_std=0.1,
            growth=1.0,
            u_center=0.0,
        ),
    )

    values = objective._value_batch_many_on_indices(x_full[indices], indices, u_matrix)
    expected_reference = np.asarray([0.4, -0.1], dtype=float)
    expected = (
        2.0 * (u_matrix - expected_reference[None, :]) ** 2
        + 0.5 * (0.1 + np.abs(u_matrix))
    )

    np.testing.assert_allclose(values, expected, atol=1e-12)
