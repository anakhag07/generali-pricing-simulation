"""JAX prepared GLM MLP-policy parity with the NumPy prepared GLM objective."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from objective.objectives.jax_prepared_glm import JaxPreparedGLMObjective  # noqa: E402
from objective.objectives.prepared_glm import PreparedGLMBatch, PreparedGLMObjective  # noqa: E402
from objective.policy import MLPPolicy, mlp_init_theta  # noqa: E402


def _batch(rng: np.random.Generator, n_rows: int, d_in: int) -> PreparedGLMBatch:
    policy_features = rng.normal(size=(n_rows, d_in))
    return PreparedGLMBatch.from_arrays(
        base_logit=0.2 + 0.1 * policy_features[:, 0],
        loss=120.0 + 3.0 * policy_features[:, 1],
        premium=np.full(n_rows, 100.0, dtype=float),
        policy_features=policy_features,
        u_coef=-3.0,
    )


def test_jax_mlp_matches_numpy_prepared_glm() -> None:
    rng = np.random.default_rng(0)
    d_in, hidden = 4, 8
    batch = _batch(rng, 20, d_in)
    policy = MLPPolicy(hidden=hidden)
    floor = 0.2

    cpu = PreparedGLMObjective(
        policy=policy,
        policy_feature_dim=batch.policy_feature_dim,
        u_coef=batch.u_coef,
        acceptance_floor=floor,
    )
    jax_obj = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
        acceptance_floor=floor,
    )
    theta = mlp_init_theta(np.random.default_rng(1), d_in=d_in, hidden=hidden)
    assert jax_obj.policy_theta_dim() == policy.theta_dim(d_in) == theta.size
    jax_obj.warmup(theta)

    # Policy actions and objective value match the NumPy MLP forward.
    np.testing.assert_allclose(
        jax_obj.policy_value(theta, batch.x_array),
        cpu.policy_value(theta, batch.x_array),
        atol=1e-9,
    )
    assert jax_obj.value(theta, batch.x_array) == pytest.approx(
        cpu.value(theta, batch.x_array), rel=1e-9, abs=1e-9
    )
    assert jax_obj.mean_acceptance(theta, batch.x_array) == pytest.approx(
        cpu.mean_acceptance(theta, batch.x_array), rel=1e-9, abs=1e-9
    )


def test_jax_mlp_grad_matches_finite_difference() -> None:
    rng = np.random.default_rng(2)
    d_in, hidden = 3, 6
    batch = _batch(rng, 16, d_in)
    policy = MLPPolicy(hidden=hidden)
    jax_obj = JaxPreparedGLMObjective(policy=policy, x_array=batch.x_array, u_coef=batch.u_coef)
    theta = mlp_init_theta(np.random.default_rng(3), d_in=d_in, hidden=hidden)
    jax_obj.warmup(theta)

    analytic = jax_obj.grad(theta, batch.x_array)
    step = 1e-6
    fd = np.zeros_like(theta)
    for i in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += step
        minus[i] -= step
        fd[i] = (jax_obj.value(plus, batch.x_array) - jax_obj.value(minus, batch.x_array)) / (2 * step)
    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-4)


def test_jax_mlp_policy_grad_not_supported() -> None:
    rng = np.random.default_rng(4)
    d_in, hidden = 3, 4
    batch = _batch(rng, 8, d_in)
    policy = MLPPolicy(hidden=hidden)
    jax_obj = JaxPreparedGLMObjective(policy=policy, x_array=batch.x_array, u_coef=batch.u_coef)
    theta = mlp_init_theta(np.random.default_rng(5), d_in=d_in, hidden=hidden)
    with pytest.raises(NotImplementedError):
        jax_obj.policy_grad(theta, batch.x_array)
