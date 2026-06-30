"""SciPy trust-constr parity with JAX-backed prepared GLM callbacks."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from objective.objectives.jax_prepared_glm import JaxPreparedGLMObjective  # noqa: E402
from objective.objectives.prepared_glm import PreparedGLMBatch, PreparedGLMObjective  # noqa: E402
from objective.policy import QuadraticFeatureMap, SoftmaxPolicy  # noqa: E402
from optimization.base import Optimization  # noqa: E402
from optimization.gradients.methods import (  # noqa: E402
    FiniteDifferenceGradient,
    GaussSteinGradient,
    SPSAGradient,
    SteinDifferenceGradient,
)
from optimization.solvers import run_first_order_minimize  # noqa: E402


def test_jax_callbacks_match_cpu_prepared_trust_constr_solution() -> None:
    rng = np.random.default_rng(456)
    n_rows = 18
    policy_features = rng.normal(size=(n_rows, 2))
    batch = PreparedGLMBatch.from_arrays(
        base_logit=0.2 + 0.1 * policy_features[:, 0],
        loss=120.0 + 3.0 * policy_features[:, 1],
        premium=np.full(n_rows, 100.0, dtype=float),
        policy_features=policy_features,
        u_coef=-3.0,
    )
    policy = SoftmaxPolicy(
        feature_map=QuadraticFeatureMap(),
        action_low=-0.1,
        action_high=0.2,
    )
    floor = 0.2
    cpu_objective = PreparedGLMObjective(
        policy=policy,
        policy_feature_dim=batch.policy_feature_dim,
        u_coef=batch.u_coef,
        acceptance_floor=floor,
    )
    jax_objective = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
        acceptance_floor=floor,
    )
    theta0 = np.zeros(cpu_objective.policy_theta_dim(), dtype=float)
    jax_objective.warmup(theta0)

    common_kwargs = dict(
        theta_start=theta0,
        x_samples=batch.x_array,
        t_steps=30,
        n_grad_samples=1,
        sigma=0.1,
        perturbation_space="theta",
        algorithm="trust-constr",
        step_size=0.01,
        batch_size=None,
        grad_norm_tol=1e-8,
        initial_constr_penalty=1.0,
        batch_rng=np.random.default_rng(1),
        gradient_rng=np.random.default_rng(2),
    )
    theta_cpu, _ = run_first_order_minimize(objective=cpu_objective, **common_kwargs)
    theta_jax, _ = run_first_order_minimize(objective=jax_objective, **common_kwargs)

    u_cpu = cpu_objective.policy_value(theta_cpu, batch.x_array)
    u_jax = jax_objective.policy_value(theta_jax, batch.x_array)
    np.testing.assert_allclose(u_jax, u_cpu, atol=1e-5, rtol=1e-5)
    assert jax_objective.base_value(theta_jax, batch.x_array) == pytest.approx(
        cpu_objective.base_value(theta_cpu, batch.x_array), rel=1e-6, abs=1e-6
    )
    assert jax_objective.mean_acceptance(theta_jax, batch.x_array) >= floor - 1e-8


def test_jax_zeroth_order_gradients_match_cpu_prepared() -> None:
    rng = np.random.default_rng(789)
    n_rows = 12
    policy_features = rng.normal(size=(n_rows, 2))
    batch = PreparedGLMBatch.from_arrays(
        base_logit=0.1 + 0.2 * policy_features[:, 0],
        loss=110.0 + 2.5 * policy_features[:, 1],
        premium=np.full(n_rows, 90.0, dtype=float),
        policy_features=policy_features,
        u_coef=-2.5,
    )
    policy = SoftmaxPolicy(
        feature_map=QuadraticFeatureMap(),
        action_low=-0.1,
        action_high=0.2,
    )
    cpu_objective = PreparedGLMObjective(
        policy=policy,
        policy_feature_dim=batch.policy_feature_dim,
        u_coef=batch.u_coef,
    )
    jax_objective = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
    )
    theta = np.linspace(
        -0.04,
        0.05,
        policy.theta_dim(batch.policy_feature_dim),
        dtype=float,
    )
    jax_objective.warmup(theta)
    indices = np.arange(n_rows, dtype=int)

    cases = [
        (FiniteDifferenceGradient, "theta"),
        (FiniteDifferenceGradient, "u"),
        (GaussSteinGradient, "theta"),
        (GaussSteinGradient, "u"),
        (SPSAGradient, "theta"),
        (SPSAGradient, "u"),
        (SteinDifferenceGradient, "theta"),
        (SteinDifferenceGradient, "u"),
    ]
    for gradient_cls, perturbation_space in cases:
        cpu_gradient = gradient_cls()
        jax_gradient = gradient_cls()
        common_kwargs = dict(
            algorithm="trust-constr",
            t_steps=3,
            n_grad_samples=4,
            sigma=0.03,
            perturbation_space=perturbation_space,
            step_size=0.01,
            batch_size=None,
        )
        cpu_optimizer = Optimization(
            cpu_objective,
            batch.x_array,
            cpu_gradient,
            **common_kwargs,
            batch_rng=np.random.default_rng(1),
            gradient_rng=np.random.default_rng(2),
        )
        jax_optimizer = Optimization(
            jax_objective,
            batch.x_array,
            jax_gradient,
            **common_kwargs,
            batch_rng=np.random.default_rng(1),
            gradient_rng=np.random.default_rng(2),
        )
        cpu_gradient.setup(cpu_optimizer, theta)
        jax_gradient.setup(jax_optimizer, theta)

        grad_cpu = cpu_gradient.theta_grad(cpu_optimizer, theta, indices)
        grad_jax = jax_gradient.theta_grad(jax_optimizer, theta, indices)

        np.testing.assert_allclose(grad_jax, grad_cpu, rtol=1e-9, atol=1e-9)
