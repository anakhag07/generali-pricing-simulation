"""Optax step-rule loop: convergence, determinism, and parity checks."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optax")

from objective.objectives.generali.jax_prepared_glm import JaxPreparedGLMObjective  # noqa: E402
from objective.objectives.synthetic.planted_logistic import PlantedLogisticObjective  # noqa: E402
from objective.objectives.generali.prepared_glm import PreparedGLMBatch  # noqa: E402
from objective.policy import ConstantPolicy, QuadraticFeatureMap, SoftmaxPolicy  # noqa: E402
from optimization import FiniteDifferenceGradient, FirstOrderGradient, Optimization  # noqa: E402
from optimization.optax_loop import optax_step_rule_optimizer  # noqa: E402


def _planted_objective() -> PlantedLogisticObjective:
    return PlantedLogisticObjective.from_parameters(
        policy=ConstantPolicy(),
        alpha=2.0,
        beta=[0.3, 0.2, 0.1],
        bias=-0.5,
        u_star=1.1,
    )


def _x_samples(n: int = 24, dim: int = 3, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim))


def _optimizer(objective, x_samples, gradient, algorithm: str, **kwargs) -> Optimization:
    defaults = dict(
        t_steps=400,
        n_grad_samples=4,
        sigma=0.1,
        step_size=0.05,
        batch_rng=np.random.default_rng(1),
        gradient_rng=np.random.default_rng(2),
    )
    defaults.update(kwargs)
    return Optimization(objective, x_samples, gradient, algorithm=algorithm, **defaults)


def test_optax_step_rule_optimizer_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="Unsupported optax step rule"):
        optax_step_rule_optimizer("l-bfgs-b", 0.01)


def test_optax_adam_converges_on_planted_logistic() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    optimizer = _optimizer(objective, x_samples, FirstOrderGradient(), "optax-adam")
    theta_final, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert abs(float(theta_final[0]) - objective.optimal_u()) < 0.05
    assert trace.objective_values[-1] <= trace.objective_values[0]


def test_optax_sgd_matches_constant_step_rule() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    theta0 = np.asarray([0.2], dtype=float)

    theta_sgd, trace_sgd = _optimizer(
        objective, x_samples, FirstOrderGradient(), "optax-sgd", t_steps=50
    ).solve(theta0)
    theta_const, trace_const = _optimizer(
        objective, x_samples, FirstOrderGradient(), "constant", t_steps=50
    ).solve(theta0)

    np.testing.assert_allclose(theta_sgd, theta_const, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(trace_sgd.theta_values), np.asarray(trace_const.theta_values), atol=1e-10
    )


def test_optax_adam_deterministic_across_runs() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    theta0 = np.asarray([0.3], dtype=float)

    results = []
    for _ in range(2):
        optimizer = _optimizer(objective, x_samples, FirstOrderGradient(), "optax-adam", t_steps=40)
        results.append(optimizer.solve(theta0))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(
        np.asarray(results[0][1].theta_values), np.asarray(results[1][1].theta_values)
    )


def test_optax_grad_norm_tol_stops_at_optimum() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    optimizer = _optimizer(
        objective,
        x_samples,
        FirstOrderGradient(),
        "optax-adam",
        grad_norm_tol=1e-8,
    )
    theta_final, trace = optimizer.solve(np.asarray([objective.optimal_u()], dtype=float))

    assert trace.optimizer_success is True
    assert trace.optimizer_status == 0
    assert "gradient norm below tolerance" in trace.optimizer_message
    np.testing.assert_allclose(theta_final, [objective.optimal_u()], atol=1e-12)


def test_optax_records_step_sizes() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    optimizer = _optimizer(objective, x_samples, FirstOrderGradient(), "optax-adam", t_steps=10)
    _, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert trace.step_sizes is not None
    # Initial record has no step size; subsequent records log the learning rate.
    assert np.isnan(trace.step_sizes[0])
    assert all(size == pytest.approx(0.05) for size in trace.step_sizes[1:])


def test_optax_adam_supports_value_only_gradients() -> None:
    objective = _planted_objective()
    x_samples = _x_samples()
    optimizer = _optimizer(
        objective, x_samples, FiniteDifferenceGradient(), "optax-adam", t_steps=150
    )
    theta_final, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert trace.objective_values[-1] < trace.objective_values[0]
    assert abs(float(theta_final[0]) - objective.optimal_u()) < 0.2


def test_optax_adam_minimizes_jax_prepared_glm_with_penalty() -> None:
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
    objective = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
        acceptance_floor=0.2,
        acceptance_penalty_weight=10.0,
    )
    theta0 = np.zeros(objective.policy_theta_dim(), dtype=float)
    objective.warmup(theta0)

    optimizer = _optimizer(
        objective, batch.x_array, FirstOrderGradient(), "optax-adam", t_steps=60
    )
    theta_final, trace = optimizer.solve(theta0)

    assert trace.objective_values[-1] < trace.objective_values[0]
    assert theta_final.shape == theta0.shape
    assert objective.mean_acceptance(theta_final, batch.x_array) >= 0.2 - 0.05
