from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.correctness import resolve_true_grad_theta_fn
from experiments.run import run_experiment
from experiments.seeding import SeedSetup
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective, NoNoise
from objective.objectives import PlantedLogisticObjective
from objective.policy import ConstantPolicy


def _base_objective() -> PlantedLogisticObjective:
    return PlantedLogisticObjective.from_parameters(
        policy=ConstantPolicy(),
        alpha=1.0,
        beta=np.asarray([0.3, -0.2], dtype=float),
        bias=0.1,
        u_star=0.0,
    )


def test_homoskedastic_gaussian_noise_is_keyed_by_x_and_u() -> None:
    noise = HomoskedasticGaussianNoise(std=0.5, seed=123)
    x = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    u = np.asarray([0.1, -0.2], dtype=float)

    first = noise.values(x, u)
    second = noise.values(x, u)
    changed_u = noise.values(x, u + np.asarray([0.0, 0.01]))

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, changed_u)


def test_homoskedastic_gaussian_noise_is_batch_order_independent() -> None:
    noise = HomoskedasticGaussianNoise(seed=9)
    x = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["a", "b", "c"],
        }
    )
    u = np.asarray([0.1, 0.2, 0.3], dtype=float)
    order = np.asarray([2, 0, 1], dtype=int)

    values = noise.values(x, u)
    permuted = noise.values(x.iloc[order].reset_index(drop=True), u[order])

    np.testing.assert_allclose(permuted, values[order])


def test_noisy_objective_wraps_value_and_action_batches() -> None:
    base = _base_objective()
    noise = HomoskedasticGaussianNoise(std=0.25, seed=11)
    objective = NoisyObjective(base, noise)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=float)
    theta = np.asarray([0.2], dtype=float)
    u = base.policy.value(theta, x)

    expected_value = base.value(theta, x) + float(np.mean(noise.values(x, u)))
    expected_batch = base._value_batch(x, u) + noise.values(x, u)
    u_matrix = np.vstack([u - 0.1, u + 0.1])
    expected_many = np.vstack([base._value_batch(x, row) for row in u_matrix]) + noise.values(x, u_matrix)

    assert objective.value(theta, x) == pytest.approx(expected_value)
    np.testing.assert_allclose(objective._value_batch(x, u), expected_batch)
    np.testing.assert_allclose(objective._value_batch_many(x, u_matrix), expected_many)


def test_noisy_objective_grad_raises_but_base_grad_remains_available() -> None:
    base = _base_objective()
    objective = NoisyObjective(base, HomoskedasticGaussianNoise(seed=3))
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    theta = np.asarray([0.2], dtype=float)

    with pytest.raises(NotImplementedError, match="no analytical gradient"):
        objective.grad(theta, x)

    assert base.grad(theta, x).shape == theta.shape


def test_no_noise_returns_zero_with_matching_shape() -> None:
    noise = NoNoise()
    x = np.asarray([[1.0], [2.0]], dtype=float)

    np.testing.assert_allclose(noise.values(x, np.asarray([0.1, 0.2])), np.zeros(2))
    np.testing.assert_allclose(noise.values(x, np.asarray([[0.1, 0.2]])), np.zeros((1, 2)))


def test_run_experiment_applies_noise_seed_stream() -> None:
    base_config = get_config("planted_logistic_base")
    noisy_objective = NoisyObjective(
        base_config.objective,
        HomoskedasticGaussianNoise(std=0.1),
    )
    config = get_config(
        "planted_logistic_base",
        overrides={
            "objective": noisy_objective,
            "enabled_estimators": ("finite_difference",),
            "correctness": CorrectnessSpec(gradient_source="none"),
            "n_samples": 5,
            "t_steps": 1,
            "plot": False,
            "verbose": False,
            "seed_setup": SeedSetup(run_seed=1, data_seed=2, split_seed=3, noise_seed=99, optimizer_seed=4),
        },
    )

    result = run_experiment(config)

    objective = result.config.objective
    assert isinstance(objective, NoisyObjective)
    assert isinstance(objective.noise, HomoskedasticGaussianNoise)
    assert objective.noise.seed == 99


def test_denoised_exact_correctness_uses_base_objective_gradient() -> None:
    base = _base_objective()
    objective = NoisyObjective(base, HomoskedasticGaussianNoise(seed=3))
    correctness = CorrectnessSpec(gradient_source="denoised_exact")
    true_grad_fn = resolve_true_grad_theta_fn(objective, correctness)
    assert true_grad_fn is not None
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    theta = np.asarray([0.2], dtype=float)

    np.testing.assert_allclose(true_grad_fn(theta, x), base.grad(theta, x))


def test_exact_correctness_still_uses_noisy_objective_gradient() -> None:
    base = _base_objective()
    objective = NoisyObjective(base, HomoskedasticGaussianNoise(seed=3))
    correctness = CorrectnessSpec(gradient_source="exact")
    true_grad_fn = resolve_true_grad_theta_fn(objective, correctness)
    assert true_grad_fn is not None
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    theta = np.asarray([0.2], dtype=float)

    with pytest.raises(NotImplementedError, match="no analytical gradient"):
        true_grad_fn(theta, x)


def test_run_experiment_supports_denoised_exact_correctness_for_noisy_objective() -> None:
    base_config = get_config("planted_logistic_base")
    config = get_config(
        "planted_logistic_base",
        overrides={
            "objective": NoisyObjective(
                base_config.objective,
                HomoskedasticGaussianNoise(std=0.1),
            ),
            "enabled_estimators": ("finite_difference",),
            "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
            "n_samples": 5,
            "t_steps": 1,
            "plot": False,
            "verbose": False,
            "seed_setup": SeedSetup(run_seed=1, data_seed=2, split_seed=3, noise_seed=99, optimizer_seed=4),
        },
    )

    result = run_experiment(config)

    trace = result.traces["finite_difference"]
    assert trace.true_theta_grad_norms is not None
    assert len(trace.true_theta_grad_norms) > 0
