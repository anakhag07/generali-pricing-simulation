from __future__ import annotations

import numpy as np
import pytest

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.configs import get_config
from experiments.correctness import resolve_true_grad_theta_fn
from objective import FixedRegressionObjective, SoftmaxPolicy
from objective.base import default_rng, sample_states
from objective.modifications import ProximalThetaRegularizer, RegularizedObjective
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective
from objective.objectives import BiasedObjective, UpperSupportHingeBias


class DummyThetaObjectiveNoGrad:
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        return float(np.sum(theta**2))


def _build_theta_objective() -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=SoftmaxPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.3],
        beta_4=0.5,
    )


def test_correctness_exact_requires_theta_grad() -> None:
    with pytest.raises(ValueError, match="objective must implement grad"):
        ExperimentConfig(
            state_dim=1,
            n_samples=1,
            step_rule="constant",
        perturbation_space="theta",
            objective=DummyThetaObjectiveNoGrad(),
            theta0=np.zeros(2, dtype=float),
            correctness=CorrectnessSpec(gradient_source="exact"),
        )


def test_resolve_true_grad_numdiff_matches_exact() -> None:
    objective = _build_theta_objective()
    correctness = CorrectnessSpec(
        gradient_source="numdiff",
        numdiff_method="central",
        numdiff_step=1e-5,
    )
    true_grad_fn = resolve_true_grad_theta_fn(objective, correctness)
    assert true_grad_fn is not None

    theta = np.zeros(2, dtype=float)
    x_batch = np.asarray([[0.8]], dtype=float)
    grad_exact = objective.grad(theta, x_batch)
    grad_numdiff = true_grad_fn(theta, x_batch)
    assert np.allclose(grad_numdiff, grad_exact, rtol=1e-4, atol=1e-4)


def test_resolve_true_grad_none_returns_none() -> None:
    objective = _build_theta_objective()
    correctness = CorrectnessSpec(gradient_source="none")
    assert resolve_true_grad_theta_fn(objective, correctness) is None


def test_correctness_accepts_denoised_exact_source() -> None:
    correctness = CorrectnessSpec(gradient_source="denoised_exact")

    assert correctness.gradient_source == "denoised_exact"


def test_correctness_accepts_noise_free_exact_source() -> None:
    correctness = CorrectnessSpec(gradient_source="noise_free_exact")

    assert correctness.gradient_source == "noise_free_exact"


def test_denoised_exact_unwraps_full_wrapper_chain_to_clean_objective() -> None:
    # Layering noise on top of support bias nests two deterministic wrappers:
    # NoisyObjective(BiasedObjective(planted)). denoised_exact must reference the
    # innermost clean (unbiased, noise-free) objective -- the first-order truth --
    # not the biased surrogate one level down.
    planted = get_config("planted_logistic_base").objective
    u_star = float(planted.optimal_u())
    biased = BiasedObjective(
        base_objective=planted,
        bias=UpperSupportHingeBias(lambda_bias=0.1, support_center=u_star, support_radius=0.05),
    )
    noisy = NoisyObjective(base_objective=biased, noise=HomoskedasticGaussianNoise(std=0.5, seed=0))

    true_grad_fn = resolve_true_grad_theta_fn(noisy, CorrectnessSpec(gradient_source="denoised_exact"))
    assert true_grad_fn is not None

    # Large intercept pushes every action above the support band so the bias
    # gradient is active, making the surrogate gradient differ from the truth.
    theta = np.array([3.0, 0.0, 0.0, 0.0], dtype=float)
    x_batch = sample_states(default_rng(0), 6, 3)

    grad_true = true_grad_fn(theta, x_batch)
    assert np.allclose(grad_true, planted.grad(theta, x_batch))
    assert not np.allclose(grad_true, biased.grad(theta, x_batch))


def test_noise_free_exact_removes_noise_and_preserves_bias_wrapper() -> None:
    planted = get_config("planted_logistic_base").objective
    u_star = float(planted.optimal_u())
    biased = BiasedObjective(
        base_objective=planted,
        bias=UpperSupportHingeBias(lambda_bias=0.1, support_center=u_star, support_radius=0.05),
    )
    noisy = NoisyObjective(base_objective=biased, noise=HomoskedasticGaussianNoise(std=0.5, seed=0))

    true_grad_fn = resolve_true_grad_theta_fn(noisy, CorrectnessSpec(gradient_source="noise_free_exact"))
    assert true_grad_fn is not None

    theta = np.array([3.0, 0.0, 0.0, 0.0], dtype=float)
    x_batch = sample_states(default_rng(0), 6, 3)

    grad_true = true_grad_fn(theta, x_batch)
    assert np.allclose(grad_true, biased.grad(theta, x_batch))
    assert not np.allclose(grad_true, planted.grad(theta, x_batch))


def test_noise_free_exact_removes_noise_and_preserves_regularization_wrapper() -> None:
    planted = get_config("planted_logistic_base").objective
    regularized = RegularizedObjective(
        planted,
        regularizers=(ProximalThetaRegularizer(weight=0.2),),
    )
    noisy = NoisyObjective(
        base_objective=regularized,
        noise=HomoskedasticGaussianNoise(std=0.5, seed=0),
    )

    true_grad_fn = resolve_true_grad_theta_fn(noisy, CorrectnessSpec(gradient_source="noise_free_exact"))
    assert true_grad_fn is not None

    theta = np.array([3.0, 0.0, 0.0, 0.0], dtype=float)
    x_batch = sample_states(default_rng(0), 6, 3)

    grad_true = true_grad_fn(theta, x_batch)
    assert np.allclose(grad_true, regularized.grad(theta, x_batch))
    assert not np.allclose(grad_true, planted.grad(theta, x_batch))


def test_numdiff_batch_is_supported_for_theta_grad() -> None:
    objective = _build_theta_objective()
    config = ExperimentConfig(
        state_dim=1,
        n_samples=1,
        step_rule="constant",
        perturbation_space="theta",
        objective=objective,
        theta0=np.zeros(2, dtype=float),
        correctness=CorrectnessSpec(
            gradient_source="numdiff",
            numdiff_aggregate="batch",
        ),
    )
    assert config.correctness.numdiff_aggregate == "batch"
