from __future__ import annotations

import numpy as np
import pytest

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from objective.modifications import (
    AcceptanceLagrangianModification,
    AcceptanceLagrangianObjective,
    AcceptancePenaltyModification,
    AcceptancePenaltyObjective,
    BiasModification,
    BiasedObjective,
    ConstantThetaRegularizer,
    HomoskedasticGaussianNoise,
    IntervalDistanceThetaRegularizer,
    LinearActionBias,
    NoiseModification,
    NoisyObjective,
    ProximalThetaRegularizer,
    RegularizationModification,
    RegularizedObjective,
    SmoothSaturatingIntervalThetaRegularizer,
    SupportThetaRegularizer,
    UpperSupportHingeBias,
    compose_objective,
)
from objective.noise import NoisyObjective as LegacyNoisyObjective
from objective.objectives import StronglyConvexQuadratic
from objective.objectives.biased import BiasedObjective as LegacyBiasedObjective


class _AcceptanceObjective:
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        return float(np.sum(theta))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del x_batch
        return np.ones_like(theta, dtype=float)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        return float(0.2 + 0.1 * theta[0])

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del x_batch
        return np.asarray([0.1, 0.0], dtype=float)


def test_legacy_noise_and_bias_imports_reexport_modification_classes() -> None:
    assert LegacyNoisyObjective is NoisyObjective
    assert LegacyBiasedObjective is BiasedObjective


def test_compose_objective_applies_modifications_in_explicit_order() -> None:
    base = get_config("planted_logistic_base").objective
    u_star = float(base.optimal_u())

    composed = compose_objective(
        base,
        (
            BiasModification(
                bias=UpperSupportHingeBias(
                    lambda_bias=0.1,
                    support_center=u_star,
                    support_radius=0.05,
                )
            ),
            RegularizationModification(
                regularizers=(ProximalThetaRegularizer(weight=0.2),)
            ),
            NoiseModification(noise=HomoskedasticGaussianNoise(std=0.1, seed=7)),
        ),
    )

    assert isinstance(composed, NoisyObjective)
    assert isinstance(composed.base_objective, RegularizedObjective)
    assert isinstance(composed.base_objective.base_objective, BiasedObjective)


def test_compose_objective_accepts_mapping_specs() -> None:
    base = get_config("planted_logistic_base").objective

    composed = compose_objective(
        base,
        (
            {
                "type": "bias",
                "bias": {"type": "LinearActionBias", "lambda_bias": 0.1},
            },
            {
                "type": "regularization",
                "regularizers": [{"type": "proximal", "weight": 0.2}],
            },
            {
                "type": "noise",
                "noise": {"type": "HomoskedasticGaussianNoise", "std": 0.1, "seed": 7},
            },
        ),
    )

    assert isinstance(composed, NoisyObjective)
    assert isinstance(composed.base_objective, RegularizedObjective)
    biased = composed.base_objective.base_objective
    assert isinstance(biased, BiasedObjective)
    assert isinstance(biased.bias, LinearActionBias)
    assert biased.lambda_bias == pytest.approx(0.1)


def test_regularized_objective_adds_theta_value_and_gradient_terms() -> None:
    base = StronglyConvexQuadratic.isotropic(2)
    objective = RegularizedObjective(
        base,
        regularizers=(
            ProximalThetaRegularizer(weight=3.0, reference=np.asarray([1.0, -1.0])),
            SupportThetaRegularizer(weight=0.5, support_center=0.0, support_growth=2.0),
        ),
    )
    theta = np.asarray([2.0, -3.0], dtype=float)
    x_batch = np.zeros((1, 1), dtype=float)

    assert objective.value(theta, x_batch) == pytest.approx(16.5)
    np.testing.assert_allclose(objective.grad(theta, x_batch), np.asarray([5.5, -9.5]))


def test_interval_envelope_regularizers_match_values_and_gradients() -> None:
    theta = np.asarray([0.0, 1.0, 1.5], dtype=float)

    constant = ConstantThetaRegularizer(height=0.4)
    linear = IntervalDistanceThetaRegularizer(slope=2.0, lower=0.75, upper=1.25)
    smooth = SmoothSaturatingIntervalThetaRegularizer(
        amplitude=0.6,
        transition_width=0.25,
        lower=0.75,
        upper=1.25,
    )

    assert constant.value(theta) == pytest.approx(0.4)
    np.testing.assert_allclose(constant.grad(theta), np.zeros(3))
    assert linear.value(theta) == pytest.approx(2.0 * np.mean([0.75, 0.0, 0.25]))
    np.testing.assert_allclose(linear.grad(theta), np.asarray([-2.0, 0.0, 2.0]) / 3.0)

    expected_terms = 0.6 * np.asarray(
        [np.exp(-((0.25 / 0.75) ** 2)), 0.0, np.exp(-1.0)]
    )
    assert smooth.value(theta) == pytest.approx(float(np.mean(expected_terms)))
    assert smooth.grad(theta)[0] < 0.0
    assert smooth.grad(theta)[1] == 0.0
    assert smooth.grad(theta)[2] > 0.0


def test_smooth_interval_envelope_is_flat_at_bounds_and_matches_numdiff() -> None:
    regularizer = SmoothSaturatingIntervalThetaRegularizer(
        amplitude=0.7,
        transition_width=0.25,
        lower=0.75,
        upper=1.25,
    )
    for boundary in (0.75, 1.25):
        theta = np.asarray([boundary], dtype=float)
        assert regularizer.value(theta) == 0.0
        np.testing.assert_allclose(regularizer.grad(theta), np.zeros(1))

    for value in (0.1, 0.5, 1.5, 2.0):
        theta = np.asarray([value], dtype=float)
        step = 1e-6
        numerical = (
            regularizer.value(theta + step) - regularizer.value(theta - step)
        ) / (2.0 * step)
        assert regularizer.grad(theta)[0] == pytest.approx(numerical, abs=1e-8)


def test_smooth_interval_envelope_is_bounded_monotone_and_nonconvex() -> None:
    regularizer = SmoothSaturatingIntervalThetaRegularizer(
        amplitude=0.8,
        transition_width=0.25,
        lower=0.75,
        upper=1.25,
    )
    distance = np.linspace(0.0, 3.0, 500)
    values = np.asarray(
        [regularizer.value(np.asarray([0.75 - value])) for value in distance]
    )

    assert values[0] == 0.0
    assert np.all(np.diff(values) >= 0.0)
    assert np.all(values <= 0.8)
    curvature = np.diff(values, n=2)
    assert np.any(curvature > 0.0)
    assert np.any(curvature < 0.0)


@pytest.mark.parametrize(
    "regularizer",
    [
        ConstantThetaRegularizer(height=0.4),
        IntervalDistanceThetaRegularizer(slope=1.2, lower=0.75, upper=1.25),
        SmoothSaturatingIntervalThetaRegularizer(
            amplitude=0.6,
            transition_width=0.25,
            lower=0.75,
            upper=1.25,
        ),
    ],
)
def test_interval_envelope_regularizers_round_trip(regularizer) -> None:
    from objective.modifications import regularizer_from_dict

    assert regularizer_from_dict(regularizer.to_dict()) == regularizer


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "constant", "height": -0.1},
        {"type": "interval_distance", "slope": 1.0, "lower": 1.0, "upper": 1.0},
        {
            "type": "smooth_saturating_interval",
            "amplitude": 0.5,
            "transition_width": 0.0,
            "lower": 0.75,
            "upper": 1.25,
        },
    ],
)
def test_interval_envelope_regularizers_reject_invalid_parameters(payload) -> None:
    from objective.modifications import regularizer_from_dict

    with pytest.raises(ValueError):
        regularizer_from_dict(payload)


def test_regularized_objective_serializes_and_forwards_noise_seed() -> None:
    base = StronglyConvexQuadratic.isotropic(2)
    noisy = NoisyObjective(base, HomoskedasticGaussianNoise(std=0.25))
    regularized = RegularizedObjective(
        noisy,
        regularizers=(ProximalThetaRegularizer(weight=0.2),),
    )

    seeded = regularized.with_noise_seed(123)

    assert isinstance(seeded.base_objective, NoisyObjective)
    assert seeded.base_objective.noise.seed == 123
    assert seeded.to_dict() == {
        "type": "RegularizedObjective",
        "regularizers": [
            {"type": "ProximalThetaRegularizer", "weight": 0.2, "reference": None}
        ],
    }


def test_acceptance_modifications_add_scalar_value_and_gradient_terms() -> None:
    base = _AcceptanceObjective()
    theta = np.asarray([1.0, 2.0], dtype=float)
    x_batch = np.zeros((1, 1), dtype=float)

    lagrangian = compose_objective(
        base,
        (AcceptanceLagrangianModification(acceptance_floor=0.5, lagrangian_lambda=2.0),),
    )
    assert isinstance(lagrangian, AcceptanceLagrangianObjective)
    assert lagrangian.value(theta, x_batch) == pytest.approx(3.0 + 2.0 * (0.5 - 0.3))
    np.testing.assert_allclose(lagrangian.grad(theta, x_batch), np.asarray([0.8, 1.0]))

    penalty = compose_objective(
        base,
        (
            AcceptancePenaltyModification(
                acceptance_floor=0.5,
                acceptance_penalty_weight=3.0,
                acceptance_penalty_temperature=0.1,
            ),
        ),
    )
    assert isinstance(penalty, AcceptancePenaltyObjective)
    soft_gap = 0.1 * float(np.logaddexp(0.0, (0.5 - 0.3) / 0.1))
    sigmoid_gap = 1.0 / (1.0 + np.exp(-((0.5 - 0.3) / 0.1)))
    expected_grad_scale = -2.0 * 3.0 * soft_gap * sigmoid_gap
    assert penalty.value(theta, x_batch) == pytest.approx(3.0 + 3.0 * soft_gap * soft_gap)
    np.testing.assert_allclose(
        penalty.grad(theta, x_batch),
        np.asarray([1.0 + expected_grad_scale * 0.1, 1.0]),
    )


def test_config_summary_records_modification_specs_and_composed_objective() -> None:
    config = get_config(
        "synthetic_quadratic_base",
        overrides={
            "objective_modifications": (
                RegularizationModification(
                    regularizers=(ProximalThetaRegularizer(weight=0.2),)
                ),
                NoiseModification(noise=HomoskedasticGaussianNoise(std=0.1, seed=5)),
            ),
            "enabled_estimators": ("finite_difference",),
            "correctness": CorrectnessSpec(gradient_source="none"),
            "plot": False,
            "verbose": False,
        },
    )

    assert isinstance(config.objective, NoisyObjective)
    assert isinstance(config.objective.base_objective, RegularizedObjective)

    payload = config.to_dict()
    assert payload["objective_modifications"] == [
        {
            "type": "RegularizationModification",
            "regularizers": [
                {"type": "ProximalThetaRegularizer", "weight": 0.2, "reference": None}
            ],
        },
        {
            "type": "NoiseModification",
            "noise": {"type": "HomoskedasticGaussianNoise", "std": 0.1, "seed": 5},
        },
    ]
    assert payload["objective"]["type"] == "NoisyObjective"
    assert payload["objective"]["base_objective"]["type"] == "RegularizedObjective"
