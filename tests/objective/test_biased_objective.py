from __future__ import annotations

import numpy as np
import pytest

from experiments.configs import get_config
from objective import (
    BiasedObjective,
    ConstantPolicy,
    LinearPolicy,
    PlantedLogisticObjective,
    UpperSupportHingeBias,
)


def _constant_base() -> PlantedLogisticObjective:
    return PlantedLogisticObjective.from_parameters(
        policy=ConstantPolicy(),
        alpha=1.0,
        beta=np.asarray([0.3, -0.2], dtype=float),
        bias=0.1,
        u_star=0.0,
    )


def test_biased_objective_wraps_values_and_action_batches() -> None:
    base = _constant_base()
    objective = BiasedObjective(base, lambda_bias=0.2)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=float)
    theta = np.asarray([0.3], dtype=float)
    u = base.policy.value(theta, x)
    u_matrix = np.vstack([u - 0.1, u + 0.1])

    assert objective.value(theta, x) == pytest.approx(base.value(theta, x) - 0.2 * float(np.mean(u)))
    np.testing.assert_allclose(objective._value_batch(x, u), base._value_batch(x, u) - 0.2 * u)
    np.testing.assert_allclose(
        objective._value_batch_many(x, u_matrix),
        np.vstack([base._value_batch(x, row) for row in u_matrix]) - 0.2 * u_matrix,
    )


def test_biased_objective_shifts_action_and_theta_gradients() -> None:
    base = PlantedLogisticObjective.from_parameters(
        policy=LinearPolicy(),
        alpha=1.0,
        beta=np.asarray([0.3, -0.2], dtype=float),
        bias=0.1,
        u_star=0.0,
    )
    objective = BiasedObjective(base, lambda_bias=0.05)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=float)
    theta = np.asarray([0.1, 0.2, -0.1], dtype=float)
    u = base.policy.value(theta, x)

    np.testing.assert_allclose(objective._grad_u_batch(x, u), base._grad_u_batch(x, u) - 0.05)
    expected_bias_grad = base.policy.weighted_grad(theta, x, np.full(x.shape[0], -0.05)) / x.shape[0]
    np.testing.assert_allclose(objective.grad(theta, x), base.grad(theta, x) + expected_bias_grad)


def test_upper_support_hinge_bias_is_zero_inside_support() -> None:
    bias = UpperSupportHingeBias(lambda_bias=0.2, support_center=0.1, support_radius=0.05)
    x = np.zeros((3, 2), dtype=float)
    u = np.asarray([0.1, 0.15, 0.2], dtype=float)

    assert bias.support_upper == pytest.approx(0.15)
    np.testing.assert_allclose(bias.excess(u), np.asarray([0.0, 0.0, 0.05]))
    np.testing.assert_allclose(bias.values(x, u), np.asarray([0.0, 0.0, -0.01]))
    np.testing.assert_allclose(bias.grad_u(x, u), np.asarray([0.0, 0.0, -0.2]))


def test_upper_support_smooth_hinge_has_smooth_gradient() -> None:
    bias = UpperSupportHingeBias(
        lambda_bias=0.2,
        support_center=0.1,
        support_radius=0.05,
        smooth_tau=0.01,
    )
    x = np.zeros((3, 2), dtype=float)
    u = np.asarray([0.1, 0.15, 0.2], dtype=float)

    values = bias.values(x, u)
    grad_u = bias.grad_u(x, u)

    assert values[0] < 0.0  # Smooth hinge has a tiny positive excess below support.
    assert values[2] < values[1] < values[0]
    assert grad_u[0] > -0.2
    assert grad_u[2] < grad_u[1] < grad_u[0]


def test_biased_objective_base_value_reports_true_objective() -> None:
    base = _constant_base()
    objective = BiasedObjective(base, lambda_bias=0.2)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    theta = np.asarray([0.3], dtype=float)

    assert objective.base_value(theta, x) == pytest.approx(base.value(theta, x))
    assert objective.base_value_at_u(x, 0.1) == pytest.approx(base.value_at_u(x, 0.1))
    assert objective.value_at_u(x, 0.1) == pytest.approx(base.value_at_u(x, 0.1) - 0.2 * 0.1)
    assert objective.value(theta, x) != pytest.approx(objective.base_value(theta, x))


def test_biased_objective_support_bias_updates_values_and_gradients() -> None:
    base = _constant_base()
    bias = UpperSupportHingeBias(lambda_bias=0.2, support_center=0.0, support_radius=0.1)
    objective = BiasedObjective(base, bias=bias)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    u = np.asarray([0.05, 0.2], dtype=float)

    np.testing.assert_allclose(
        objective._value_batch(x, u),
        base._value_batch(x, u) + np.asarray([0.0, -0.02]),
    )
    np.testing.assert_allclose(
        objective._grad_u_batch(x, u),
        base._grad_u_batch(x, u) + np.asarray([0.0, -0.2]),
    )


def test_biased_objective_config_serialization_includes_parameters() -> None:
    base_config = get_config("planted_logistic_base")
    config = get_config(
        "planted_logistic_base",
        overrides={
            "objective": BiasedObjective(base_config.objective, lambda_bias=0.05),
            "enabled_estimators": ("first_order",),
            "plot": False,
            "verbose": False,
        },
    )

    payload = config.to_dict()["objective"]

    assert payload["type"] == "BiasedObjective"
    assert payload["lambda_bias"] == 0.05
    assert payload["bias"] == {"type": "LinearActionBias", "lambda_bias": 0.05}
    assert payload["base_objective"]["type"] == "PlantedLogisticObjective"


def test_biased_objective_support_bias_config_serialization_includes_support() -> None:
    base_config = get_config("planted_logistic_base")
    config = get_config(
        "planted_logistic_base",
        overrides={
            "objective": BiasedObjective(
                base_config.objective,
                bias=UpperSupportHingeBias(
                    lambda_bias=0.1,
                    support_center=0.1,
                    support_radius=0.05,
                    smooth_tau=0.01,
                ),
            ),
            "enabled_estimators": ("first_order",),
            "plot": False,
            "verbose": False,
        },
    )

    payload = config.to_dict()["objective"]["bias"]

    assert payload == {
        "type": "UpperSupportHingeBias",
        "lambda_bias": 0.1,
        "support_center": 0.1,
        "support_radius": 0.05,
        "support_upper": 0.15000000000000002,
        "smooth_tau": 0.01,
    }
