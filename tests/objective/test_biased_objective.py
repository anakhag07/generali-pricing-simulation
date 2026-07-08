from __future__ import annotations

import numpy as np
import pytest

from experiments.configs import get_config
from objective import BiasedObjective, ConstantPolicy, LinearPolicy, PlantedLogisticObjective


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


def test_biased_objective_base_value_reports_true_objective() -> None:
    base = _constant_base()
    objective = BiasedObjective(base, lambda_bias=0.2)
    x = np.asarray([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    theta = np.asarray([0.3], dtype=float)

    assert objective.base_value(theta, x) == pytest.approx(base.value(theta, x))
    assert objective.base_value_at_u(x, 0.1) == pytest.approx(base.value_at_u(x, 0.1))
    assert objective.value_at_u(x, 0.1) == pytest.approx(base.value_at_u(x, 0.1) - 0.2 * 0.1)
    assert objective.value(theta, x) != pytest.approx(objective.base_value(theta, x))


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
    assert payload["base_objective"]["type"] == "PlantedLogisticObjective"
