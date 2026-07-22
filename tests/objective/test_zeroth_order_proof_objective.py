from __future__ import annotations

import numpy as np
import pytest

from experiments.config import _objective_to_dict
from objective import (
    ArctanRemainderThetaBias,
    ArctanThetaBias,
    LinearThetaBias,
    ThetaBiasModification,
    ThetaBiasedObjective,
    ZerothOrderProofObjective,
    compose_objective,
)


X_DUMMY = np.zeros((1, 1), dtype=float)


def test_proof_objective_formula_gradient_and_known_optimum() -> None:
    objective = ZerothOrderProofObjective()
    x = 0.7
    theta = np.asarray([x])

    assert objective.value(theta, X_DUMMY) == pytest.approx(x * x + 0.5 * (np.sin(x) - x))
    np.testing.assert_allclose(
        objective.grad(theta, X_DUMMY),
        np.asarray([2.0 * x + 0.5 * (np.cos(x) - 1.0)]),
    )
    np.testing.assert_allclose(objective.optimal_theta(), np.zeros(1))
    assert objective.optimal_value() == 0.0


def test_proof_objective_gradient_matches_central_difference() -> None:
    objective = ZerothOrderProofObjective()
    theta = np.asarray([0.3])
    step = 1e-6
    numerical = (
        objective.value(theta + step, X_DUMMY) - objective.value(theta - step, X_DUMMY)
    ) / (2.0 * step)

    assert objective.grad(theta, X_DUMMY)[0] == pytest.approx(numerical, abs=1e-9)


@pytest.mark.parametrize(
    ("bias", "value", "gradient"),
    [
        (LinearThetaBias(0.2), lambda x: 0.2 * x, lambda x: 0.2),
        (ArctanThetaBias(0.2), lambda x: 0.2 * np.arctan(x), lambda x: 0.2 / (1.0 + x * x)),
        (
            ArctanRemainderThetaBias(0.2),
            lambda x: 0.2 * (x - np.arctan(x)),
            lambda x: 0.2 * x * x / (1.0 + x * x),
        ),
    ],
)
def test_theta_bias_formulas(bias, value, gradient) -> None:
    theta = np.asarray([0.4])
    assert bias.value(theta) == pytest.approx(value(theta[0]))
    np.testing.assert_allclose(bias.grad(theta), np.asarray([gradient(theta[0])]))


def test_theta_bias_bounds_match_proof_constants() -> None:
    alpha = 0.2
    nonlinear = ArctanThetaBias(alpha).derivative_bounds()
    remainder = ArctanRemainderThetaBias(alpha).derivative_bounds()
    expected_curvature = 3.0 * np.sqrt(3.0) * alpha / 8.0

    assert nonlinear.beta == pytest.approx(alpha)
    assert nonlinear.kappa_minus == pytest.approx(expected_curvature)
    assert nonlinear.kappa_plus == pytest.approx(expected_curvature)
    assert nonlinear.rho == pytest.approx(2.0 * alpha)
    assert remainder == nonlinear


def test_theta_bias_modification_composes_and_serializes() -> None:
    objective = compose_objective(
        ZerothOrderProofObjective(),
        (
            ThetaBiasModification(
                bias={"type": "ArctanRemainderThetaBias", "alpha": 0.1}
            ),
        ),
    )

    assert isinstance(objective, ThetaBiasedObjective)
    assert isinstance(objective.bias, ArctanRemainderThetaBias)
    assert _objective_to_dict(objective) == {
        "type": "ThetaBiasedObjective",
        "base_objective": ZerothOrderProofObjective().to_dict(),
        "bias": {"type": "ArctanRemainderThetaBias", "alpha": 0.1},
    }


def test_arctan_remainder_is_cubic_near_zero() -> None:
    bias = ArctanRemainderThetaBias(1.0)
    x = 1e-3
    assert bias.value(np.asarray([x])) / x**3 == pytest.approx(1.0 / 3.0, rel=1e-6)
