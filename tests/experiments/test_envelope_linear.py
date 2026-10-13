import numpy as np
import pytest
from scipy.linalg import hadamard

from experiments.envelopes.core import CalibrationData, FittedEnvelope, TargetDomain
from experiments.envelopes.linear import (
    LinearModel, check_constant_linear_containment, check_linear_containment,
    features, fit_linear_transfer, linear_guarantee,
)


def orthogonal_data():
    X = np.array([[-1.], [-1.], [1.], [1.]])
    actions = np.array([-1., 1., -1., 1.])
    truth = LinearModel([1., 2., -3.])
    errors = np.array([.5, .5, -.5, -.5])
    return CalibrationData(X, actions, truth(X, actions) + errors), truth


def represented_envelope(center, beta, covariance=None, scale_squared=1.):
    center = LinearModel(center)
    beta = np.asarray(beta, dtype=float)
    covariance = np.eye(len(beta)) if covariance is None else np.asarray(covariance)
    auxiliary = LinearModel(beta)

    def radius(X, actions):
        phi = features(X, actions)
        return np.abs(center(X, actions) - auxiliary(X, actions)) + np.sqrt(
            scale_squared * np.einsum("ij,jk,ik->i", phi, covariance, phi)
        )

    return FittedEnvelope(
        center, radius,
        state={"beta": beta, "covariance": covariance, "scale_squared": scale_squared},
        metadata={"construction": "linear_known_variance_transfer"},
    )


def test_qr_fit_width_and_leverage_match_independent_orthogonal_design():
    data, _ = orthogonal_data()
    settings = {"delta": .2, "sigma_bound": 2.}
    fit = fit_linear_transfer(data, LinearModel([0., 0., 0.]), TargetDomain(data.X), settings, None)
    # For these four rows P'P=4I, and the error vector subtracts .5 from beta_x.
    np.testing.assert_allclose(fit.state["beta"], [1., 1.5, -3.], atol=1e-14)
    np.testing.assert_allclose(fit.state["covariance"], np.eye(3)/4, atol=1e-15)
    evaluated = fit.evaluate(data.X, data.actions)
    np.testing.assert_allclose(evaluated.components["width"]**2, 45.)
    assert np.mean(evaluated.components["width"]**2) == pytest.approx(4*3**2/(.2*4))
    phi = features(data.X, data.actions)
    leverage = np.einsum("ij,jk,ik->i", phi, fit.state["covariance"], phi)
    assert leverage.sum() == pytest.approx(3.)
    query_X, query_a = np.array([[2.], [-4.]]), np.array([7., -.25])
    np.testing.assert_allclose(
        fit.evaluate(query_X, query_a).components["width"],
        np.sqrt(15 * (1 + query_X[:, 0]**2 + query_a**2)),
    )


def test_transfer_slack_bounds_hold_when_auxiliary_band_contains_truth():
    data, truth = orthogonal_data()
    predictor = LinearModel([-3., .8, 11.])
    target = TargetDomain(data.X)
    fit = fit_linear_transfer(data, predictor, target, {"delta": .05, "sigma_bound": 1.}, None)
    decision = check_linear_containment(fit, truth, target)
    assert decision.status == "covered"
    assert decision.details["auxiliary_covered"]
    X, actions = np.array([[-2.], [0.], [3.], [4.]]), np.array([-100., -.2, 0., 17.])
    evaluated = fit.evaluate(X, actions)
    slack = evaluated.radius - np.abs(evaluated.center-truth(X, actions))
    assert np.all(slack >= -1e-12)
    assert np.all(slack <= 2*evaluated.components["width"] + 1e-12)


def test_three_context_dimensions_use_five_coefficients_without_context_union_bound():
    design = hadamard(8)[:, :5].astype(float)
    X, actions = design[:, 1:4], design[:, 4]
    truth = LinearModel([.1, .2, .3, .4, .5])
    data = CalibrationData(X, actions, truth(X, actions))
    settings = {"delta": .1, "sigma_bound": 2.}
    fit = fit_linear_transfer(data, truth, TargetDomain(X), settings, None)
    np.testing.assert_allclose(fit.state["covariance"], np.eye(5)/8, atol=1e-15)
    np.testing.assert_allclose(fit.evaluate(X, actions).components["width"]**2, 4*25/(.1*8))
    guarantee = linear_guarantee(data, truth, truth, TargetDomain(X), settings, 2.)
    assert guarantee.value == pytest.approx(.9)


def test_continuum_checker_detects_failure_outside_display_points():
    fit = represented_envelope([0., 0., 2.], [0., 0., 2.])
    truth = LinearModel([0., 0., 0.])
    X, actions = np.zeros((3, 1)), np.array([-.2, 0., .2])
    evaluated = fit.evaluate(X, actions)
    assert np.all(np.abs(evaluated.center-truth(X, actions)) <= evaluated.radius)
    result = check_linear_containment(fit, truth, TargetDomain([[0.]]))
    assert result.status == "violated"
    assert result.details["auxiliary_covered"] is False


def test_exact_equality_is_covered_and_auxiliary_failure_is_not_final_failure():
    truth = LinearModel([0., 0., 0.])
    target = TargetDomain([[0.]])
    touching = represented_envelope([1., 0., 0.], [1., 0., 0.])
    assert check_linear_containment(touching, truth, target).status == "covered"
    transferred = represented_envelope([0., 0., 0.], [2., -4., 8.], scale_squared=0.)
    decision = check_linear_containment(transferred, truth, target)
    assert decision.status == "covered"
    assert decision.details["auxiliary_covered"] is False


def test_context_contraction_uses_exact_represented_values_before_polynomial_signs():
    # Floating evaluation loses the +1. The represented affine formula does not.
    fit = represented_envelope([1., 1., 0.], [1., 1., 0.], scale_squared=0.)
    truth = LinearModel([1e16, 0., 0.])
    X, actions = np.array([[1e16]]), np.array([0.])
    np.testing.assert_array_equal(fit.center(X, actions), truth(X, actions))
    result = check_linear_containment(fit, truth, TargetDomain(X))
    assert result.status == "violated"
    assert "does not enclose" in result.details["arithmetic"]


def test_known_variance_guarantee_is_a_lower_bound_with_explicit_assumptions():
    data, truth = orthogonal_data()
    predictor = lambda X, a: np.sin(a) + X[:, 0]**2
    bound = linear_guarantee(data, truth, predictor, TargetDomain(data.X),
                             {"delta": .05, "sigma_bound": 1.}, noise_std=.7)
    assert bound.kind == "lower_bound"
    assert bound.value == pytest.approx(.95)
    assert any("known" in assumption for assumption in bound.assumptions)
    assert "No union bound" in bound.derivation


@pytest.mark.parametrize("settings,noise_std,message", [
    ({"delta": .05, "sigma_bound": .3}, .7, "noise_std"),
    ({"delta": .05, "sigma_bound": 1., "width_multiplier": .5}, .7, "below one"),
    ({"delta": 1., "sigma_bound": 1.}, .7, "delta"),
    ({"delta": .05, "sigma_bound": 0.}, 0., "positive"),
])
def test_invalid_analytical_claims_are_rejected(settings, noise_std, message):
    data, truth = orthogonal_data()
    with pytest.raises(ValueError, match=message):
        linear_guarantee(data, truth, truth, TargetDomain(data.X), settings, noise_std)


def test_rank_deficiency_and_incompatible_model_or_target_are_rejected():
    data, truth = orthogonal_data()
    target = TargetDomain(data.X)
    settings = {"delta": .05, "sigma_bound": 1.}
    rank_deficient = CalibrationData(data.X, data.X[:, 0], data.y)
    with pytest.raises(ValueError, match="full column rank"):
        fit_linear_transfer(rank_deficient, truth, target, settings, None)
    with pytest.raises(ValueError, match="dimension-matched"):
        linear_guarantee(data, lambda X, a: np.sin(a), truth, target, settings, .7)
    fit = fit_linear_transfer(data, truth, target, settings, None)
    with pytest.raises(ValueError, match="real_line"):
        check_linear_containment(fit, truth, TargetDomain(data.X, np.array([0., 1.])))
    with pytest.raises(ValueError, match="LinearModel"):
        check_linear_containment(fit, lambda X, a: np.sin(a), target)
    with pytest.raises(ValueError, match="dimensions"):
        check_linear_containment(fit, LinearModel([0., 1., 2., 3.]), target)


def test_multiplier_scales_width_only_and_repeated_queries_are_deterministic():
    data, truth = orthogonal_data()
    target = TargetDomain(data.X)
    settings = {"delta": .05, "sigma_bound": 1.}
    original = fit_linear_transfer(data, truth, target, settings, None)
    reduced = fit_linear_transfer(data, truth, target, dict(settings, width_multiplier=.25), None)
    first, second = original.evaluate(data.X, data.actions), reduced.evaluate(data.X, data.actions)
    np.testing.assert_allclose(first.components["discrepancy"], second.components["discrepancy"])
    np.testing.assert_allclose(first.components["width"]*.25, second.components["width"])
    np.testing.assert_array_equal(second.radius, reduced.evaluate(data.X, data.actions).radius)


def constant_envelope(coefficients, radius):
    return FittedEnvelope(
        LinearModel(coefficients), lambda X, a: np.full(len(a), radius),
        state={"radius": radius}, metadata={"construction": "constant_radius"},
    )


def test_literal_constant_detects_arbitrarily_distant_tail_failure():
    fit = constant_envelope([0., 0., np.nextafter(1., 2.)], 3.)
    truth = LinearModel([0., 0., 1.])
    X, actions = np.zeros((3, 1)), np.array([-2., 0., 2.])
    evaluated = fit.evaluate(X, actions)
    assert np.all(np.abs(evaluated.center-truth(X, actions)) <= evaluated.radius)
    result = check_constant_linear_containment(fit, truth, TargetDomain([[-10.], [0.], [10.]]))
    assert result.status == "violated"
    assert result.details["slope_mismatch"]
    assert "tails" in result.details["reason"]
    assert result.details["context_count"] == 3
    assert result.details["contexts"] == [
        {"context_index": index, "covered": False} for index in range(3)
    ]


@pytest.mark.parametrize("contexts,expected", [
    ([[-1.], [0.]], "covered"),
    ([[-1.], [1.]], "covered"),  # Equality at x=1 is included.
    ([[-1.], [3.]], "violated"),
])
def test_literal_constant_with_equal_slopes_checks_contextual_error(contexts, expected):
    fit = constant_envelope([.5, .5, 2.], 1.)
    truth = LinearModel([0., 0., 2.])
    result = check_constant_linear_containment(fit, truth, TargetDomain(contexts))
    assert result.status == expected
    assert not result.details["slope_mismatch"]
    assert result.details["contexts"][0]["absolute_error_exact"] == "0"


def test_literal_constant_checker_rejects_other_construction_or_finite_target():
    fit = constant_envelope([0., 0., 1.], 3.)
    truth = LinearModel([0., 0., 1.])
    with pytest.raises(ValueError, match="real_line"):
        check_constant_linear_containment(fit, truth, TargetDomain([[0.]], [0., 1.]))
    with pytest.raises(ValueError, match="literal constant_radius"):
        check_constant_linear_containment(
            represented_envelope([0., 0., 1.], [0., 0., 1.]), truth, TargetDomain([[0.]]),
        )
