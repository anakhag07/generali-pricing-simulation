"""Ensure policy optimization uses individual saved bands and exact derivatives."""

import numpy as np

from experiments.envelopes.wild import WildLinearDesign
from objective.policy import IdentityFeatureMap, SoftmaxPolicy
from scratch.generali_wild_policy import _FrozenFibers, _Responses, _policy_band, _AcceptanceBandObjective


def _fixture():
    z = np.array([[-2., .4], [-1., -.2], [0., .7], [.5, -.8], [1., .1], [2., .9]])
    a = np.array([-.08, .01, .04, .09, .12, .17])
    design = WildLinearDesign(z, a)
    beta = np.array([-.2, 1.2, -.4, -.8])
    fibers = _FrozenFibers(z, design._triangular, beta, .003)
    arrays = {"row_indices": np.arange(len(z)), "premium": np.linspace(100., 160., len(z)),
              "glm_loss": np.linspace(40., 80., len(z))}
    response = _Responses("glm", arrays, glm_logit=np.linspace(1., 2., len(z)), glm_slope=-3.)
    return z, design, beta, fibers, response


def test_cached_fibers_match_native_band_values_and_action_derivatives():
    z, design, beta, fibers, _ = _fixture()
    a = np.linspace(-.071, .183, len(z))
    actual = fibers.evaluate(slice(None), a)
    expected = design.residual_components(z, a, beta, .003)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13)
    h = 1e-6
    plus = fibers.evaluate(slice(None), a+h)["radius_lower"]
    minus = fibers.evaluate(slice(None), a-h)["radius_lower"]
    np.testing.assert_allclose(actual["lower_adjustment_derivative"], -(plus-minus)/(2*h), atol=1e-9)


def test_policy_objective_averages_individual_lower_endpoints_and_chain_rule():
    z, _, _, fibers, response = _fixture()
    policy = SoftmaxPolicy(feature_map=IdentityFeatureMap(), action_low=-.1, action_high=.2)
    theta = np.array([.1, -.2, .3])
    band = _policy_band(response, fibers, np.arange(len(z)), z, [-.1, .2])
    objective = _AcceptanceBandObjective(band, policy, endpoint="lower", sense="maximize",
                                         artifact={}, acceptance_floor=.7)
    actions = policy.value(theta, z)
    p, b = response.evaluate(slice(None), actions), fibers.evaluate(slice(None), actions)
    np.testing.assert_allclose(objective.value(theta, z), -np.mean(p["center"]-b["radius_lower"]))
    # Averaging before the positive part gives a different objective here.
    wrong = -np.mean(p["center"])+max(np.mean(b["width"])-np.mean(b["residual_fit"]), 0.)
    assert abs(objective.value(theta, z)-wrong) > .01
    h = 1e-6
    finite = np.array([(objective.value(theta+e*h, z)-objective.value(theta-e*h, z))/(2*h)
                       for e in np.eye(len(theta))])
    np.testing.assert_allclose(objective.grad(theta, z), finite, atol=2e-8)
    acceptance_finite = np.array([(objective.mean_acceptance(theta+e*h, z)-objective.mean_acceptance(theta-e*h, z))/(2*h)
                                  for e in np.eye(len(theta))])
    np.testing.assert_allclose(objective.mean_acceptance_grad(theta, z), acceptance_finite, atol=2e-10)


def test_saturated_softmax_endpoint_roundoff_is_normalized():
    z, _, _, fibers, response = _fixture()
    policy = SoftmaxPolicy(feature_map=IdentityFeatureMap(), action_low=-.1, action_high=.2)
    band = _policy_band(response, fibers, np.arange(len(z)), z, [-.1, .2])
    theta = np.array([1000., 0., 0.])
    actions = policy.value(theta, z)
    evaluated = band.evaluate(z, actions)
    expected = response.evaluate(slice(None), np.full(len(z), .2))["center"]
    np.testing.assert_array_equal(evaluated.center, expected)
