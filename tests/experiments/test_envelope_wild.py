"""Independent OLS and exact-containment checks for thesis §1.3."""

import numpy as np
import pytest
from scipy.linalg import hadamard

from experiments.envelopes.core import TargetDomain
from experiments.envelopes.linear import LinearModel, check_linear_containment, features
from experiments.envelopes.wild import WildLinearDesign


def orthogonal_design():
    matrix = hadamard(8)[:, :3].astype(float)
    return WildLinearDesign(matrix[:, 1:2], matrix[:, 2])


def test_raw_residual_draws_equal_independent_ols_refits_and_keep_prefixes():
    rng = np.random.default_rng(1301)
    design = WildLinearDesign(rng.normal(size=(31, 2)), rng.normal(size=31))
    y = design.design @ np.array([1., -.5, .25, 2.]) + rng.normal(size=31)
    signs = rng.choice([-1, 1], size=(9, 31))
    beta, errors = design.bootstrap_errors(y, signs)
    expected = np.linalg.lstsq(design.design, y, rcond=None)[0]
    np.testing.assert_allclose(beta, expected, atol=1e-14)
    residuals = y - design.design @ expected
    for index in range(len(signs)):
        response = design.design @ expected + residuals * signs[index]
        refitted = np.linalg.lstsq(design.design, response, rcond=None)[0]
        np.testing.assert_allclose(errors[index], refitted - expected, atol=2e-14)
    small_beta, small_errors = design.bootstrap_errors(y, signs[:4])
    np.testing.assert_array_equal(beta, small_beta)
    np.testing.assert_allclose(errors[:4], small_errors, rtol=0, atol=2e-16)
    np.testing.assert_allclose(design.leverage.sum(), 4., atol=1e-14)


def test_all_real_statistic_matches_orthogonal_formula_including_tail_limit():
    design = orthogonal_design()
    errors = np.array([[0., 0., 1.], [1., 2., 3.], [-3., 1., -.2], [0., 0., 0.]])
    expected = np.sqrt(4 * (np.abs(errors[:, 0]) + np.abs(errors[:, 1]))**2 + 8 * errors[:, 2]**2)
    np.testing.assert_allclose(design.statistics(errors), expected, atol=1e-13)
    # A pure slope reaches its supremum only in the action tails, so a
    # bounded rendering grid cannot calibrate the same statistic.
    actions = np.array([-2., 0., 2.])
    phi = features(np.ones((3, 1)), actions)
    scales = np.sqrt(np.einsum("ij,jk,ik->i", phi, design.covariance, phi))
    assert np.max(np.abs(actions) / scales) < design.statistics(errors[:1])[0]


def test_statistic_bounds_every_fiber_and_is_equivariant_under_response_scaling():
    rng = np.random.default_rng(1302)
    design = WildLinearDesign(rng.normal(size=(17, 2)), rng.normal(size=17))
    errors = rng.normal(size=(11, 4))
    statistics = design.statistics(errors)
    actions = np.linspace(-30, 30, 301)
    X = np.repeat(design.X, len(actions), axis=0)
    phi = features(X, np.tile(actions, len(design.X)))
    scale = np.sqrt(np.einsum("ij,jk,ik->i", phi, design.covariance, phi))
    evaluated = np.abs(errors @ phi.T) / scale
    assert np.all(evaluated <= statistics[:, None] + 1e-12)
    np.testing.assert_allclose(design.statistics(3.5 * errors), 3.5 * statistics, atol=1e-12)


def test_envelope_state_replays_geometric_width_and_transfer_components():
    design = orthogonal_design()
    predictor = LinearModel([.1, -.5, 1.])
    beta, critical = np.array([1., .25, 2.]), 2.1
    envelope = design.envelope(predictor, beta, critical)
    observed = envelope.evaluate(design.X, design.design[:, -1])
    np.testing.assert_allclose(observed.components["width"], critical * np.sqrt(3 / 8))
    np.testing.assert_allclose(observed.components["auxiliary"], design.design @ beta)
    np.testing.assert_allclose(observed.radius, observed.components["width"] + observed.components["discrepancy"])
    assert envelope.state["scale_squared"] == critical**2
    assert envelope.metadata["residuals"] == "raw"
    np.testing.assert_array_equal(observed.radius, envelope.evaluate(design.X, design.design[:, -1]).radius)


@pytest.mark.parametrize("seed", range(12))
def test_fast_transferred_containment_matches_existing_exact_oracle(seed):
    rng = np.random.default_rng(1500 + seed)
    design = WildLinearDesign(rng.normal(size=(7, 1)), rng.normal(size=7))
    truth, predictor, beta = rng.normal(size=(3, 3))
    critical = [.2, 1., 3., 8.][seed % 4]
    envelope = design.envelope(LinearModel(predictor), beta, critical)
    expected = check_linear_containment(envelope, LinearModel(truth), TargetDomain(design.X))
    assert design.transferred_contains(beta, predictor, truth, critical) == (expected.status == "covered")


def test_transfer_tail_and_equality_cases_use_full_real_domain():
    design = orthogonal_design()
    zero, slope = np.zeros(3), np.array([0., 0., 1.])
    assert design.transferred_contains(zero, zero, zero, 0.)
    # Predictor error is no larger than its discrepancy, though auxiliary
    # truth containment fails; transfer must not just copy the auxiliary test.
    assert design.transferred_contains(slope, zero, -.5 * slope, 0.)
    assert not design.transferred_contains(zero, zero, slope, 1.)
    critical = np.sqrt(8.)
    for value in (critical, np.nextafter(critical, 0.), np.nextafter(critical, np.inf)):
        expected = check_linear_containment(
            design.envelope(LinearModel(zero), zero, value), LinearModel(slope), TargetDomain(design.X),
        )
        assert design.transferred_contains(zero, zero, slope, value) == (expected.status == "covered")
    # Exact equality with a discrepancy-only radius.
    assert design.transferred_contains(slope, zero, -slope, 0.)


def test_transfer_checks_interior_deficit_even_when_tails_are_covered():
    design = orthogonal_design()
    predictor = np.zeros(3)
    truth, auxiliary = np.array([-1., 0., 0.]), np.array([0., 0., 1.])
    # |predictor-truth|-|predictor-auxiliary|=1-|a| is positive
    # around zero and negative in both tails.
    assert not design.transferred_contains(auxiliary, predictor, truth, 1.)
    assert design.transferred_contains(auxiliary, predictor, truth, 3.)


def test_invalid_design_draws_and_queries_are_rejected():
    with pytest.raises(ValueError, match="full column rank"):
        WildLinearDesign(np.zeros((8, 1)), np.arange(8.))
    design = orthogonal_design()
    with pytest.raises(ValueError, match="responses"):
        design.fit([1., 2.])
    with pytest.raises(ValueError, match="Multipliers"):
        design.bootstrap_errors(np.ones(8), np.zeros((2, 8)))
    with pytest.raises(ValueError, match="Multipliers"):
        design.bootstrap_errors(np.ones(8), np.full((2, 8), 1j))
    with pytest.raises(ValueError, match="Coefficient errors"):
        design.statistics(np.zeros((2, 8)))
    with pytest.raises(ValueError, match="Critical"):
        design.envelope(LinearModel([0., 0., 0.]), np.zeros(3), -1.)
    with pytest.raises(ValueError, match="Critical"):
        design.envelope(LinearModel([0., 0., 0.]), np.zeros(3), 1e300)


@pytest.mark.parametrize('moved', [3., 10., 20., 40., 100., 1000.])
def test_quadratic_residual_bootstrap_agrees_with_translated_response_fit(moved):
    rng = np.random.default_rng(1301)
    actions = np.clip(rng.normal(size=300), -3., 3.)
    actions[0] = moved
    design = WildLinearDesign(np.empty((300, 0)), actions, action_degree=2)
    predictor_coeff = np.array([.10142434006958467, 4.824476863339917, -5.127996631955374])
    predictor = design.design @ predictor_coeff
    y = design.design @ [0., 5., -5.] + .5*rng.normal(size=300)
    signs = rng.choice([-1, 1], size=(10, 300))
    direct, direct_errors = design.bootstrap_errors(y, signs)
    residual, residual_errors = design.bootstrap_errors(y, signs, predictor=predictor)
    np.testing.assert_allclose(residual, direct-predictor_coeff, atol=2e-9)
    np.testing.assert_allclose(residual_errors, direct_errors, atol=2e-9)
    # Independent refits check that wild draws use fitted noise errors, not R itself.
    observed = y-predictor
    errors = observed-design.design@residual
    refit = np.linalg.lstsq(design.design, design.design@residual+errors*signs[0], rcond=None)[0]
    np.testing.assert_allclose(residual_errors[0], refit-residual, atol=2e-9)


@pytest.mark.parametrize('degree', [1, 2])
def test_signed_transfer_is_distinct_from_residual_band_and_symmetric_transfer(degree):
    from experiments.envelopes.wild import residual_radii
    design = WildLinearDesign(np.empty((9, 0)), np.linspace(-2, 2, 9), action_degree=degree)
    for sign in (-1., 1.):
        truth = np.r_[sign, np.zeros(degree)]
        fitted = 2*truth
        assert not design.auxiliary_contains(fitted, truth, 0.)
        assert design.residual_contains(fitted, truth, 0.)
        assert not design.residual_contains(-fitted, truth, 0.)
        assert design.residual_contains(fitted, np.zeros(degree+1), 0.)
    lower, upper = residual_radii(np.array([-2., 0., 2.]), .5)
    np.testing.assert_array_equal(lower, [2.5, .5, 0.])
    np.testing.assert_array_equal(upper, [0., .5, 2.5])


def test_quadratic_certificates_include_action_tails_and_bracket_suprema():
    design = WildLinearDesign(np.empty((9, 0)), np.linspace(-2, 2, 9), action_degree=2)
    errors = np.array([[0., 0., 1.], [.3, -.2, .1]])
    bounds = design.statistic_bounds(errors)
    assert np.all(bounds[:, 1]-bounds[:, 0] <= 2e-8*np.maximum(1., bounds[:, 1]))
    for error, (lo, hi) in zip(errors, bounds):
        assert design.auxiliary_contains(error, np.zeros(3), hi)
        assert not design.auxiliary_contains(error, np.zeros(3), max(0., lo-1e-7))
    assert not design.residual_contains(np.zeros(3), errors[0], 1e-5)
    assert design.residual_contains(errors[0], errors[0], 0.)


def test_affine_signed_fast_check_matches_exact_polynomial_certificate():
    import sympy as sp
    from experiments.envelopes.wild import _residual_polynomials_contain
    from experiments.bootstrap_band_continuous import VARIABLE
    rng = np.random.default_rng(1301)
    design = WildLinearDesign(rng.normal(size=(12, 1)), rng.normal(size=12))
    for _ in range(5):
        beta, truth = rng.normal(size=(2, 3))
        critical = float(rng.uniform(.1, 5.))
        expected = True
        for x in design.X:
            basis = sp.Matrix([1, sp.Rational(float(x[0])), VARIABLE])
            poly = lambda v: sp.Poly((basis.T*sp.Matrix(v).applyfunc(lambda c: sp.Rational(float(c))))[0], VARIABLE)
            q = sp.Poly((basis.T*sp.Matrix(design.covariance).applyfunc(lambda c: sp.Rational(float(c)))*basis)[0], VARIABLE)
            expected &= _residual_polynomials_contain(poly(truth), poly(beta), q, critical)
        assert design.residual_contains(beta, truth, critical) == expected
