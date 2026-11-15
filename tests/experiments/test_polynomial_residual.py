"""Independent identities for the fixed quartic residual study."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import roots_hermitenorm
import sympy as sp

from experiments.envelopes.feature_wild import WildFeatureDesign
from experiments.envelopes.polynomial import (
    AdditivePolynomial, PolynomialFeatureBand, polynomial_features, fit_polynomial_band,
    coverage_events, polynomial_containment, residual_metrics, integrated_width,
)
from experiments.envelopes.polynomial_study import prepare_environment, PolynomialLCBObjective
from experiments.envelopes.wild_study import load_manifest, build_launch_plan
from experiments.envelopes.quadratic import VARIABLE
from optimization.gradients.methods import SteinDifferenceGradient

ROOT = Path(__file__).resolve().parents[2]
TRUTH = AdditivePolynomial((1., -.5, .25), (0., .5, -.5))
RESIDUAL = AdditivePolynomial((.1, 0., 0.), (.2, .25, .15, -.08, .04))


@pytest.fixture
def sample():
    rng = np.random.default_rng(291)
    x = rng.normal(size=(80, 3))
    a = .5+np.sqrt(.45)*rng.normal(size=80)
    signs = rng.choice([-1, 1], size=(31, len(a)))
    return x, a, signs


def test_polynomial_band_rejects_empty_contexts():
    with pytest.raises(ValueError, match='nonempty context matrix'):
        PolynomialFeatureBand(np.empty((0, 3)), 1, np.zeros(5), np.eye(5), 1.)


def test_interaction_indices_round_trip_numpy_integers(tmp_path):
    path = tmp_path/'indices.npz'
    np.savez(path, indices=np.array([[0, 1], [1, 2]], dtype=np.int64))
    with np.load(path, allow_pickle=False) as saved:
        interactions = tuple((j, power, value) for (j, power), value in
                             zip(saved['indices'], [.35, .2]))
    model = AdditivePolynomial((.1, 0., 0.), (.2, .25, .15), interactions)
    assert all(type(j) is type(power) is int for j, power, _ in model.interactions)
    x, a = np.array([[2., 3., 4.]]), np.array([.5])
    expected = .1*2+.2+.25*.5+.15*.5**2+.35*2*.5+.2*3*.5**2
    np.testing.assert_allclose(model(x, a), [expected])
    np.testing.assert_allclose(polynomial_features(x, a, 2, 2)@model.feature_coefficients(2, 2), [expected])
    assert float(model.exact(x[0]).eval(.5)) == pytest.approx(expected)


@pytest.mark.parametrize('index,power', [(0., 1), (0, 1.), (True, 1),
    (0, np.bool_(True)), (-1, 1), (3, 1), (0, 0), (0, 3), (np.nan, 1)])
def test_interaction_indices_reject_nonintegers_and_invalid_ranges(index, power):
    with pytest.raises(ValueError, match='Invalid context/action interaction'):
        AdditivePolynomial((0., 0., 0.), (0., 0., 0.), ((index, power, 1.),))


@pytest.mark.parametrize('degree', [4, 5, 6])
def test_noiseless_recovery_and_bootstrap_refits(sample, degree):
    x, a, signs = sample
    y = RESIDUAL(x, a)
    band, state = fit_polynomial_band(x, a, y, degree, signs, .05)
    metrics = residual_metrics(band, RESIDUAL, .5, .45)
    assert metrics['rmse'] < 1e-11
    assert state['critical'] < 1e-11
    matrix = polynomial_features(x, a, degree)
    observed = y+.3*np.sin(np.arange(len(y)))
    noisy, saved = fit_polynomial_band(x, a, observed, degree, signs, .05)
    errors = observed-matrix@noisy.beta
    for i in range(3):
        refit = np.linalg.lstsq(matrix, matrix@noisy.beta+errors*signs[i], rcond=None)[0]
        np.testing.assert_allclose(refit-noisy.beta, saved['bootstrap_errors'][i], atol=1e-11)


def test_offgrid_asymmetric_values_and_gaussian_moments(sample):
    x, a, signs = sample
    band, _ = fit_polynomial_band(x, a, RESIDUAL(x, a)+.3*np.cos(np.arange(len(a))), 6, signs, .05)
    actions = np.linspace(-1.7, 2.7, len(x))
    phi = polynomial_features(x, actions, 6)
    fitted, width, lower, upper = band.evaluate(actions)
    np.testing.assert_allclose(fitted, phi@band.beta, atol=1e-11)
    np.testing.assert_allclose(width, band.critical*np.linalg.norm(phi@band.inverse, axis=1), atol=1e-10)
    np.testing.assert_allclose(lower, np.maximum(width-fitted, 0))
    np.testing.assert_allclose(upper, np.maximum(width+fitted, 0))
    assert np.any(abs(lower-upper) > .01)
    z, weights = roots_hermitenorm(14)
    weights /= np.sqrt(2*np.pi)
    actual = RESIDUAL(x, (.5+np.sqrt(.45)*z)[:, None])
    predicted = band.evaluate((.5+np.sqrt(.45)*z)[:, None])[0]
    error = np.sum(weights*np.mean((actual-predicted)**2, axis=1))
    mean = np.sum(weights*np.mean(actual, axis=1))
    variance = np.sum(weights*np.mean((actual-mean)**2, axis=1))
    metrics = residual_metrics(band, RESIDUAL, .5, .45)
    assert metrics['mse'] == pytest.approx(error, rel=1e-10)
    assert metrics['r_squared'] == pytest.approx(1-error/variance, rel=1e-10)


def test_whole_line_certificates_distinguish_transfer_auxiliary_and_tails(sample):
    a = VARIABLE
    poly = lambda x: sp.Poly(x, a, domain=sp.QQ)
    assert polynomial_containment(poly(0), poly(a**2), poly(0)) == (False, True)
    assert polynomial_containment(poly(a**4), poly(0), poly(1+a**2)) == (False, False)
    assert polynomial_containment(poly(1), poly(0), poly(4*a**2)) == (False, False)
    assert polynomial_containment(poly(a**2), poly(a**2+1), poly(1)) == (True, True)
    x, actions, signs = sample
    for k in [1, 2, 3]:
        band, _ = fit_polynomial_band(x, actions, RESIDUAL(x, actions), k, signs, .05)
        assert not coverage_events(band, RESIDUAL)['transferred']
    design = WildFeatureDesign(polynomial_features(x, actions, 4))
    exact = PolynomialFeatureBand(x, 4, RESIDUAL.feature_coefficients(4), np.linalg.inv(design.triangular), .1)
    assert coverage_events(exact, RESIDUAL)['coefficient']


def test_width_integration_keeps_full_width_and_sides_separate(sample):
    x, a, _ = sample
    design = WildFeatureDesign(polynomial_features(x, a, 1))
    band = PolynomialFeatureBand(x, 1, np.array([2., 0., 0., 0., 0.]), np.linalg.inv(design.triangular), 0.)
    widths = integrated_width(band, .5, .45)
    assert widths['mean_full_width'] == pytest.approx(2.)
    assert widths['mean_lower_radius'] == 0
    assert widths['mean_upper_radius'] == pytest.approx(2.)
    assert widths['mean_residual_half_width'] == 0


def test_gp_covariance_and_mode_dispatch_preserve_existing_family():
    manifest = load_manifest(ROOT/'manifests/synthetic-polynomial-residual-degree4.json')
    p = json.loads(json.dumps(manifest.payload))
    p['n'] = 20
    first, second = prepare_environment(p), prepare_environment(p)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    x, a = first['X'], first['actions']
    expected = np.exp(-np.sum((x[:, None]-x[None, :])**2, axis=2)/2-(a[:, None]-a[None, :])**2/(2*.45))
    np.testing.assert_allclose(first['gp_covariance'], expected, atol=1e-15)
    assert np.all((first['sigma'] > .15) & (first['sigma'] < 1.))
    plan = build_launch_plan(manifest)
    assert plan.name == manifest.name
    assert plan.task_count == 100


def test_native_stein_identity_policy_has_correct_sign_and_mean_scaling(sample):
    x, a, signs = sample
    band, _ = fit_polynomial_band(x, a, RESIDUAL(x, a), 4, signs, .05)
    objective = PolynomialLCBObjective(band, TRUTH, RESIDUAL)
    theta = np.linspace(-.3, 1.2, len(x))
    sigma, m, seed = .01, 7, 719
    optimizer = SimpleNamespace(objective=objective, x_array=x, n_total=len(x),
                                perturbation_space='u', sigma=sigma, n_grad_samples=m,
                                rng=np.random.default_rng(seed))
    actual = SteinDifferenceGradient().theta_grad(optimizer, theta, np.arange(len(x)))
    directions = np.random.default_rng(seed).normal(size=m)
    plus = objective._value_batch_many(x, theta+sigma*directions[:, None])
    minus = objective._value_batch_many(x, theta-sigma*directions[:, None])
    expected = np.mean((plus-minus)*directions[:, None]/(2*sigma), axis=0)/len(x)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    with pytest.raises(AssertionError):
        objective.grad(theta, x)
