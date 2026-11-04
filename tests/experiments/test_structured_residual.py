"""Independent interaction identities and frozen-network calibration checks."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import roots_hermitenorm

from experiments.envelopes.feature_wild import WildFeatureDesign
from experiments.envelopes.polynomial import (AdditivePolynomial, polynomial_features,
    fit_polynomial_band, coefficient_contains, residual_metrics)
from experiments.envelopes.mlp_head import (FrozenResidualMLP, FrozenMLPBand, SiLUResidualLoss,
    fit_residual_mlp, calibrate_frozen_head)
from experiments.envelopes.structured_residual import (explicit_model, grid_coverage, coverage_grid,
    evaluation_rows, restore_structured_band, fit_structured_models, numerical_residual_metrics)
from experiments.envelopes.wild_study import load_manifest, build_launch_plan
from experiments.envelopes.polynomial_study import prepare_environment

ROOT = Path(__file__).resolve().parents[2]


def test_interaction_embedding_recovery_and_moments():
    rng = np.random.default_rng(42)
    x, a = rng.normal(size=(100, 3)), rng.normal(.5, np.sqrt(.45), 100)
    g = AdditivePolynomial((.1, 0, 0), (.2, .25, .15, -.08, .04), ((0, 1, .35), (1, 2, .2)))
    signs = rng.choice([-1, 1], (51, 100))
    for q in [0, 1, 2]:
        matrix = polynomial_features(x, a, 6, q)
        assert matrix.shape == (100, 10+3*q)
        if q < 2:
            with pytest.raises(ValueError):
                g.feature_coefficients(6, q)
        band, _ = fit_polynomial_band(x, a, g(x, a), 6, signs, .05, interaction_order=q)
        if q < 2:
            assert not coefficient_contains(band, g)
        else:
            np.testing.assert_allclose(matrix@g.feature_coefficients(6, q), g(x, a), atol=1e-13)
            assert residual_metrics(band, g, .5, .45)['rmse'] < 1e-11
        np.testing.assert_allclose(band.evaluate(a)[0], matrix@band.beta, atol=1e-11)
        z, w = roots_hermitenorm(16)
        actions = (.5+np.sqrt(.45)*z)[:, None]
        expected = np.sum(w*np.mean((g(x, actions)-band.evaluate(actions)[0])**2, axis=1))/np.sqrt(2*np.pi)
        assert residual_metrics(band, g, .5, .45)['mse'] == pytest.approx(expected, abs=1e-12)
        fit, squared = band.exact_fiber(x[0])
        for action in [-2., .2, 3.]:
            values = band.evaluate(action)
            assert float(fit.eval(action)) == pytest.approx(values[0][0], abs=1e-10)
            assert float(squared.eval(action)) == pytest.approx(values[1][0]**2, abs=1e-10)


def test_frozen_head_bootstrap_matches_refits_without_observed_response_fit():
    rng = np.random.default_rng(918)
    matrix = np.column_stack((np.ones(80), rng.normal(size=(80, 5))))
    beta = rng.normal(size=6)
    original = beta.copy()
    response = matrix@beta+2+.4*rng.normal(size=80)
    signs = rng.choice([-1, 1], (31, 80))
    with patch.object(WildFeatureDesign, 'fit', side_effect=AssertionError('Calibration must not fit observed targets')):
        state = calibrate_frozen_head(matrix, beta, response, signs, .05)
    np.testing.assert_array_equal(state['beta'], original)
    np.testing.assert_array_equal(beta, original)
    assert not np.allclose(state['beta'], np.linalg.lstsq(matrix, response, rcond=None)[0])
    for b in range(5):
        pseudo = matrix@beta+signs[b]*(response-matrix@beta)
        refitted = np.linalg.lstsq(matrix, pseudo, rcond=None)[0]
        np.testing.assert_allclose(refitted, beta+state['bootstrap_errors'][b], atol=1e-12)


def test_silu_gradient_including_weight_penalty_and_unpenalized_biases():
    rng = np.random.default_rng(78)
    h = 4
    theta = rng.normal(0, .3, 4*h+h+h*h+h+h+1)
    inputs, target = rng.normal(size=(17, 4)), rng.normal(size=17)
    loss = SiLUResidualLoss(inputs, target, h, .01)
    gradient = loss.grad(theta, inputs)
    for _ in range(10):
        direction = rng.normal(size=len(theta))
        eps = 1e-6
        numerical = (loss.value(theta+eps*direction, inputs)-loss.value(theta-eps*direction, inputs))/(2*eps)
        assert numerical == pytest.approx(gradient@direction, rel=1e-6, abs=1e-8)


def test_mlp_calibration_cannot_change_point_predictor():
    rng = np.random.default_rng(713)
    x, a = rng.normal(size=(90, 3)), rng.normal(size=90)
    target = .2+.1*x[:, 0]+.25*a
    settings = {'hidden_width':4, 'weight_decay':.001, 'max_steps':40,
                'gradient_tolerance':1e-5, 'ftol':1e-12}
    network, _, info = fit_residual_mlp(x[:45], a[:45], target[:45], settings, 71)
    assert info['solver'] == 'optimization.solvers.run_first_order_minimize'
    np.testing.assert_allclose(network.mean, np.column_stack((x[:45], a[:45])).mean(0))
    features = network.features(x[45:], a[45:])
    signs = rng.choice([-1, 1], (31, 45))
    initial = network(x[45:], a[45:]).copy()
    one = calibrate_frozen_head(features, network.beta, target[45:], signs, .05)
    two = calibrate_frozen_head(features, network.beta, target[45:]+100, signs, .05)
    first = FrozenMLPBand(x[45:], network, one['inverse'], one['critical'])
    second = FrozenMLPBand(x[45:], network, two['inverse'], two['critical'])
    np.testing.assert_array_equal(first.evaluate(a[45:])[0], initial)
    np.testing.assert_array_equal(second.evaluate(a[45:])[0], initial)
    assert second.critical > first.critical
    assert first.evaluate(np.zeros((7, 45)))[0].shape == (7, 45)


def test_grid_containment_detects_bump_and_retains_domain_scope():
    class ZeroBand:
        contexts = np.array([[0., 1., 0.]])
        def evaluate(self, actions):
            shape = np.broadcast_shapes(np.shape(actions), (1,))
            return (np.zeros(shape), np.full(shape, .1), np.full(shape, .1), np.full(shape, .1))
    spec = {'context':[0,0,0], 'action':[0.], 'bump':{'amplitude':1.,'context_index':1,'center':.5,'scale':.1}}
    residual = explicit_model(spec)
    assert grid_coverage(ZeroBand(), residual, np.array([-1., 2.]))['transferred']
    event = grid_coverage(ZeroBand(), residual, np.array([-1., .5, 2.]))
    assert not event['transferred'] and event['certificate'] == 'finite_grid_only'
    metrics = numerical_residual_metrics(ZeroBand(), residual, .5, .45, 1e-7)
    expected = np.tanh(1.)**2*.1/np.sqrt(.1**2+2*.45)
    assert metrics['mse'] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize('suffix', ['interactions', 'nonlinear'])
def test_new_manifest_dispatch_and_unchanged_frozen_environment(suffix):
    manifest = load_manifest(ROOT/f'manifests/synthetic-polynomial-residual-{suffix}.json')
    assert build_launch_plan(manifest).task_count == 100
    old = load_manifest(ROOT/'manifests/synthetic-polynomial-residual-degree4.json')
    first, second = prepare_environment(manifest.payload), prepare_environment(old.payload)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    if suffix == 'nonlinear':
        np.testing.assert_array_equal(evaluation_rows(manifest.payload), np.arange(150, 300))
        np.testing.assert_allclose(coverage_grid(manifest.payload), coverage_grid(manifest.payload, True)[::4])
