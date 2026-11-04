"""Independent interaction identities and frozen-network calibration checks."""

import json
import copy
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
    evaluation_rows, restore_structured_band, fit_structured_models, numerical_residual_metrics,
    observed_pair_metrics, validate_structured)
from experiments.envelopes.wild_study import load_manifest, build_launch_plan
from experiments.envelopes.polynomial_study import prepare_environment

ROOT = Path(__file__).resolve().parents[2]


def test_observed_pair_scores_keep_negative_values_and_constant_target_undefined():
    score = observed_pair_metrics(np.array([0., 2.]), np.array([4., 4.]))
    assert score['r_squared'] == -9.
    assert score['rmse'] == pytest.approx(np.sqrt(10))
    assert observed_pair_metrics(np.ones(3), np.zeros(3))['r_squared'] is None
    with pytest.raises(ValueError):
        observed_pair_metrics(np.ones(3), np.zeros(2))


def test_capacity_manifest_validates_head_sample_size_and_preserves_split():
    p = load_manifest(ROOT/'manifests/synthetic-polynomial-residual-nonlinear-capacity.json').payload
    assert {m['hidden_width'] for m in p['models'] if m['kind']=='mlp'} == {2,4,8,16,32}
    np.testing.assert_array_equal(evaluation_rows(p), np.arange(150,300))
    p = copy.deepcopy(p)
    p['models'][-1]['hidden_width'] = 149
    with pytest.raises(ValueError, match='exceed head dimension'):
        validate_structured(p)


def test_longer_training_preserves_same_initialization_and_trajectory_prefix():
    rng = np.random.default_rng(88)
    x, a = rng.normal(size=(45,3)), rng.normal(size=45)
    response = np.sin(a)+.5*rng.normal(size=45)
    settings = {'hidden_width':2, 'weight_decay':.001, 'max_steps':20,
                'gradient_tolerance':1e-12, 'ftol':1e-15}
    _, first, first_info = fit_residual_mlp(x,a,response,settings,52)
    _, second, second_info = fit_residual_mlp(x,a,response,{**settings,'max_steps':40},52)
    np.testing.assert_array_equal(first['losses'], second['losses'][:len(first['losses'])])
    assert second_info['final_objective'] <= first_info['final_objective']
    assert first_info['parameters'] == second_info['parameters'] == 19


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


def test_split_pipeline_and_replay_preserve_training_prediction(tmp_path):
    """Changing calibration responses cannot influence training inputs or the saved center."""
    p = load_manifest(ROOT/'manifests/synthetic-polynomial-residual-nonlinear.json').payload
    p = copy.deepcopy(p)
    p['target'].update(grid_points=5, refinement_points=9)
    snapshot = prepare_environment(p)
    rng = np.random.default_rng(91)
    responses = explicit_model(p['residual'])(snapshot['X'], snapshot['actions']) + rng.normal(size=300)
    signs = rng.choice([-1, 1], size=(19, 300))
    inputs = np.column_stack((snapshot['X'][:150], snapshot['actions'][:150]))
    network = FrozenResidualMLP(rng.normal(0, .2, 1249), inputs.mean(0), inputs.std(0), 32)
    seen = []
    def training(x, a, r, settings, seed):
        seen.append((x.copy(), a.copy(), r.copy()))
        return network, network.state(), {'success':True, 'training_rmse':1.}
    states = []
    with patch('experiments.envelopes.structured_residual.fit_residual_mlp', side_effect=training):
        for index, offset in enumerate([0., 10.]):
            targets = responses.copy()
            targets[150:] += offset
            state = {}
            fit_structured_models(p, index, tmp_path, 'test', snapshot, targets, signs, state)
            states.append(state)
    assert len(seen) == 2
    for first, second in zip(seen[0], seen[1]):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(seen[0][2], responses[:150])
    for spec in p['models']:
        name = spec['name']
        np.testing.assert_array_equal(states[0][name+'_beta'], states[1][name+'_beta'])
        assert states[0][name+'_critical'] != states[1][name+'_critical']
        band = restore_structured_band(p, snapshot, states[0], name)
        assert band.evaluate(snapshot['actions'][150:])[0].shape == (150,)
    restored = restore_structured_band(p, snapshot, states[0], 'mlp-frozen-head')
    np.testing.assert_array_equal(restored.evaluate(snapshot['actions'][150:])[0],
                                  network(snapshot['X'][150:], snapshot['actions'][150:]))
    states[0]['mlp-frozen-head_beta'][0] += .01
    with pytest.raises(ValueError, match='replaced'):
        restore_structured_band(p, snapshot, states[0], 'mlp-frozen-head')
