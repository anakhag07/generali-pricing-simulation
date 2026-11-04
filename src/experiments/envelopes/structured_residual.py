"""Structured residual models, frozen-center calibration and finite-grid evaluation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec

from experiments.provenance import file_record, read_json, save_arrays_atomic, write_json_atomic
from .feature_wild import WildFeatureDesign
from .polynomial import (AdditivePolynomial, PolynomialFeatureBand, polynomial_features,
                         fit_polynomial_band, coverage_events)
from .mlp_head import FrozenResidualMLP, FrozenMLPBand, fit_residual_mlp, calibrate_frozen_head


class LocalizedResidual:
    """Evaluate the declared additive/interaction polynomial plus a localized nonlinear bump."""

    def __init__(self, base, amplitude, context_index, center, scale):
        self.base, self.amplitude, self.context_index = base, amplitude, context_index
        self.center, self.scale = center, scale

    def __call__(self, contexts, actions):
        return (self.base(contexts, actions)+self.amplitude*np.tanh(np.asarray(contexts)[..., self.context_index])
                *np.exp(-.5*((np.asarray(actions)-self.center)/self.scale)**2))


def explicit_model(spec):
    """Construct a residual/objective from explicit coefficients without fitting."""
    base = AdditivePolynomial(tuple(spec['context']), tuple(spec['action']),
                              tuple(tuple(v) for v in spec.get('interactions', [])))
    if 'bump' not in spec:
        return base
    return LocalizedResidual(base, **spec['bump'])


def evaluation_rows(p):
    """Return all contexts for ordinary OLS, or calibration-half contexts for frozen heads."""
    return np.arange(p['n']//2 if p['fit_protocol'] == 'frozen_training_head' else 0, p['n'])


def validate_structured(p):
    """Validate model, split, domain and training choices before creating artifacts."""
    if p['fit_protocol'] not in {'same_sample_ols', 'frozen_training_head'}:
        raise ValueError('Unknown structured fitting protocol.')
    if p['n'] % 2 or p['target']['contexts'] != 'calibration':
        raise ValueError('Require an even sample and calibration evaluation contexts.')
    nonlinear = 'bump' in p['residual']
    if p['target']['actions'] not in {'real_line', 'finite_grid'}:
        raise ValueError('Coverage actions must be real_line or finite_grid.')
    if nonlinear != (p['target']['actions'] == 'finite_grid'):
        raise ValueError('Nonlinear residuals require explicitly finite-grid coverage.')
    if nonlinear != (p['fit_protocol'] == 'frozen_training_head'):
        raise ValueError('Nonlinear comparison requires the frozen-center 50/50 split.')
    if nonlinear:
        grid = p['target']
        if (type(grid['grid_points']) is not int or grid['grid_points'] < 3
                or type(grid['refinement_points']) is not int
                or grid['refinement_points'] <= grid['grid_points']
                or (grid['refinement_points']-1) % (grid['grid_points']-1)
                or not np.isfinite(grid['standard_deviations']) or grid['standard_deviations'] <= 0):
            raise ValueError('Require a positive window and nested primary/refinement grids.')
        b = p['residual']['bump']
        if (set(b) != {'amplitude', 'context_index', 'center', 'scale'} or b['scale'] <= 0
                or type(b['context_index']) is not int or not 0 <= b['context_index'] < 3
                or not np.isfinite(list(b.values())).all()):
            raise ValueError('Invalid localized residual parameters.')
    names = []
    for model in p['models']:
        name = model['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name) or name in names:
            raise ValueError('Require unique safe model names.')
        names.append(name)
        if model['kind'] == 'polynomial':
            if model['degree'] != 6 or type(model['interaction_order']) is not int or not 0 <= model['interaction_order'] <= 2:
                raise ValueError('Require action degree six and interaction order 0, 1 or 2.')
            dimension = 10+3*model['interaction_order']
        elif model['kind'] == 'mlp':
            if not nonlinear or model['hidden_width'] != 32:
                raise ValueError('MLP requires the nonlinear split and two width-32 hidden layers.')
            if (type(model['max_steps']) is not int or model['max_steps'] < 1
                    or min(model['gradient_tolerance'], model['ftol'], model['weight_decay']) <= 0
                    or not np.isfinite([model['gradient_tolerance'], model['ftol'], model['weight_decay']]).all()
                    or type(p['seeds']['mlp']) is not int or p['seeds']['mlp'] < 0):
                raise ValueError('Invalid declared MLP optimizer/regularization settings.')
            dimension = 33
        else:
            raise ValueError('Unknown residual model kind.')
        n_fit = p['n']//2 if nonlinear else p['n']
        if n_fit <= dimension:
            raise ValueError('Each fitting/calibration sample must exceed head dimension.')
    if not names:
        raise ValueError('At least one model is required.')
    explicit_model(p['residual'])


def restore_structured_band(p, snapshot, state, name, *, contexts=None):
    """Restore a fitted band without training, resampling or replacing its original head."""
    spec = next(m for m in p['models'] if m['name'] == name)
    X = snapshot['X'][evaluation_rows(p)] if contexts is None else np.asarray(contexts)
    saved = {k[len(name)+1:]: v for k, v in state.items() if k.startswith(name+'_')}
    if spec['kind'] == 'polynomial':
        return PolynomialFeatureBand(X, spec['degree'], saved['beta'], saved['inverse'],
                                     saved['critical'], spec['interaction_order'])
    network = FrozenResidualMLP(saved['theta'], saved['input_mean'], saved['input_scale'], saved['hidden'])
    if not np.array_equal(network.beta, saved['beta']):
        raise ValueError('Calibration replaced the original trained MLP output head.')
    return FrozenMLPBand(X, network, saved['inverse'], saved['critical'])


def grid_coverage(band, residual, grid):
    """Check joint containment at specified points only, never claiming continuum coverage."""
    auxiliary, transferred, margin = True, True, np.inf
    for start in range(0, len(grid), 32):
        a = np.asarray(grid[start:start+32])[:, None]
        fitted, width, lower, upper = band.evaluate(a)
        actual = residual(band.contexts, a)
        auxiliary &= bool(np.all(abs(actual-fitted) <= width))
        deficit = np.minimum(actual+lower, upper-actual)
        transferred &= bool(np.all(deficit >= 0))
        margin = min(margin, float(deficit.min()))
    return {'coefficient': None, 'auxiliary': auxiliary, 'transferred': transferred,
            'certificate': 'finite_grid_only', 'contexts_checked': len(band.contexts),
            'action_points': len(grid), 'minimum_transferred_margin': margin}


def coverage_grid(p, refined=False):
    """Return the predeclared finite action grid, independent of fits and observations."""
    mean = p['design']['action_mean']
    span = p['target']['standard_deviations']*np.sqrt(p['design']['action_variance'])
    count = p['target']['refinement_points' if refined else 'grid_points']
    return np.linspace(mean-span, mean+span, count)


def fit_structured_models(p, index, project, digest, snapshot, responses, signs, arrays):
    """Fit each requested model and save calibration errors, perturbations and coverage events."""
    X, a = snapshot['X'], snapshot['actions']
    frozen = p['fit_protocol'] == 'frozen_training_head'
    training = np.arange(p['n']//2) if frozen else np.arange(p['n'])
    calibration = evaluation_rows(p)
    arrays.update(training_rows=training, calibration_rows=calibration)
    residual = explicit_model(p['residual'])
    events, diagnostics = {}, {}
    for spec in p['models']:
        name = spec['name']
        if spec['kind'] == 'polynomial':
            matrix = polynomial_features(X, a, spec['degree'], spec['interaction_order'])
            if frozen:
                beta = WildFeatureDesign(matrix[training]).fit(responses[training])
                state = calibrate_frozen_head(matrix[calibration], beta, responses[calibration],
                                              signs[:, calibration], p['bootstrap']['delta'])
            else:
                _, state = fit_polynomial_band(X, a, responses, spec['degree'], signs,
                    p['bootstrap']['delta'], interaction_order=spec['interaction_order'])
            info = {'training_rmse': float(np.sqrt(np.mean((responses[training]-matrix[training]@state['beta'])**2))),
                    'success': True, 'solver': 'ordinary_least_squares_qr'}
        else:
            stem = project/'fits'/f'rep-{index:03d}-{name}'
            if stem.with_suffix('.json').exists():
                info = read_json(stem.with_suffix('.json'))
                if info['contract_sha256'] != digest or info['arrays'] != file_record(stem.with_suffix('.npz')):
                    raise ValueError('Incompatible saved MLP training state.')
                with np.load(stem.with_suffix('.npz'), allow_pickle=False) as z:
                    network_state = dict(z)
                network = FrozenResidualMLP(network_state['theta'], network_state['input_mean'], network_state['input_scale'], network_state['hidden'])
            else:
                network, network_state, info = fit_residual_mlp(X[training], a[training], responses[training], spec, p['seeds']['mlp'])
                network_state.pop('theta_trace', None)
                save_arrays_atomic(stem.with_suffix('.npz'), **network_state)
                info = {**info, 'contract_sha256': digest, 'arrays': file_record(stem.with_suffix('.npz'))}
                write_json_atomic(stem.with_suffix('.json'), info)
            state = calibrate_frozen_head(network.features(X[calibration], a[calibration]), network.beta,
                                          responses[calibration], signs[:, calibration], p['bootstrap']['delta'])
            state.update({key: value for key, value in network.state().items() if key != 'beta'})
            if not np.array_equal(state['beta'], network.beta):
                raise AssertionError('The original training head must remain unchanged.')
        if frozen:
            info['calibration_rmse'] = float(np.sqrt(np.mean(state['calibration_errors']**2)))
        info.update(training_rows=len(training), calibration_rows=len(calibration),
                    head_dimension=len(state['beta']), condition_number=float(state['condition_number']))
        diagnostics[name] = info
        arrays.update({name+'_'+key: value for key, value in state.items()})
        band = restore_structured_band(p, snapshot, arrays, name)
        if p['target']['actions'] == 'real_line':
            events[name] = coverage_events(band, residual)
        else:
            event = grid_coverage(band, residual, coverage_grid(p))
            fine = grid_coverage(band, residual, coverage_grid(p, refined=True))
            event.update(refined_transferred=fine['transferred'], refined_auxiliary=fine['auxiliary'],
                         refinement_points=fine['action_points'], refined_minimum_transferred_margin=fine['minimum_transferred_margin'])
            events[name] = event
    return events, diagnostics


def numerical_residual_metrics(band, residual, mean, variance, tolerance):
    """Integrate true-residual error over all Gaussian actions, with refinement checks."""
    def integrand(z):
        density = np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        if density == 0:
            return np.zeros(3)
        a = mean+np.sqrt(variance)*z
        actual = residual(band.contexts, a)
        fitted = band.evaluate(a)[0]
        return density*np.array([np.mean((actual-fitted)**2), np.mean(actual), np.mean(actual**2)])
    first, _ = quad_vec(integrand, -np.inf, np.inf, epsabs=tolerance, epsrel=tolerance)
    values, error = quad_vec(integrand, -np.inf, np.inf, epsabs=tolerance/5, epsrel=tolerance/5)
    difference = float(np.max(abs(first-values)))
    if not np.isfinite(values).all() or error > 10*tolerance or difference > 10*tolerance:
        raise RuntimeError('Residual metric integration did not converge.')
    mse, mean_g, second = values
    denominator = second-mean_g**2
    if denominator <= 0:
        raise ValueError('Residual variance must be positive.')
    return {'r_squared': float(1-mse/denominator), 'rmse': float(np.sqrt(mse)),
            'mse': float(mse), 'residual_variance': float(denominator), 'residual_mean': float(mean_g),
            'metric_integration_error': float(error), 'metric_refinement_difference': difference}


__all__ = ['LocalizedResidual', 'explicit_model', 'restore_structured_band', 'grid_coverage']
