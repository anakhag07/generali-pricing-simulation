"""SiLU residual fitting and calibration-only bootstrap of a frozen output head."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit

from optimization.solvers import run_first_order_minimize
from .feature_wild import WildFeatureDesign
from .wild import residual_radii


def _unpack(theta, hidden):
    h = hidden
    k = 4*h
    w1, b1 = theta[:k].reshape(4, h), theta[k:k+h]
    k += h
    w2, b2 = theta[k:k+h*h].reshape(h, h), theta[k+h*h:k+h*h+h]
    k += h*h+h
    return w1, b1, w2, b2, theta[k:k+h], theta[-1]


def _activation(value):
    sigmoid = expit(value)
    return value*sigmoid, sigmoid+value*sigmoid*(1-sigmoid)


class SiLUResidualLoss:
    """Full-batch residual MSE plus an L2 penalty on weights, with exact gradients."""

    def __init__(self, inputs, targets, hidden, weight_decay):
        self.inputs, self.targets = np.asarray(inputs), np.asarray(targets)
        self.hidden, self.weight_decay = hidden, weight_decay
        self.last, self.cached = None, None

    def _evaluate(self, theta):
        if self.last is None or not np.array_equal(theta, self.last):
            w1, b1, w2, b2, w3, b3 = _unpack(theta, self.hidden)
            h1, d1 = _activation(self.inputs@w1+b1)
            h2, d2 = _activation(h1@w2+b2)
            residual = h2@w3+b3-self.targets
            delta = residual/len(residual)
            second = delta[:, None]*w3*d2
            first = (second@w2.T)*d1
            penalty = self.weight_decay
            gradient = np.r_[(self.inputs.T@first+penalty*w1).ravel(), first.sum(0),
                             (h1.T@second+penalty*w2).ravel(), second.sum(0),
                             h2.T@delta+penalty*w3, delta.sum()]
            value = .5*np.mean(residual**2)+.5*penalty*sum(np.sum(w*w) for w in (w1, w2, w3))
            self.last, self.cached = np.array(theta, copy=True), (float(value), gradient)
        return self.cached

    def value(self, theta, inputs):
        """Return the fixed full-training-batch objective."""
        return self._evaluate(theta)[0]

    def grad(self, theta, inputs):
        """Return the gradient in flattened network-weight order."""
        return self._evaluate(theta)[1]


class FrozenResidualMLP:
    """Replay a trained two-layer SiLU network and its unchanged linear head."""

    def __init__(self, theta, mean, scale, hidden):
        self.theta, self.mean, self.scale = (np.array(a, dtype=float, copy=True) for a in (theta, mean, scale))
        self.hidden = int(hidden)
        expected = 4*self.hidden+self.hidden+self.hidden**2+self.hidden+self.hidden+1
        if (self.theta.shape != (expected,) or self.mean.shape != (4,) or self.scale.shape != (4,)
                or np.any(self.scale <= 0) or not all(np.isfinite(a).all() for a in (self.theta, self.mean, self.scale))):
            raise ValueError('Invalid saved MLP state.')
        for array in (self.theta, self.mean, self.scale):
            array.setflags(write=False)
        *_, output, intercept = _unpack(self.theta, self.hidden)
        self.beta = np.r_[intercept, output]
        self.beta.setflags(write=False)

    def features(self, contexts, actions):
        """Evaluate intercept plus frozen hidden features for broadcast context/action inputs."""
        x, a = np.asarray(contexts), np.asarray(actions)
        shape = np.broadcast_shapes(x.shape[:-1], a.shape)
        inputs = np.concatenate((np.broadcast_to(x, shape+(3,)), np.broadcast_to(a, shape)[..., None]), axis=-1)
        scaled = ((inputs-self.mean)/self.scale).reshape(-1, 4)
        w1, b1, w2, b2, _, _ = _unpack(self.theta, self.hidden)
        first = _activation(scaled@w1+b1)[0]
        second = _activation(first@w2+b2)[0]
        return np.column_stack((np.ones(len(second)), second)).reshape(shape+(self.hidden+1,))

    def __call__(self, contexts, actions):
        return self.features(contexts, actions)@self.beta

    def state(self):
        """Return complete replay state, including the original training head."""
        return {'theta': self.theta, 'input_mean': self.mean, 'input_scale': self.scale,
                'hidden': np.array(self.hidden), 'beta': self.beta}


def fit_residual_mlp(contexts, actions, responses, settings, seed):
    """Train the complete residual network once through the repository first-order solver."""
    inputs = np.column_stack((contexts, actions))
    mean, scale = inputs.mean(0), inputs.std(0)
    if np.any(scale <= 0):
        raise ValueError('MLP training inputs have a constant coordinate.')
    scaled = (inputs-mean)/scale
    h, rng = settings['hidden_width'], np.random.default_rng(seed)
    theta0 = np.r_[rng.uniform(-np.sqrt(6/(4+h)), np.sqrt(6/(4+h)), 4*h), np.zeros(h),
                   rng.uniform(-np.sqrt(3/h), np.sqrt(3/h), h*h), np.zeros(h),
                   rng.uniform(-np.sqrt(6/(h+1)), np.sqrt(6/(h+1)), h), 0.]
    objective = SiLUResidualLoss(scaled, responses, h, settings['weight_decay'])
    theta, trace = run_first_order_minimize(theta0, scaled, objective,
        t_steps=settings['max_steps'], n_grad_samples=1, sigma=0., algorithm='l-bfgs-b',
        grad_norm_tol=settings['gradient_tolerance'], ftol=settings['ftol'])
    network = FrozenResidualMLP(theta, mean, scale, h)
    state = {**network.state(), 'losses': np.asarray(trace.objective_values),
             'theta_trace': np.asarray(trace.theta_values)}
    info = {'success': bool(trace.optimizer_success), 'message': str(trace.optimizer_message),
            'steps': int(trace.steps[-1]), 'parameters': len(theta), 'initialization_seed': seed,
            'training_rmse': float(np.sqrt(np.mean((network(contexts, actions)-responses)**2))),
            'solver': 'optimization.solvers.run_first_order_minimize'}
    return network, state, info


def calibrate_frozen_head(features, original_beta, responses, multipliers, delta):
    """Bootstrap head deviations without fitting or replacing the original point head."""
    features = np.asarray(features, dtype=float)
    if not 0 < delta < 1:
        raise ValueError('Require miscoverage strictly between zero and one.')
    design = WildFeatureDesign(features)
    beta = np.array(original_beta, copy=True)
    if beta.shape != (features.shape[1],):
        raise ValueError('Frozen head and calibration features disagree.')
    errors = np.asarray(responses)-design.design@beta
    deviations, statistics = design.bootstrap_errors(errors, multipliers)
    order = int(np.ceil(len(statistics)*(1-delta)))-1
    critical = float(np.partition(statistics, order)[order])
    inverse = solve_triangular(design.triangular, np.eye(len(beta)))
    return {'beta': beta, 'inverse': inverse, 'triangular': design.triangular,
            'critical': np.array(critical), 'bootstrap_errors': deviations,
            'statistics': statistics, 'calibration_errors': errors,
            'leverage': design.leverage, 'condition_number': np.array(design.condition_number)}


class FrozenMLPBand:
    """Evaluate a calibration ellipsoid around the unchanged trained residual network."""

    def __init__(self, contexts, network, inverse, critical):
        self.contexts, self.network = np.asarray(contexts), network
        self.inverse, self.critical = np.asarray(inverse), float(critical)
        self.beta = network.beta
        if self.inverse.shape != (len(self.beta), len(self.beta)) or not np.isfinite(self.inverse).all() or not np.isfinite(self.critical) or self.critical < 0:
            raise ValueError('Invalid frozen-head band geometry.')

    def evaluate(self, actions):
        """Return residual prediction, auxiliary width and both asymmetric radii."""
        phi = self.network.features(self.contexts, actions)
        fitted = phi@self.beta
        width = self.critical*np.linalg.norm(phi@self.inverse, axis=-1)
        if not np.isfinite(fitted).all() or not np.isfinite(width).all():
            raise FloatingPointError('Nonfinite MLP query.')
        lower, upper = residual_radii(fitted, width)
        return fitted, width, lower, upper


__all__ = ['FrozenResidualMLP', 'FrozenMLPBand', 'SiLUResidualLoss',
           'fit_residual_mlp', 'calibrate_frozen_head']
