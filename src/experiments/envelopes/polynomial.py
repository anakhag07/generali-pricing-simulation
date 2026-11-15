"""Additive polynomial residual queries, Gaussian metrics and all-real certificates (§7.22)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np
from numpy.polynomial import polynomial as pol
from scipy.integrate import quad_vec
from scipy.linalg import solve_triangular
import sympy as sp

from .feature_wild import WildFeatureDesign
from .quadratic import VARIABLE, nonnegative_on_real_line
from .wild import residual_radii
from experiments.offline_bootstrap_math import strict_conjunction_empty


def polynomial_features(contexts, actions, degree, interaction_order=0):
    """Evaluate additive context columns and original-action powers through degree k."""
    x, a = np.asarray(contexts, float), np.asarray(actions, float)
    if x.ndim != 2 or a.shape != (len(x),) or degree < 1:
        raise ValueError("Require paired contexts/actions and positive degree.")
    if type(interaction_order) is not int or not 0 <= interaction_order <= degree:
        raise ValueError('Interaction order must be an integer between zero and action degree.')
    return np.column_stack((np.ones(len(x)), x, *(a**j for j in range(1, degree+1)),
                            *(x[:, j]*a**power for power in range(1, interaction_order+1)
                              for j in range(x.shape[1]))))


@dataclass(frozen=True)
class AdditivePolynomial:
    """Represent an explicit context-linear function plus an action polynomial."""

    context: tuple
    action: tuple
    interactions: tuple = ()

    def __post_init__(self):
        if not self.action or not np.isfinite([*self.context, *self.action]).all():
            raise ValueError('Require finite polynomial coefficients.')
        for j, power, value in self.interactions:
            if (not isinstance(j, Integral) or isinstance(j, (bool, np.bool_))
                    or not isinstance(power, Integral) or isinstance(power, (bool, np.bool_))
                    or not 0 <= j < len(self.context) or not 1 <= power < len(self.action)
                    or not np.isfinite(value)):
                raise ValueError('Invalid context/action interaction coefficient.')
        object.__setattr__(self, 'interactions', tuple(
            (int(j), int(power), value) for j, power, value in self.interactions))

    def __call__(self, contexts, actions):
        x, a = np.asarray(contexts), np.asarray(actions)
        result = x @ self.context + pol.polyval(a, self.action)
        for j, power, value in self.interactions:
            result = result + value*x[..., j]*a**power
        return result

    def coefficients(self, contexts):
        """Return ascending action coefficients at each saved context."""
        result = np.tile(self.action, (len(contexts), 1)).astype(float)
        result[:, 0] += np.asarray(contexts) @ self.context
        for j, power, value in self.interactions:
            result[:, power] += value*np.asarray(contexts)[:, j]
        return result

    def feature_coefficients(self, degree, interaction_order=0):
        """Return the exact feature-coordinate embedding, rejecting misspecification."""
        if degree < len(self.action)-1 or any(power > interaction_order and value != 0 for _, power, value in self.interactions):
            raise ValueError("Residual is not represented in this feature space.")
        result = np.r_[self.action[0], self.context, self.action[1:],
                       np.zeros(degree+1-len(self.action)+len(self.context)*interaction_order)]
        for j, power, value in self.interactions:
            if value:
                result[1+len(self.context)+degree+(power-1)*len(self.context)+j] += value
        return result

    def exact(self, context):
        """Return a rational polynomial from the saved function and context floats."""
        constant = sum(_r(x)*_r(c) for x, c in zip(context, self.context))
        return _poly(self.action) + constant + sum(
            (_r(value)*_r(context[j])*VARIABLE**power for j, power, value in self.interactions), _poly([0]))


class PolynomialFeatureBand:
    """Replay a fixed polynomial head and QR geometry with asymmetric transfer."""

    def __init__(self, contexts, degree, beta, inverse, critical, interaction_order=0):
        self.contexts = np.asarray(contexts, float)
        if self.contexts.ndim != 2 or not len(self.contexts):
            raise ValueError('Polynomial bands require a nonempty context matrix.')
        self.degree = int(degree)
        self.interaction_order = int(interaction_order)
        if not 0 <= self.interaction_order <= self.degree:
            raise ValueError('Invalid interaction order.')
        self.beta, self.inverse = np.asarray(beta, float), np.asarray(inverse, float)
        self.critical = float(critical)
        n, d = self.contexts.shape
        p = d+1+degree+d*self.interaction_order
        if (self.beta.shape != (p,) or self.inverse.shape != (p, p)
                or not all(np.isfinite(v).all() for v in (self.contexts, self.beta, self.inverse))
                or not np.isfinite(critical) or critical < 0):
            raise ValueError("Incompatible or nonfinite polynomial band state.")
        mapping = np.zeros((n, p, degree+1))
        mapping[:, 0, 0] = 1
        mapping[:, 1:d+1, 0] = self.contexts
        for j in range(1, degree+1):
            mapping[:, d+j, j] = 1
        for power in range(1, self.interaction_order+1):
            for j in range(d):
                mapping[:, d+1+degree+(power-1)*d+j, power] = self.contexts[:, j]
        self.fitted = np.einsum('npj,p->nj', mapping, self.beta)
        self.norm_basis = np.einsum('pq,npj->nqj', self.inverse, mapping)
        self.norm_squared = np.zeros((n, 2*degree+1))
        for i in range(degree+1):
            for j in range(degree+1):
                self.norm_squared[:, i+j] += np.sum(
                    self.norm_basis[:, :, i]*self.norm_basis[:, :, j], axis=1)

    def evaluate(self, actions):
        """Evaluate arrays of personalized actions with final dimension n."""
        a = np.broadcast_to(np.asarray(actions, float), np.broadcast_shapes(np.shape(actions), (len(self.contexts),)))
        fitted = pol.polyval(a, self.fitted.T, tensor=False)
        squared = pol.polyval(a, self.norm_squared.T, tensor=False)
        if not np.isfinite(squared).all() or np.any(squared <= 0):
            raise FloatingPointError("Nonpositive/nonfinite polynomial query norm.")
        width = self.critical*np.sqrt(squared)
        lower, upper = residual_radii(fitted, width)
        return fitted, width, lower, upper

    def exact_fiber(self, context):
        """Return exact rational fit and squared radius without rounding coefficient products."""
        phi = [_poly([1]), *[_poly([x]) for x in context]]
        phi += [sp.Poly(VARIABLE**j, VARIABLE, domain=sp.QQ) for j in range(1, self.degree+1)]
        phi += [sp.Poly(_r(x)*VARIABLE**power, VARIABLE, domain=sp.QQ)
                for power in range(1, self.interaction_order+1) for x in context]
        fitted = sum((_r(b)*v for b, v in zip(self.beta, phi)), _poly([0]))
        whitened = [sum((_r(c)*v for c, v in zip(column, phi)), _poly([0]))
                    for column in self.inverse.T]
        squared = _r(self.critical)**2*sum((v*v for v in whitened), _poly([0]))
        return fitted, squared


def fit_polynomial_band(contexts, actions, responses, degree, multipliers, delta, *, design=None, interaction_order=0):
    """Calibrate the existing raw-residual feature ellipsoid from residual observations."""
    design = design or WildFeatureDesign(polynomial_features(contexts, actions, degree, interaction_order))
    beta = design.fit(responses)
    errors, statistics = design.bootstrap_errors(responses-design.design @ beta, multipliers)
    order = int(np.ceil(len(statistics)*(1-delta)))-1
    critical = float(np.partition(statistics, order)[order])
    inverse = solve_triangular(design.triangular, np.eye(len(beta)))
    band = PolynomialFeatureBand(contexts, degree, beta, inverse, critical, interaction_order)
    return band, {"beta": beta, "inverse": inverse, "triangular": design.triangular,
                  "critical": np.array(critical), "statistics": statistics,
                  "bootstrap_errors": errors, "leverage": design.leverage,
                  "condition_number": np.array(design.condition_number)}


def coefficient_contains(band, truth):
    """Certify the sufficient coefficient event using exact rational triangular solves."""
    if band.degree < len(truth.action)-1:
        return False
    try:
        actual = truth.feature_coefficients(band.degree, band.interaction_order)
    except ValueError:
        return False
    error = [Fraction(float(b))-Fraction(float(t)) for b, t in zip(band.beta, actual)]
    t = [Fraction(0)]*len(error)
    for i in range(len(error)-1, -1, -1):
        t[i] = (error[i]-sum(Fraction(float(band.inverse[i, j]))*t[j]
                            for j in range(i+1, len(t))))/Fraction(float(band.inverse[i, i]))
    return sum(v*v for v in t) <= Fraction(band.critical)**2


def polynomial_containment(truth, fitted, squared_radius):
    """Return auxiliary and asymmetric transferred containment on the entire real line."""
    difference = truth-fitted
    deficit = difference**2-squared_radius
    auxiliary = nonnegative_on_real_line(-deficit)
    if auxiliary:
        return True, True
    transferred = (strict_conjunction_empty([truth, difference, deficit])
                   and strict_conjunction_empty([-truth, -difference, deficit]))
    return False, bool(transferred)


def coverage_events(band, truth):
    """Certify joint events at all saved contexts; underdegree tails fail exactly."""
    if band.degree < len(truth.action)-1:
        return {"coefficient": False, "auxiliary": False, "transferred": False,
                "certificate": "residual_degree_exceeds_radius_degree", "contexts_checked": 0}
    if coefficient_contains(band, truth):
        return {"coefficient": True, "auxiliary": True, "transferred": True,
                "certificate": "exact_rational_coefficient_event", "contexts_checked": 0}
    auxiliary, transferred = True, True
    for index, context in enumerate(band.contexts):
        fitted, squared = band.exact_fiber(context)
        aux, trans = polynomial_containment(truth.exact(context), fitted, squared)
        auxiliary &= aux
        transferred &= trans
        if not auxiliary and not transferred:
            break
    return {"coefficient": False, "auxiliary": bool(auxiliary), "transferred": bool(transferred),
            "certificate": "exact_rational_polynomial_signs", "contexts_checked": index+1}


def gaussian_moments(mean, variance, degree):
    """Return raw Gaussian moments through the requested degree."""
    moments = np.ones(degree+1)
    if degree:
        moments[1] = mean
    for j in range(2, degree+1):
        moments[j] = mean*moments[j-1]+(j-1)*variance*moments[j-2]
    return moments


def residual_metrics(band, truth, mean, variance):
    """Integrate squared true-residual errors analytically over Gaussian actions."""
    degree = max(band.degree, len(truth.action)-1)
    actual = np.pad(truth.coefficients(band.contexts), ((0, 0), (0, degree+1-len(truth.action))))
    fitted = np.pad(band.fitted, ((0, 0), (0, degree-band.degree)))
    moments = gaussian_moments(mean, variance, 2*degree)
    moment_matrix = moments[np.add.outer(np.arange(degree+1), np.arange(degree+1))]
    grand_mean = float(np.mean(actual @ moments[:degree+1]))
    centered = actual.copy()
    centered[:, 0] -= grand_mean
    error = actual-fitted
    mse = float(np.mean(np.einsum('ni,ij,nj->n', error, moment_matrix, error)))
    denominator = float(np.mean(np.einsum('ni,ij,nj->n', centered, moment_matrix, centered)))
    return {"r_squared": 1-mse/denominator, "rmse": float(np.sqrt(max(0., mse))),
            "mse": mse, "residual_variance": denominator, "residual_mean": grand_mean}


def integrated_width(band, mean, variance, tolerance=1e-7):
    """Integrate both transferred radii over all Gaussian actions, averaging after transfer."""
    def integrand(z):
        density = np.exp(-z*z/2)/np.sqrt(2*np.pi)
        if density == 0:
            return np.zeros(4)
        fitted, width, lower, upper = band.evaluate(mean+np.sqrt(variance)*z)
        return density*np.array([np.mean(width), np.mean(lower), np.mean(upper), np.mean(abs(fitted))])
    first, error = quad_vec(integrand, -np.inf, np.inf, epsabs=tolerance, epsrel=tolerance,
                            limit=10000, norm='max')
    refined, refined_error = quad_vec(integrand, -np.inf, np.inf, epsabs=tolerance/5,
                                     epsrel=tolerance/5, limit=10000, norm='max')
    difference = float(np.max(abs(first-refined)))
    if not np.isfinite(refined).all() or difference > 10*tolerance or refined_error > 10*tolerance:
        raise RuntimeError("Gaussian width integration did not meet its accuracy target.")
    return {"mean_residual_half_width": float(refined[0]), "mean_lower_radius": float(refined[1]),
            "mean_upper_radius": float(refined[2]), "mean_full_width": float(refined[1]+refined[2]),
            "mean_fitted_magnitude": float(refined[3]), "integration_error": float(refined_error),
            "integration_refinement_difference": difference}


def certify_policy_gap(band, truth, residual, actions, tolerance):
    """Certify an upper gap for actual repository endpoints without choosing new actions."""
    fitted, width, lower, _ = band.evaluate(actions)
    values = truth(band.contexts, actions)-residual(band.contexts, actions)-lower
    certified = []
    for context, value in zip(band.contexts, values):
        h, q = band.exact_fiber(context)
        predictor = truth.exact(context)-residual.exact(context)
        # A small upward rounding guard protects the evaluated endpoint threshold.
        level = _r(float(np.nextafter(value+tolerance, np.inf)))
        first, second = predictor-level, predictor+h-level
        certified.append(strict_conjunction_empty([first, second, second**2-q]))
    return np.asarray(certified, dtype=bool)


def _r(value):
    return sp.Rational(float(value))


def _poly(coefficients):
    return sp.Poly.from_list([_r(x) for x in reversed(coefficients)], VARIABLE, domain=sp.QQ)


__all__ = ['AdditivePolynomial', 'PolynomialFeatureBand', 'polynomial_features',
           'fit_polynomial_band', 'coverage_events', 'residual_metrics', 'integrated_width']
