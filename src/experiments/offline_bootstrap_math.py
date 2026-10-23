"""Exact transferred-envelope checks and repository LCB optimization (§7.7)."""

from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.envelopes.quadratic import (
    VARIABLE, QuadraticErrorCertificate, nonnegative_on_real_line,
)
from objective.base import Objective
from optimization.solvers import run_first_order_minimize


def _coefficients(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Three finite quadratic coefficients are required.")
    return result


def _poly(coefficients):
    return sp.Poly.from_list(
        [sp.Rational(float(c)) for c in _coefficients(coefficients)[::-1]],
        VARIABLE, domain=sp.QQ,
    )


def _radius(radius_squared):
    result = sp.Poly(radius_squared, VARIABLE, domain=sp.QQ)
    if not nonnegative_on_real_line(result):
        raise ValueError("Squared radius must be nonnegative on the real line.")
    return result


def strict_conjunction_empty(polynomials):
    """Decide whether strict polynomial inequalities have no real solution."""
    active = []
    for value in polynomials:
        poly = sp.Poly(value, VARIABLE, domain=sp.QQ)
        if poly.is_zero or nonnegative_on_real_line(-poly):
            return True
        if poly.degree() > 0:
            active.append(poly)
    if not active:
        return False
    product = sp.Poly(1, VARIABLE, domain=sp.QQ)
    for poly in active:
        product *= poly
    product = product.sqf_part()
    precision = None
    for _ in range(100):
        intervals = [interval for interval, _ in product.intervals(eps=precision)]
        if not intervals:
            samples = [sp.Rational(0)]
        else:
            samples = [intervals[0][0] - 1, intervals[-1][1] + 1]
            samples += [(left[1] + right[0])/2
                        for left, right in zip(intervals, intervals[1:])]
        # Touching isolating intervals can place a midpoint exactly at a root.
        # Strict inequalities define open sets, so only open sign cells count.
        if all(product.eval(point) != 0 for point in samples):
            return not any(all(poly.eval(point) > 0 for poly in active)
                           for point in samples)
        precision = sp.Rational(1, 10**8) if precision is None else precision/100
    raise RuntimeError("Could not separate adjacent algebraic sign cells.")


def _branches(candidate, auxiliary):
    fitted = _poly(auxiliary)
    return fitted, 2*_poly(candidate) - fitted


def _above_branches_empty(branches, radius_squared):
    inequalities = []
    for branch in branches:
        inequalities.extend((branch, branch**2 - radius_squared))
    return strict_conjunction_empty(inequalities)


def transferred_contains(candidate, auxiliary, truth, radius_squared):
    """Certify containment of quadratic truth by the transferred band on R."""
    q = _radius(radius_squared)
    target = _poly(truth)
    branches = _branches(candidate, auxiliary)
    # The triangle-inequality implication also avoids unnecessary root isolation.
    if nonnegative_on_real_line(q - (branches[0] - target)**2):
        return True
    return (_above_branches_empty([branch - target for branch in branches], q)
            and _above_branches_empty([target - branch for branch in branches], q))


def transferred_below_level(candidate, auxiliary, radius_squared, level):
    """Certify a global upper LCB level without selecting an optimizer action."""
    if not np.isfinite(level):
        raise ValueError("A finite LCB level is required.")
    target = sp.Rational(float(level))
    return _above_branches_empty(
        [branch - target for branch in _branches(candidate, auxiliary)],
        _radius(radius_squared),
    )


class TransferredLCBObjective(Objective):
    """Negative transferred LCB with the auxiliary branch chosen at exact ties."""

    def __init__(self, candidate, auxiliary, covariance, scale):
        self.candidate = _coefficients(candidate)
        self.auxiliary = _coefficients(auxiliary)
        self.covariance = np.asarray(covariance, dtype=float)
        self.scale = float(scale)
        if (self.covariance.shape != (3, 3)
                or not np.isfinite(self.covariance).all()
                or not np.isfinite(self.scale) or self.scale < 0):
            raise ValueError("A finite covariance and nonnegative radius scale are required.")

    def _branch(self, p):
        reflected = 2*self.candidate - self.auxiliary
        return self.auxiliary if p @ self.auxiliary <= p @ reflected else reflected

    def value(self, theta, x_batch):
        """Evaluate the negative lower envelope at a scalar action."""
        a = float(theta[0])
        p = np.array([1., a, a*a])
        return float(-p @ self._branch(p)
                     + self.scale*np.sqrt(p @ self.covariance @ p))

    def grad(self, theta, x_batch):
        """Return the analytical derivative, using a deterministic kink subgradient."""
        a = float(theta[0])
        p, dp = np.array([1., a, a*a]), np.array([0., 1., 2*a])
        return np.array([-dp @ self._branch(p)
                         + self.scale*(dp @ self.covariance @ p)
                         / np.sqrt(p @ self.covariance @ p)])


def _candidate_lower_value(branches, radius_squared, action):
    a = sp.Rational(float(action))
    q = radius_squared.eval(a)
    root_upper = float(np.nextafter(np.sqrt(float(q)), np.inf)) if q else 0.
    if not np.isfinite(root_upper):
        return -np.inf
    while sp.Rational(root_upper)**2 < q:
        root_upper = float(np.nextafter(root_upper, np.inf))
    exact_lower = min(branch.eval(a) for branch in branches) - sp.Rational(root_upper)
    lower = float(exact_lower)
    if np.isfinite(lower):
        while sp.Rational(lower) > exact_lower:
            lower = float(np.nextafter(lower, -np.inf))
    return lower


def optimize_transferred_lcb(candidate, auxiliary, covariance, sigma_hat, critical, config):
    """Optimize through repository solvers and certify the global all-real gap."""
    if not np.isfinite(critical) or critical < 0:
        raise ValueError("A finite nonnegative critical value is required.")
    tolerance = float(config["global_gap_tolerance"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A finite positive global gap tolerance is required.")
    cert = QuadraticErrorCertificate(covariance, sigma_hat)
    q = cert.q.mul_ground(sp.Rational(float(critical))**2)
    branches = _branches(candidate, auxiliary)
    leading = min(branch.nth(2) for branch in branches)
    tail_square = q.nth(4)
    if leading >= 0 and leading**2 >= tail_square:
        return {"optimization_state": "unbounded" if leading**2 > tail_square else "degenerate_tail",
                "attempts": [], "action": None, "global_gap_upper": None}
    objective = TransferredLCBObjective(candidate, auxiliary, covariance, critical*sigma_hat)
    attempts = []
    for start in config["starts"]:
        theta, trace = run_first_order_minimize(
            np.array([start], dtype=float), np.zeros((1, 1)), objective,
            t_steps=config["t_steps"], n_grad_samples=1, sigma=0.,
            algorithm=config["step_rule"], grad_norm_tol=config["gradient_tolerance"],
            ftol=config["ftol"],
        )
        action = float(theta[0])
        attempts.append({"start": start, "action": action,
                         "value": float(-objective.value(theta, None)),
                         "success": bool(trace.optimizer_success),
                         "status": trace.optimizer_status, "message": trace.optimizer_message,
                         "gradient_norm": float(np.linalg.norm(objective.grad(theta, None)))})
    finite = [attempt for attempt in attempts
              if np.isfinite([attempt["action"], attempt["value"]]).all()]
    if not finite:
        return {"optimization_state": "optimizer_failed", "attempts": attempts,
                "action": None, "global_gap_upper": None}
    # Only actual repository solver outputs are eligible actions.
    best = max(finite, key=lambda attempt: attempt["value"])
    lower = _candidate_lower_value(branches, q, best["action"])
    exact_upper = sp.Rational(lower) + sp.Rational(tolerance) if np.isfinite(lower) else None
    upper = float(exact_upper) if exact_upper is not None else np.nan
    # Round down so a successful certificate also respects the requested gap.
    while np.isfinite(upper) and sp.Rational(upper) > exact_upper:
        upper = float(np.nextafter(upper, -np.inf))
    certified = (np.isfinite(upper)
                 and transferred_below_level(candidate, auxiliary, q, upper))
    gap = None
    if certified:
        exact_gap = sp.Rational(upper) - sp.Rational(lower)
        gap = float(exact_gap)
        while sp.Rational(gap) < exact_gap:
            gap = float(np.nextafter(gap, np.inf))
    return {"optimization_state": "certified" if certified else "uncertified",
            "action": best["action"], "attempts": attempts,
            "solver_success": best["success"], "candidate_value_lower": lower,
            "global_value_upper": upper if certified else None,
            "global_gap_upper": gap}
