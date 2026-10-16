"""Fixed-design raw-residual wild bootstrap and linear bands (MATH §7.9)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .core import FittedEnvelope, TargetDomain, locations, values
from .linear import LinearModel, check_linear_containment, features


class WildLinearDesign:
    """Cache OLS geometry and the all-real design-scaled statistic at logged contexts."""

    def __init__(self, X, actions):
        X, actions = locations(X, actions)
        self.X = np.array(X, copy=True)
        self.design = features(X, actions)
        n, p = self.design.shape
        if n <= p or np.linalg.matrix_rank(self.design) != p:
            raise ValueError("Wild OLS requires full column rank and n > d+2.")
        self._orthogonal, self._triangular = np.linalg.qr(self.design, mode="reduced")
        inverse = solve_triangular(self._triangular, np.eye(p))
        self.covariance = inverse @ inverse.T
        self.leverage = np.sum(self._orthogonal**2, axis=1)
        # With no contexts every action fiber is identical; retain all rows only
        # for fitting and observed leverage, not for repeated supremum checks.
        self._contexts = self.design[:1, :-1] if X.shape[1] == 0 else self.design[:, :-1]
        self._cross = self._contexts @ self.covariance[:-1, -1]
        self._constant = np.einsum(
            "ij,jk,ik->i", self._contexts, self.covariance[:-1, :-1], self._contexts,
        )
        self._action_variance = self.covariance[-1, -1]
        self._schur = self._constant - self._cross**2 / self._action_variance
        if (not np.isfinite(self.covariance).all() or self._action_variance <= 0
                or np.any(self._schur <= 0)):
            raise ValueError("Numerically singular all-real design geometry.")
        for array in (self.X, self.design, self.covariance, self.leverage):
            array.flags.writeable = False

    def fit(self, y):
        """Fit the auxiliary OLS coefficients using the cached QR factors."""
        y = values(y, len(self.design), "responses")
        return solve_triangular(self._triangular, self._orthogonal.T @ y)

    def bootstrap_errors(self, y, multipliers):
        """Return OLS and raw-residual Rademacher coefficient differences for supplied draws."""
        y = values(y, len(self.design), "responses")
        multipliers = np.asarray(multipliers)
        if (multipliers.ndim != 2 or multipliers.shape[1] != len(self.design)
                or len(multipliers) == 0 or np.iscomplexobj(multipliers)
                or not np.all((multipliers == -1) | (multipliers == 1))):
            raise ValueError("Multipliers must have shape (B,n) with values -1 or +1.")
        beta = self.fit(y)
        residuals = y - self.design @ beta
        # This is exactly refitting y*=P beta+residuals*xi on unchanged P.
        errors = solve_triangular(
            self._triangular, self._orthogonal.T @ (multipliers * residuals).T,
        ).T
        return beta, errors

    def statistics(self, errors):
        r"""Evaluate $$T(e)^2=\max_i (H_i'e)'(H_i'Q^{-1}H_i)^{-1}(H_i'e).$$"""
        errors = np.asarray(errors, dtype=float)
        if (errors.ndim != 2 or errors.shape[1] != self.design.shape[1]
                or not np.isfinite(errors).all()):
            raise ValueError("Coefficient errors must have finite shape (B,p).")
        intercepts = errors[:, :-1] @ self._contexts.T
        intercepts -= errors[:, -1, None] * (self._cross / self._action_variance)
        # Schur-complement form is a sum of nonnegative squares and includes
        # the projective limit in both action tails, without selecting an action.
        squared = intercepts**2 / self._schur
        squared += errors[:, -1, None]**2 / self._action_variance
        return np.sqrt(np.max(squared, axis=1))

    def envelope(self, predictor, beta, critical, metadata=None):
        r"""Transfer $$w=q\sqrt{\phi'Q^{-1}\phi}$$ to a frozen predictor with replayable state."""
        beta = values(beta, self.design.shape[1], "coefficients").copy()
        critical = _critical(critical)
        auxiliary_model = LinearModel(beta)
        covariance = self.covariance

        def components(X, actions):
            phi = features(X, actions)
            if phi.shape[1] != len(beta):
                raise ValueError("Envelope and evaluation context dimensions disagree.")
            auxiliary = auxiliary_model(X, actions)
            leverage = np.einsum("ij,jk,ik->i", phi, covariance, phi)
            if not np.isfinite(leverage).all() or np.any(leverage < 0):
                raise ValueError("Numerical leverage evaluation is negative or nonfinite.")
            return {"auxiliary": auxiliary, "width": np.sqrt(critical**2 * leverage),
                    "discrepancy": np.abs(values(predictor(X, actions), len(actions), "predictor") - auxiliary)}

        def radius(X, actions):
            parts = components(X, actions)
            return parts["discrepancy"] + parts["width"]

        return FittedEnvelope(
            center=predictor, radius=radius, components=components,
            state={"beta": beta, "covariance": covariance, "scale_squared": critical**2},
            metadata={"construction": "linear_wild_bootstrap_transfer", "critical": critical,
                      "residuals": "raw", "multipliers": "Rademacher",
                      **({} if metadata is None else metadata)},
        )

    def transferred_contains(self, beta, predictor_coeff, truth_coeff, critical):
        """Check all-real transferred containment, using exact fallback near numerical boundaries."""
        p = self.design.shape[1]
        beta = values(beta, p, "coefficients")
        predictor_coeff = values(predictor_coeff, p, "predictor coefficients")
        truth_coeff = values(truth_coeff, p, "truth coefficients")
        critical = _critical(critical)
        if np.array_equal(predictor_coeff, truth_coeff):
            return True
        # Extended precision reduces cancellation before the boundary guard;
        # exact fallback treats the stored floats as rationals, as linear.py does.
        dtype = np.longdouble
        contexts = self._contexts.astype(dtype)
        h = predictor_coeff.astype(dtype) - truth_coeff.astype(dtype)
        g = predictor_coeff.astype(dtype) - beta.astype(dtype)
        h0, g0 = contexts @ h[:-1], contexts @ g[:-1]
        covariance = self.covariance.astype(dtype)
        q0 = np.einsum("ij,jk,ik->i", contexts, covariance[:-1, :-1], contexts)
        q1 = 2 * (contexts @ covariance[:-1, -1])
        q2 = covariance[-1, -1]
        scale = dtype(float(critical**2))
        for index in range(len(contexts)):
            result = _fiber_contains((h0[index], h[-1]), (g0[index], g[-1]),
                                     scale * np.array([q0[index], q1[index], q2], dtype=dtype))
            if result is None:
                envelope = self.envelope(LinearModel(predictor_coeff), beta, critical)
                exact = check_linear_containment(
                    envelope, LinearModel(truth_coeff), TargetDomain(self.X[index:index+1]),
                )
                result = exact.status == "covered"
            if not result:
                return False
        return True


def _critical(value):
    value = float(value)
    if (not np.isfinite(value) or value < 0
            or value > np.sqrt(np.finfo(float).max)):
        raise ValueError("Critical value must have a finite nonnegative square.")
    return value


def _fiber_contains(h, g, q):
    # On each sign region, |h|-|g|=k is affine. Only k>=0 needs a
    # comparison: q-k^2>=0 on the intersection of three affine half-lines.
    for sign_h in (-1, 1):
        for sign_g in (-1, 1):
            first = sign_h * np.asarray(h)
            second = sign_g * np.asarray(g)
            gap = first - second
            lower, upper = np.longdouble(-np.inf), np.longdouble(np.inf)
            empty = False
            for constant, slope in (first, second, gap):
                if slope == 0:
                    if constant < 0:
                        empty = True
                        break
                elif slope > 0:
                    lower = max(lower, -constant / slope)
                else:
                    upper = min(upper, -constant / slope)
            if empty or lower > upper:
                continue
            if np.all(gap == 0):
                continue  # Zero deficit is covered even when critical=0.
            square = np.array([gap[0]**2, 2 * gap[0] * gap[1], gap[1]**2])
            result = _quadratic_nonnegative(q - square, lower, upper, np.abs(q) + np.abs(square))
            if result is not True:
                return result
    return True


def _quadratic_nonnegative(coefficients, lower, upper, scales):
    c, b, a = coefficients
    tolerance = 256 * np.finfo(float).eps
    # Leading-coefficient cancellation changes tail behavior; do not guess.
    if abs(a) <= tolerance * scales[2]:
        return None
    if a < 0 and (np.isneginf(lower) or np.isposinf(upper)):
        return False
    candidates = [bound for bound in (lower, upper) if np.isfinite(bound)]
    if a > 0:
        vertex = -b / (2 * a)
        if lower <= vertex <= upper:
            candidates.append(vertex)
    for point in candidates:
        result = (a * point + b) * point + c
        error = tolerance * (scales[2] * point**2 + scales[1] * abs(point) + scales[0])
        if not np.isfinite(result) or abs(result) <= error:
            return None
        if result < 0:
            return False
    return True
