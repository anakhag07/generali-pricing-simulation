"""Shared observed-pairs quadratic bands and exact polynomial certificates (MATH §7.1–7.2)."""
from __future__ import annotations
from typing import Any
import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular

def features(a: np.ndarray) -> np.ndarray:
    """Return the quadratic feature matrix with an intercept."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("Inputs must be a finite one-dimensional array.")
    return np.column_stack((np.ones_like(a), a, a**2))


def refit_pairs(design: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Refit original rows, preserving input/response pairs (MATH.md §7.1).

    Indices are supplied by the dedicated bootstrap stream. Singular resamples
    fail explicitly: silently dropping or redrawing them changes the bootstrap.
    """
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] != len(design)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(design))):
        raise ValueError("Require an integer (B, N) array of original row indices.")
    fits = np.empty((len(indices), design.shape[1]))
    for b, rows in enumerate(indices):
        fit, _, rank, _ = np.linalg.lstsq(design[rows], y[rows], rcond=None)
        if rank != design.shape[1]:
            raise ValueError(f"Pairs-bootstrap replicate {b} has a rank-deficient design.")
        fits[b] = fit
    return fits


def bootstrap_band(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, *,
    draws: int, delta: float, bootstrap_seed: int,
) -> dict[str, Any]:
    """Fit and calibrate a grid band using only observations, never truth."""
    design, evaluation = features(x), features(grid)
    y = np.asarray(y, dtype=float)
    n = len(design)
    if n <= 3 or y.shape != (n,) or not np.all(np.isfinite(y)):
        raise ValueError("OLS requires n > 3 and one finite response per input.")
    if draws < 2 or not 0 < delta < 1 or not len(evaluation):
        raise ValueError("Require draws >= 2, 0 < delta < 1, and a nonempty grid.")
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("Quadratic design is rank deficient.")
    q, triangular = np.linalg.qr(design, mode="reduced")
    beta = solve_triangular(triangular, q.T @ y)
    residual = y - design @ beta
    sigma_hat = float(np.sqrt(residual @ residual / (n - 3)))
    if sigma_hat <= np.finfo(float).eps * max(1.0, float(np.linalg.norm(y))):
        raise ValueError("Residual variance is numerically zero; standardized band is undefined.")
    # P'P = R'R; ||R^{-T}p(a)|| is the prediction SE divided by sigma_hat.
    transformed = solve_triangular(triangular.T, evaluation.T, lower=True)
    se = sigma_hat * np.linalg.norm(transformed, axis=0)
    fitted = evaluation @ beta
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, n, size=(draws, n))
    bootstrap_beta = refit_pairs(design, y, indices)
    # This maximum is the user-specified coverage statistic, not an action selector.
    maxima = np.max(np.abs((bootstrap_beta - beta) @ evaluation.T) / se, axis=1)
    critical = float(np.quantile(maxima, 1.0 - delta, method="higher"))
    radius = critical * se
    return {
        "x": np.asarray(x), "y": y, "grid": np.asarray(grid), "beta_hat": beta,
        "residuals": residual, "sigma_hat": sigma_hat, "sigma2_hat": sigma_hat**2,
        "design_condition_number": float(np.linalg.cond(design)),
        "bootstrap_beta": bootstrap_beta, "bootstrap_indices": indices,
        "bootstrap_maxima": maxima,
        "critical_value": critical, "fitted": fitted, "standard_error": se,
        "radius": radius, "lower": fitted - radius, "upper": fitted + radius,
    }


def evaluate_band(band: dict[str, Any], true_values: np.ndarray) -> dict[str, Any]:
    """Check the fitted band against synthetic truth at every grid point."""
    truth = np.asarray(true_values, dtype=float)
    if truth.shape != band["fitted"].shape or not np.all(np.isfinite(truth)):
        raise ValueError("Truth must provide one finite value per evaluation point.")
    standardized_error = np.abs(band["fitted"] - truth) / band["standard_error"]
    covered = standardized_error <= band["critical_value"]
    return {
        "truth": truth, "standardized_error": standardized_error,
        "observed_max_statistic": float(np.max(standardized_error)),
        "simultaneous_covered": bool(np.all(covered)),
        "fraction_grid_covered": float(np.mean(covered)),
        "mean_radius": float(np.mean(band["radius"])),
    }


VARIABLE = sp.Symbol("a")


def nonnegative_on_real_line(poly: sp.Poly) -> bool:
    """Decide global polynomial nonnegativity using exact real-root counts."""
    if poly.is_zero:
        return True
    if poly.LC() < 0 or poly.degree() % 2:
        return False
    if poly.degree() == 0 or poly.count_roots(-sp.oo, sp.oo) == 0:
        return True
    # A real root of odd multiplicity changes sign; even roots only touch zero.
    return all(
        multiplicity % 2 == 0 or factor.count_roots(-sp.oo, sp.oo) == 0
        for factor, multiplicity in poly.sqf_list()[1]
    )


class QuadraticErrorCertificate:
    """Exact threshold tests for |p(a)'d| / sqrt(q(a)), with positive quartic q."""

    def __init__(self, covariance: np.ndarray, sigma_hat: float):
        v = [[sp.Rational(float(entry)) for entry in row] for row in covariance]
        matrix = sp.Matrix(v)
        if matrix != matrix.T or any(matrix[:i, :i].det() <= 0 for i in (1, 2, 3)):
            raise ValueError("Represented covariance must be symmetric positive definite.")
        scale = sp.Rational(float(sigma_hat))**2
        if scale <= 0:
            raise ValueError("Positive residual scale is required.")
        self.q_coefficients = [scale * c for c in (
            v[0][0], 2*v[0][1], v[1][1] + 2*v[0][2], 2*v[1][2], v[2][2],
        )]
        self.q = sp.Poly.from_list(self.q_coefficients[::-1], VARIABLE, domain=sp.QQ)
        self.q_float = np.array(self.q_coefficients, dtype=float)

    @staticmethod
    def difference(first: np.ndarray, second: np.ndarray) -> sp.Poly:
        coefficients = [sp.Rational(float(a)) - sp.Rational(float(b)) for a, b in zip(first, second)]
        if len(coefficients) != 3:
            raise ValueError("Three quadratic coefficients are required.")
        return sp.Poly.from_list(coefficients[::-1], VARIABLE, domain=sp.QQ)

    def threshold_polynomial(self, square: sp.Poly, threshold: float) -> sp.Poly:
        return self.q.mul_ground(sp.Rational(float(threshold))**2) - square

    def contains(self, difference: sp.Poly, threshold: float) -> bool:
        """Certify the entire error curve lies inside threshold*sqrt(q(a))."""
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative.")
        return nonnegative_on_real_line(self.threshold_polynomial(difference**2, threshold))

    def supremum_interval(self, difference: sp.Poly, tolerance: float) -> tuple[float, float]:
        """Certify a tight interval for the all-real statistic, with no grid."""
        if difference.is_zero:
            return 0.0, 0.0
        d = np.array([float(difference.nth(i)) for i in range(3)])
        q = self.q_float
        from numpy.polynomial import polynomial as polynomial
        # Leading degree-five terms cancel identically for quadratic/quartic.
        stationary = polynomial.polysub(
            2 * polynomial.polymul(polynomial.polyder(d), q),
            polynomial.polymul(d, polynomial.polyder(q)),
        )[:5]
        roots = polynomial.polyroots(stationary)
        candidates = [0.0] + [float(r.real) for r in roots if abs(r.imag) <= 1e-7 * max(1, abs(r.real))]
        values = [abs(d[2]) / np.sqrt(q[4])]  # Same limit in both tails.
        for a in candidates:
            # Homogeneous evaluation avoids overflow in far-tail stationary points.
            if abs(a) > 1:
                numerator = polynomial.polyval(1/a, d[::-1])
                denominator = polynomial.polyval(1/a, q[::-1])
            else:
                numerator = polynomial.polyval(a, d)
                denominator = polynomial.polyval(a, q)
            if denominator > 0:
                values.append(abs(numerator) / np.sqrt(denominator))
        proposal = float(max(values))
        if not np.isfinite(proposal):
            raise ValueError("Nonfinite supremum proposal.")
        square = difference**2

        def below(t):
            return nonnegative_on_real_line(self.threshold_polynomial(square, t))

        padding = tolerance * max(1, proposal)
        low, high = max(0.0, proposal - padding), proposal + padding
        for _ in range(100):
            if below(high):
                break
            high = 2*high + padding
        else:
            raise RuntimeError("Could not certify an upper supremum bound.")
        for _ in range(100):
            if low == 0 or not below(low):
                break
            low /= 2
        else:
            raise RuntimeError("Could not certify a lower supremum bound.")
        for _ in range(100):
            if high-low <= 2*tolerance*max(1, high):
                return low, high
            middle = (low+high)/2
            if below(middle):
                high = middle
            else:
                low = middle
        raise RuntimeError("Supremum certificate did not reach its requested tolerance.")
