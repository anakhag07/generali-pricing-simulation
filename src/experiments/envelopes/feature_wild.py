"""Wild coefficient-ellipsoid calibration conditional on a frozen feature map."""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular


class WildFeatureDesign:
    """Cache full-rank QR geometry for the frozen-feature ball in MATH section 7.20."""

    def __init__(self, features):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Require a finite two-dimensional frozen design.")
        n, p = matrix.shape
        if not 0 < p < n:
            raise ValueError("Require n > p > 0.")
        self.design = matrix
        self.orthogonal, self.triangular = qr(matrix, mode="economic", check_finite=False)
        singular = np.linalg.svd(self.triangular, compute_uv=False)
        self.rank_tolerance = float(np.finfo(float).eps * max(n, p) * singular[0])
        if singular[-1] <= self.rank_tolerance:
            raise ValueError(f"Numerically rank-deficient frozen design: smallest singular value "
                             f"{singular[-1]:.6g} <= {self.rank_tolerance:.6g}.")
        self.condition_number = float(singular[0] / singular[-1])
        self.leverage = np.einsum("ij,ij->i", self.orthogonal, self.orthogonal)

    def fit(self, responses):
        """Fit the linear head conditional on its frozen features using cached QR."""
        y = np.asarray(responses, dtype=float)
        if y.shape != (len(self.design),) or not np.isfinite(y).all():
            raise ValueError("Require one finite response per design row.")
        return solve_triangular(self.triangular, self.orthogonal.T @ y)

    def bootstrap_errors(self, residuals, multipliers):
        """Return raw-residual wild coefficient errors and their joint-ball norms."""
        residuals, signs = np.asarray(residuals, dtype=float), np.asarray(multipliers)
        if residuals.shape != (len(self.design),) or not np.isfinite(residuals).all():
            raise ValueError("Require one finite fitted residual per design row.")
        if (signs.ndim != 2 or signs.shape[1] != len(residuals) or len(signs) == 0
                or not np.all((signs == -1) | (signs == 1))):
            raise ValueError("Require nonempty (B,n) Rademacher multipliers.")
        whitened = self.orthogonal.T @ (signs * residuals).T
        errors = solve_triangular(self.triangular, whitened).T
        return errors, np.linalg.norm(whitened, axis=0)


def feature_band_components(features, beta, triangular, critical):
    """Evaluate the exact frozen-feature residual fit and coefficient-ball width."""
    from .wild import residual_radii
    phi = np.asarray(features, dtype=float)
    beta, triangular = np.asarray(beta), np.asarray(triangular)
    if (phi.ndim != 2 or phi.shape[1:] != beta.shape
            or triangular.shape != (len(beta), len(beta))
            or not all(np.isfinite(x).all() for x in (phi, beta, triangular))
            or not np.isfinite(critical) or critical < 0):
        raise ValueError("Incompatible or nonfinite frozen-feature band state.")
    fitted = phi @ beta
    width = float(critical) * np.linalg.norm(
        solve_triangular(triangular.T, phi.T, lower=True), axis=0)
    lower, upper = residual_radii(fitted, width)
    return {"residual_fit": fitted, "width": width, "radius_lower": lower, "radius_upper": upper}
