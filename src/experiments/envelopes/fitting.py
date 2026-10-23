"""Single-fit OLS/wild bands and numeric-state replay of established constructions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from experiments.provenance import file_sha256, read_json, write_json_atomic, save_arrays_atomic
from .core import CalibrationData, FittedEnvelope, TargetDomain
from .linear import LinearModel, QuadraticModel, features, fit_linear_transfer
from .wild import WildLinearDesign, residual_radii


def predictor_from_spec(spec):
    """Restore an explicit frozen affine or action-only quadratic predictor."""
    if spec.get("type") == "linear":
        return LinearModel(spec["coefficients"])
    if spec.get("type") == "quadratic":
        return QuadraticModel(spec["coefficients"])
    raise ValueError("Saved polynomial bands require a linear or quadratic predictor.")


def fit_band(data: CalibrationData, predictor, target: TargetDomain, settings, *, multipliers=None):
    """Fit an established band from observations and explicit bootstrap draws, never truth."""
    method = settings["method"]
    allowed = {"method", "delta", "sigma_bound"} if method != "residual_wild" else {"method", "delta", "B", "supremum_tolerance"}
    if set(settings)-allowed:
        raise ValueError(f"Unsupported {method} settings: {sorted(set(settings)-allowed)}")
    delta = float(settings["delta"])
    if not 0 < delta < 1:
        raise ValueError("delta must be strictly between zero and one.")
    if not isinstance(predictor, (LinearModel, QuadraticModel)):
        raise ValueError("Use an explicit frozen linear/quadratic predictor for replayable polynomial bands.")
    if target.kind != "real_line" or not np.array_equal(target.contexts, data.X):
        raise ValueError("These constructions require the calibration contexts and all real actions.")
    metadata = {"method": method, "settings": dict(settings), "predictor": predictor.to_dict(),
                "domain": "calibration_contexts_cross_real_line", "delta": delta,
                "mathematics": "MATH.md sections 7.8-7.11", "target": "function"}
    if method == "ols_transfer":
        if not isinstance(predictor, LinearModel):
            raise ValueError("ols_transfer requires the established contextual affine basis.")
        fitted = fit_linear_transfer(data, predictor, target, settings, None)
        state = dict(fitted.state)
        metadata.update(fitted.metadata, calibration="known_variance_bound")
    elif method in {"residual_ols", "residual_wild"}:
        degree = 2 if isinstance(predictor, QuadraticModel) else 1
        design = WildLinearDesign(data.X, data.actions, action_degree=degree,
                                 supremum_tolerance=settings.get("supremum_tolerance", 1e-8))
        predicted = predictor(data.X, data.actions)
        if method == "residual_wild":
            if multipliers is None:
                raise ValueError("residual_wild requires explicit Rademacher multipliers from the setting's method stream.")
            if "B" in settings and settings["B"] != len(multipliers):
                raise ValueError("B must equal the number of supplied multiplier rows.")
            beta, errors = design.bootstrap_errors(data.y, multipliers, predictor=predicted)
            bounds = design.statistic_bounds(errors)
            order = math.ceil(len(errors)*(1-delta))-1
            critical = float(np.partition(bounds[:, 1], order)[order])
            extra = {"multipliers": np.asarray(multipliers), "bootstrap_errors": errors,
                     "bootstrap_statistic_bounds": bounds}
            metadata.update(calibration="wild_bootstrap_approximation", residuals="raw",
                            multipliers="rademacher", quantile="order_statistic")
        else:
            sigma = float(settings["sigma_bound"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("sigma_bound must be finite and positive.")
            beta = design.fit(data.y-predicted)
            critical = sigma*np.sqrt(len(beta)/delta)
            extra = {}
            metadata.update(calibration="known_variance_bound")
        state = {"beta": beta, "covariance": design.covariance,
                 "triangular": design._triangular.copy(), "critical": critical, **extra}
        metadata.update(target="residual_transfer", action_degree=degree)
    else:
        raise ValueError(f"Unknown polynomial band method: {method}.")
    state.update(X=data.X, actions=data.actions, y=data.y)
    return band_from_state(state, metadata)


def band_from_state(state, metadata):
    """Restore evaluation from saved coefficients and geometry without fitting or RNG use."""
    if metadata.get("method") == "support_grid":
        from .support import support_from_state
        return support_from_state(state, metadata)
    predictor = predictor_from_spec(metadata["predictor"])
    state = {key: np.array(value, copy=True) for key, value in state.items()}
    beta = state["beta"]
    data = CalibrationData(state["X"], state["actions"], state["y"])
    context_keys = {tuple(row) for row in data.X}
    if any(not np.isfinite(value).all() for value in state.values()):
        raise ValueError("Saved polynomial band state must be finite.")
    if beta.ndim != 1 or state["covariance"].shape != (len(beta), len(beta)):
        raise ValueError("Saved band coefficients and geometry are incompatible.")
    method = metadata["method"]
    if method not in {"ols_transfer", "residual_ols", "residual_wild"}:
        raise ValueError(f"Unknown fitted band method: {method}.")

    def basis(X, actions):
        if any(tuple(row) not in context_keys for row in X):
            raise ValueError("Polynomial band queries must belong to its calibration-context domain.")
        phi = features(X, actions)
        if metadata.get("action_degree", 1) == 2:
            if X.shape[1] != 0:
                raise ValueError("Action-only quadratic band cannot accept contexts.")
            phi = np.column_stack((phi, actions**2))
        if phi.shape[1] != len(beta):
            raise ValueError("Band feature dimension differs from query inputs.")
        return phi

    def components(X, actions):
        phi = basis(X, actions)
        auxiliary = phi @ beta
        if method == "ols_transfer":
            leverage = np.einsum("ij,jk,ik->i", phi, state["covariance"], phi)
            if np.any(leverage < 0):
                raise ValueError("Negative represented leverage.")
            width = np.sqrt(float(state["scale_squared"])*leverage)
            return {"auxiliary": auxiliary, "width": width,
                    "discrepancy": np.abs(predictor(X, actions)-auxiliary)}
        width = float(state["critical"])*np.linalg.norm(
            solve_triangular(state["triangular"].T, phi.T, lower=True), axis=0)
        lower, upper = residual_radii(auxiliary, width)
        return {"residual_fit": auxiliary, "width": width,
                "radius_lower": lower, "radius_upper": upper}

    def derivatives(X, actions):
        phi = basis(X, actions)
        dphi = np.zeros_like(phi)
        dphi[:, X.shape[1]+1] = 1.
        if metadata.get("action_degree", 1) == 2:
            dphi[:, -1] = 2*actions
        center_grad = predictor.action_gradient(X, actions)
        parts = components(X, actions)
        if method == "ols_transfer":
            width_grad = float(state["scale_squared"])*np.einsum(
                "ij,jk,ik->i", dphi, state["covariance"], phi)
            width_grad = np.divide(width_grad, parts["width"], out=np.zeros_like(width_grad),
                                   where=parts["width"] > 0)
            radius_grad = width_grad + np.sign(predictor(X, actions)-parts["auxiliary"])*(center_grad-dphi @ beta)
            lower_grad, upper_grad = center_grad-radius_grad, center_grad+radius_grad
        else:
            z = solve_triangular(state["triangular"].T, phi.T, lower=True)
            dz = solve_triangular(state["triangular"].T, dphi.T, lower=True)
            width_grad = float(state["critical"])*np.sum(z*dz, axis=0)/np.linalg.norm(z, axis=0)
            lower_grad = center_grad - (parts["radius_lower"] > 0)*(width_grad-dphi @ beta)
            upper_grad = center_grad + (parts["radius_upper"] > 0)*(width_grad+dphi @ beta)
        return {"center": center_grad, "lower": lower_grad, "upper": upper_grad}

    kwargs = {"center": predictor, "components": components, "state": state,
              "metadata": {**metadata, "derivative_convention": "zero derivative for abs/positive-part at a kink"},
              "action_derivatives": derivatives}
    if method == "ols_transfer":
        return FittedEnvelope(**kwargs, radius=lambda X, a: components(X, a)["width"]+components(X, a)["discrepancy"])
    return FittedEnvelope(**kwargs, radii=lambda X, a: (
        components(X, a)["radius_lower"], components(X, a)["radius_upper"]))


def save_band(band, path):
    """Save verified numeric state and a relative array reference for deterministic replay."""
    path = Path(path)
    arrays_path = path.with_suffix(".npz")
    if path.exists() or arrays_path.exists():
        raise FileExistsError(f"Preserve the fitted band at {path}; choose a fresh output location.")
    # Validate that this state has a supported reader before writing anything.
    band_from_state(band.state, band.metadata)
    save_arrays_atomic(arrays_path, **band.state)
    write_json_atomic(path, {"schema": 1, "metadata": band.metadata,
                            "arrays": arrays_path.name, "sha256": file_sha256(arrays_path)})
    return path


def load_band(path):
    """Load a checksummed fitted band without regenerating observations or bootstrap draws."""
    path = Path(path)
    record = read_json(path)
    if record.get("schema") != 1:
        raise ValueError("Unsupported fitted-band artifact schema.")
    arrays_path = path.parent / record["arrays"]
    if file_sha256(arrays_path) != record["sha256"]:
        raise ValueError("Saved band arrays do not match their recorded hash.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return band_from_state(dict(arrays), record["metadata"])


__all__ = ["fit_band", "save_band", "load_band", "band_from_state", "predictor_from_spec"]
