"""Contextual linear construction and exact containment adapters (MATH §7.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular

from experiments.bootstrap_band_continuous import VARIABLE, nonnegative_on_real_line
from experiments.offline_bootstrap_math import strict_conjunction_empty

from .core import CalibrationData, ContainmentResult, FittedEnvelope, ProbabilityStatement, TargetDomain


def features(X: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return the contextual affine design with rows $$(1,x_i',a_i).$$"""
    X, actions = np.asarray(X, dtype=float), np.asarray(actions, dtype=float)
    if X.ndim != 2 or actions.shape != (len(X),):
        raise ValueError("Contexts must be a matrix and actions one value per row.")
    if not np.isfinite(X).all() or not np.isfinite(actions).all():
        raise ValueError("Contexts and actions must be finite.")
    return np.column_stack((np.ones(len(X)), X, actions))


@dataclass(frozen=True)
class LinearModel:
    r"""Evaluate $$f(x,a)=\beta_0+\beta_x'x+\beta_a a$$ with fixed coefficients."""

    coefficients: np.ndarray

    def __post_init__(self):
        coefficients = np.array(self.coefficients, dtype=float, copy=True)
        if coefficients.ndim != 1 or len(coefficients) < 2 or not np.isfinite(coefficients).all():
            raise ValueError("At least two finite affine coefficients are required.")
        coefficients.flags.writeable = False
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def context_dim(self) -> int:
        """Return the number of contextual coordinates."""
        return len(self.coefficients) - 2

    def __call__(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Evaluate one response per context/action pair."""
        design = features(X, actions)
        if design.shape[1] != len(self.coefficients):
            raise ValueError("Linear coefficient and context dimensions disagree.")
        return design @ self.coefficients


def _settings(settings):
    delta = float(settings["delta"])
    sigma = float(settings["sigma_bound"])
    multiplier = float(settings.get("width_multiplier", 1.0))
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_bound must be finite and positive.")
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("width_multiplier must be finite and nonnegative.")
    return delta, sigma, multiplier


def _design(data, target):
    design = features(data.X, data.actions)
    n, p = design.shape
    if n < p or np.linalg.matrix_rank(design) != p:
        raise ValueError("Calibration design must have full column rank and n >= d+2.")
    if np.shape(data.y) != (n,) or not np.isfinite(data.y).all():
        raise ValueError("Calibration responses must be finite, with one value per row.")
    if target.contexts.ndim != 2 or target.contexts.shape[1] != p - 2:
        raise ValueError("Calibration and target context dimensions disagree.")
    return design


def fit_linear_transfer(
    data: CalibrationData, predictor, target: TargetDomain, settings: dict, rng,
) -> FittedEnvelope:
    """Fit OLS and transfer its known-variance width to a frozen predictor."""
    delta, sigma, multiplier = _settings(settings)
    design = _design(data, target)
    orthogonal, triangular = np.linalg.qr(design, mode="reduced")
    beta = solve_triangular(triangular, orthogonal.T @ data.y)
    inverse = solve_triangular(triangular, np.eye(design.shape[1]))
    covariance = inverse @ inverse.T
    scale_squared = (sigma * multiplier)**2 * design.shape[1] / delta
    if not np.isfinite(scale_squared) or not np.isfinite(covariance).all():
        raise ValueError("The represented width must have finite scale and covariance.")
    auxiliary_model = LinearModel(beta)

    def components(X, actions):
        phi = features(X, actions)
        if phi.shape[1] != len(beta):
            raise ValueError("Envelope and evaluation context dimensions disagree.")
        auxiliary = auxiliary_model(X, actions)
        # Use the saved covariance so evaluation and the certificate represent
        # the same formula. Floating-point evaluation error is not enclosed.
        leverage = np.einsum("ij,jk,ik->i", phi, covariance, phi)
        if not np.isfinite(leverage).all() or np.any(leverage < 0):
            raise ValueError("Numerical leverage evaluation is negative or nonfinite.")
        width = np.sqrt(scale_squared * leverage)
        discrepancy = np.abs(np.asarray(predictor(X, actions), dtype=float) - auxiliary)
        return {"auxiliary": auxiliary, "width": width, "discrepancy": discrepancy}

    def radius(X, actions):
        values = components(X, actions)
        return values["discrepancy"] + values["width"]

    return FittedEnvelope(
        center=predictor, radius=radius,
        state={"beta": beta, "covariance": covariance, "scale_squared": scale_squared},
        metadata={"construction": "linear_known_variance_transfer", "delta": delta,
                  "sigma_bound": sigma, "width_multiplier": multiplier},
        components=components,
    )


def _rational(value):
    return sp.Rational(float(value))


def _affine_polynomial(coefficients, context):
    coefficients = [_rational(value) for value in coefficients]
    intercept = coefficients[0] + sum(
        coefficient * _rational(value)
        for coefficient, value in zip(coefficients[1:-1], context)
    )
    return sp.Poly(intercept + coefficients[-1] * VARIABLE, VARIABLE, domain=sp.QQ)


def _squared_width(covariance, scale_squared, context):
    phi = sp.Matrix([sp.Integer(1), *map(_rational, context), VARIABLE])
    return sp.Poly(_rational(scale_squared) * (phi.T * covariance * phi)[0], VARIABLE, domain=sp.QQ)


def _above_branches_empty(branches, radius_squared):
    inequalities = []
    for branch in branches:
        inequalities.extend((branch, branch**2 - radius_squared))
    return strict_conjunction_empty(inequalities)


def check_linear_containment(
    envelope: FittedEnvelope, truth, target: TargetDomain,
) -> ContainmentResult:
    """Decide transferred-envelope containment on every specified all-real fiber."""
    if target.kind != "real_line":
        raise ValueError("The affine all-real checker requires a real_line target.")
    if not isinstance(truth, LinearModel) or not isinstance(envelope.center, LinearModel):
        raise ValueError("The affine all-real checker requires LinearModel truth and predictor.")
    if envelope.metadata.get("construction") != "linear_known_variance_transfer":
        raise ValueError("The affine all-real checker requires the linear transfer construction.")
    p = target.contexts.shape[1] + 2
    beta = np.asarray(envelope.state["beta"], dtype=float)
    covariance = np.asarray(envelope.state["covariance"], dtype=float)
    scale_squared = float(envelope.state["scale_squared"])
    if (len(truth.coefficients) != p or len(envelope.center.coefficients) != p
            or beta.shape != (p,) or covariance.shape != (p, p)):
        raise ValueError("Fit, model, and target dimensions disagree.")
    if (not np.isfinite(beta).all() or not np.isfinite(covariance).all()
            or not np.isfinite(scale_squared) or scale_squared < 0):
        raise ValueError("Saved coefficients and covariance must be finite with nonnegative scale.")
    exact_covariance = sp.Matrix([list(map(_rational, row)) for row in covariance])
    if exact_covariance != exact_covariance.T or any(
        exact_covariance[:size, :size].det() <= 0 for size in range(1, p + 1)
    ):
        raise ValueError("Represented covariance must be symmetric positive definite.")

    contexts = []
    for index, context in enumerate(target.contexts):
        fitted = _affine_polynomial(beta, context)
        center = _affine_polynomial(envelope.center.coefficients, context)
        reference = _affine_polynomial(truth.coefficients, context)
        q = _squared_width(exact_covariance, scale_squared, context)
        auxiliary_covered = nonnegative_on_real_line(q - (fitted - reference)**2)
        branches = (fitted, 2 * center - fitted)
        # The auxiliary event suffices, but its failure says nothing by itself
        # about the final envelope. Test both exact transfer branches then.
        covered = auxiliary_covered or (
            _above_branches_empty([branch - reference for branch in branches], q)
            and _above_branches_empty([reference - branch for branch in branches], q)
        )
        contexts.append({"context_index": index, "covered": bool(covered),
                         "auxiliary_covered": bool(auxiliary_covered),
                         "width_squared_coefficients": [str(q.nth(i)) for i in range(3)]})
    return ContainmentResult(
        status="covered" if all(row["covered"] for row in contexts) else "violated",
        method="affine_transfer_all_real", domain="real_line",
        details={"context_count": len(contexts), "contexts": contexts,
                 "auxiliary_covered": all(row["auxiliary_covered"] for row in contexts),
                 "arithmetic": "Exact rational polynomial signs for stored floating-point coefficients and covariance; does not enclose floating-point fitting or evaluation error."},
    )


def check_constant_linear_containment(
    envelope: FittedEnvelope, truth, target: TargetDomain,
) -> ContainmentResult:
    """Certify a literal constant radius for affine truth/predictor on every real action."""
    if target.kind != "real_line":
        raise ValueError("The constant affine checker requires a real_line target.")
    if not isinstance(truth, LinearModel) or not isinstance(envelope.center, LinearModel):
        raise ValueError("The constant affine checker requires LinearModel truth and predictor.")
    if envelope.metadata.get("construction") != "constant_radius":
        raise ValueError("The constant affine checker requires the literal constant_radius construction.")
    radius = float(envelope.state["radius"])
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Constant radius must be finite and nonnegative.")
    p = target.contexts.shape[1] + 2
    if len(truth.coefficients) != p or len(envelope.center.coefficients) != p:
        raise ValueError("Model and target context dimensions disagree.")
    exact_radius = _rational(radius)
    slope = _rational(envelope.center.coefficients[-1]) - _rational(truth.coefficients[-1])
    details = {
        "context_count": len(target.contexts), "radius_exact": str(exact_radius),
        "action_slope_difference": str(slope), "slope_mismatch": slope != 0,
        "arithmetic": "Exact rational comparison of stored floating-point coefficients, contexts, and constant radius; does not enclose floating-point evaluation error.",
    }
    if slope != 0:
        details["reason"] = "Unequal action slopes make the absolute prediction error exceed every finite constant radius in the tails."
        return ContainmentResult("violated", "constant_affine_all_real", "real_line", details)
    contexts = []
    for index, context in enumerate(target.contexts):
        difference = (_affine_polynomial(envelope.center.coefficients, context)
                      - _affine_polynomial(truth.coefficients, context))
        gap = abs(difference.nth(0))
        contexts.append({"context_index": index, "absolute_error_exact": str(gap),
                         "covered": bool(gap <= exact_radius)})
    details["contexts"] = contexts
    details["reason"] = "Equal action slopes leave a constant prediction error on each context fiber."
    return ContainmentResult(
        "covered" if all(row["covered"] for row in contexts) else "violated",
        "constant_affine_all_real", "real_line", details,
    )


def linear_guarantee(
    data: CalibrationData, truth, predictor, target: TargetDomain,
    settings: dict, noise_std: float,
) -> ProbabilityStatement:
    """State the conditional moment-bound guarantee after checking its model inputs."""
    delta, sigma, multiplier = _settings(settings)
    design = _design(data, target)
    if not isinstance(truth, LinearModel) or len(truth.coefficients) != design.shape[1]:
        raise ValueError("The linear moment bound requires a dimension-matched LinearModel truth.")
    if not np.isfinite(noise_std) or noise_std < 0 or sigma < noise_std:
        raise ValueError("sigma_bound must bound the actual Gaussian observation noise_std.")
    if multiplier < 1:
        raise ValueError("width_multiplier below one does not retain the linear moment guarantee.")
    predictions = np.asarray(predictor(data.X, data.actions), dtype=float)
    if predictions.shape != data.y.shape or not np.isfinite(predictions).all():
        raise ValueError("The frozen predictor must return one finite value per input row.")
    return ProbabilityStatement(
        kind="lower_bound", value=1 - delta,
        event="Simultaneous transferred-envelope containment at every target context and permitted action.",
        conditioning="Frozen calibration design and frozen predictor.",
        assumptions=(
            "Truth is globally affine in the declared contextual features and action.",
            "Calibration feature matrix has full column rank.",
            "Conditional observation errors have zero means, zero cross-covariances, and variances bounded by sigma_bound squared; this launch uses independent Gaussian errors.",
            "sigma_bound is known, not estimated from residuals.",
            "Exact-arithmetic OLS and width formula; floating-point fitting error is not enclosed.",
        ),
        derivation="E[e'Qe | Z] <= p*sigma_bound^2; Markov gives e'Qe <= p*sigma_bound^2/delta with probability >= 1-delta. Cauchy-Schwarz gives simultaneous auxiliary containment; the triangle inequality transfers it to the frozen predictor. No union bound over contexts or actions.",
    )
