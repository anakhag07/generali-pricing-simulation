"""Replayable adapters for existing customer-local and action-only support widths."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from data.coverage import normalized_coverage_widths
from objective.gridded import interpolate_customer_curves
from .core import FittedEnvelope, locations


def fit_support_band(X, action_grid, predictor_grid, settings, *, support=None,
                     support_width=None, acceptance_grid=None, row_indices=None):
    """Bind an existing support construction to explicit customer rows and action bounds."""
    allowed = {"method", "construction", "scale", "quantity"}
    if set(settings)-allowed or settings.get("method", "support_grid") != "support_grid":
        raise ValueError("Unsupported support-band settings.")
    mode = settings["construction"]
    if mode == "normalized_local":
        if support is None:
            raise ValueError("normalized_local requires a customer-by-action support matrix.")
        widths = normalized_coverage_widths(support, scale=settings["scale"])
        interpolation = "linear"
    elif mode == "absolute_action":
        widths = np.asarray(support_width, dtype=float)
        if widths.ndim != 1 or np.any(widths <= 0):
            raise ValueError("absolute_action requires a positive one-dimensional support_width.")
        interpolation = "natural_cubic"
    else:
        raise ValueError("Specify normalized_local or absolute_action support construction.")
    state = {"X": np.asarray(X), "action_grid": np.asarray(action_grid),
             "predictor_grid": np.asarray(predictor_grid), "support_width": widths}
    if support is not None:
        state["support"] = np.asarray(support)
    if acceptance_grid is not None:
        state["acceptance_grid"] = np.asarray(acceptance_grid)
    if row_indices is not None:
        state["row_indices"] = np.asarray(row_indices)
    return support_from_state(state, {"method": "support_grid", "settings": dict(settings),
        "construction": mode, "interpolation": interpolation,
        "domain": "saved_contexts_cross_bounded_action_interval", "calibration": "descriptive_support_penalty",
        "quantity": settings["quantity"], "probability_guarantee": None})


def support_from_state(state, metadata):
    """Restore support endpoints with exact row matching and explicit interpolation semantics."""
    state = {key: np.array(value, copy=True) for key, value in state.items()}
    X, _ = locations(state["X"], np.zeros(len(state["X"])))
    grid, curves, widths = (state[key] for key in ("action_grid", "predictor_grid", "support_width"))
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Support action_grid must be finite and strictly increasing with at least two points.")
    if curves.shape != (len(X), len(grid)):
        raise ValueError("Predictor curves must align with every saved context and action.")
    interpolate_customer_curves(curves, grid, np.full(len(X), grid[0]))
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError("Support widths must be finite and nonnegative.")
    keys = {tuple(row): i for i, row in enumerate(X)}
    if len(keys) != len(X):
        raise ValueError("Support contexts must identify distinct rows; ambiguous duplicate contexts are unsupported.")
    cubic = metadata["interpolation"] == "natural_cubic"
    if cubic:
        if widths.shape != grid.shape:
            raise ValueError("Action-only width must align with action_grid.")
        spline = CubicSpline(grid, widths, bc_type="natural", extrapolate=False)
        # Cubic interpolation may overshoot between knots: validate every local
        # extremum before using a width, without selecting any optimizer action.
        probes = np.r_[grid, spline.derivative().roots(extrapolate=False)]
        if np.any(spline(probes) < 0):
            raise ValueError("Natural-cubic support width becomes negative between knots.")
    elif metadata["interpolation"] != "linear" or widths.shape != curves.shape:
        raise ValueError("Local widths require the customer-by-action shape and linear interpolation.")
    if "acceptance_grid" in state:
        acceptance = state["acceptance_grid"]
        if acceptance.shape != curves.shape or not np.isfinite(acceptance).all() or np.any((acceptance < 0)|(acceptance > 1)):
            raise ValueError("Acceptance grid must align with predictor rows and lie in [0,1].")

    def rows(X, actions):
        if np.any((actions < grid[0]) | (actions > grid[-1])):
            raise ValueError("Support-band query is outside its declared action interval.")
        try:
            return np.array([keys[tuple(row)] for row in X])
        except KeyError as error:
            raise ValueError("Support-band query does not match a saved customer row.") from error

    def evaluate(X, actions):
        indices = rows(X, actions)
        center, slope = interpolate_customer_curves(curves[indices], grid, actions)
        if cubic:
            width, width_slope = spline(actions), spline(actions, 1)
        else:
            width, width_slope = interpolate_customer_curves(widths[indices], grid, actions)
        return center, slope, width, width_slope

    def components(X, actions):
        parts = {"support_width": evaluate(X, actions)[2]}
        if "acceptance_grid" in state:
            acceptance, derivative = interpolate_customer_curves(state["acceptance_grid"][rows(X, actions)], grid, actions)
            parts.update(acceptance=acceptance, acceptance_derivative=derivative)
        return parts

    def derivatives(X, actions):
        _, center, _, width = evaluate(X, actions)
        return {"center": center, "lower": center-width, "upper": center+width}

    return FittedEnvelope(lambda X, a: evaluate(X, a)[0], lambda X, a: evaluate(X, a)[2],
                          state=state, metadata=dict(metadata), components=components,
                          action_derivatives=derivatives)


__all__ = ["fit_support_band", "support_from_state"]
