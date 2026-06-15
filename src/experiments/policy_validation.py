"""Reusable policy validation helpers independent of optimizer training."""

from __future__ import annotations

import numpy as np

from experiments.results import PolicyEvaluation
from objective.utils import _policy_value, value_for_reporting


def row_count(x_samples: object) -> int:
    """Return the number of rows in a 2D array-like or DataFrame batch."""
    return int(x_samples.shape[0])


def policy_u_values(
    objective: object,
    theta: np.ndarray,
    x_samples: object,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Return one policy action per row, using objective-owned preprocessing hooks."""
    theta_arr = np.asarray(theta, dtype=float)
    u_values = np.asarray(_policy_value(objective, theta_arr, x_samples), dtype=float).reshape(-1)
    if clip:
        clip_fn = getattr(objective, "_clip_u", None)
        if callable(clip_fn):
            u_values = np.asarray(clip_fn(u_values), dtype=float).reshape(-1)
    if u_values.shape != (row_count(x_samples),):
        raise ValueError("policy.value(theta, x_batch) must return one value per row.")
    return u_values


def evaluate_policy(objective: object, theta: np.ndarray, x_samples: object) -> PolicyEvaluation:
    """Evaluate a fixed theta policy on a supplied data batch."""
    theta_arr = np.asarray(theta, dtype=float)
    n_samples = row_count(x_samples)
    objective_value = value_for_reporting(objective, theta_arr, x_samples)
    u_values = policy_u_values(objective, theta_arr, x_samples)
    mean_acceptance_fn = getattr(objective, "mean_acceptance", None)
    mean_acceptance = (
        float(mean_acceptance_fn(theta_arr, x_samples)) if callable(mean_acceptance_fn) else None
    )
    projected_loss = None
    projected_revenue = None
    step_metrics_fn = getattr(objective, "_step_metrics", None)
    if callable(step_metrics_fn):
        step_metrics = step_metrics_fn(theta_arr, x_samples)
        if "projected_loss" in step_metrics:
            projected_loss = float(step_metrics["projected_loss"])
        if "projected_revenue" in step_metrics:
            projected_revenue = float(step_metrics["projected_revenue"])
        if mean_acceptance is None and "mean_acceptance" in step_metrics:
            mean_acceptance = float(step_metrics["mean_acceptance"])
    q25, q75 = np.quantile(u_values, [0.25, 0.75])
    return PolicyEvaluation(
        n_samples=n_samples,
        objective_value=objective_value,
        objective_sum=n_samples * objective_value,
        mean_u=float(np.mean(u_values)),
        u_q25=float(q25),
        u_q75=float(q75),
        mean_acceptance=mean_acceptance,
        projected_loss=projected_loss,
        projected_revenue=projected_revenue,
    )


__all__ = ["evaluate_policy", "policy_u_values", "row_count"]
