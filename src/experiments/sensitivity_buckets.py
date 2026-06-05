"""GLM price-sensitivity scoring and bucket construction for real-data runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from data.loader import (
    eligible_csv_row_indices,
    extract_glm_churn_coefficients,
    load_model_artifacts,
    load_observed_u_array,
    load_x_frame,
)
from objective._math import _sigmoid

SensitivityBucketName = Literal["low", "medium", "high"]
SENSITIVITY_BUCKETS: tuple[SensitivityBucketName, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class SensitivityBucket:
    """One tertile of customers ranked by local GLM price sensitivity."""

    name: SensitivityBucketName
    row_indices: np.ndarray
    scores: np.ndarray


def median_observed_u(model_type: str = "glm") -> float:
    """Return median observed historical pricing action over complete rows."""
    observed_u = load_observed_u_array(model_type, n_rows=None)
    return float(np.median(np.asarray(observed_u, dtype=float)))


def glm_price_sensitivity_scores(
    acceptance_model: Any,
    x_frame: pd.DataFrame,
    *,
    u_ref: float,
    u_coef: float | None = None,
) -> np.ndarray:
    r"""Return $$|d p_{accept}(x, u_ref) / du|$$ for GLM acceptance rows."""
    coeffs = extract_glm_churn_coefficients(acceptance_model)
    x_feature_cols = tuple(getattr(acceptance_model, "x_feature_cols", tuple(x_frame.columns)))
    raw_frame = x_frame.loc[:, list(x_feature_cols)].copy()
    raw_frame["U"] = float(u_ref)

    model_frame_fn = getattr(acceptance_model, "model_frame", None)
    model_frame = model_frame_fn(raw_frame) if callable(model_frame_fn) else raw_frame
    feature_names = list(coeffs["x_feature_names"])
    x_matrix = model_frame.loc[:, feature_names].to_numpy(dtype=float)
    beta_x = np.asarray(coeffs["x_coef"], dtype=float)
    beta_u = float(u_coef) if u_coef is not None else float(coeffs["u_coef"])
    logit = float(coeffs["intercept"]) + x_matrix @ beta_x + beta_u * float(u_ref)
    class1 = _sigmoid(logit)
    if coeffs.get("probability_target", getattr(acceptance_model, "probability_target", "acceptance")) == "acceptance":
        acceptance = class1
    else:
        acceptance = 1.0 - class1
    return np.abs(beta_u) * acceptance * (1.0 - acceptance)


def split_sensitivity_tertiles(
    row_indices: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[SensitivityBucket, SensitivityBucket, SensitivityBucket]:
    """Split row indices into low/medium/high score tertiles deterministically."""
    row_indices_arr = np.asarray(row_indices, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    if row_indices_arr.ndim != 1 or scores_arr.ndim != 1:
        raise ValueError("row_indices and scores must be 1D arrays.")
    if row_indices_arr.size != scores_arr.size:
        raise ValueError("row_indices and scores must have the same length.")
    if row_indices_arr.size < len(SENSITIVITY_BUCKETS):
        raise ValueError("Need at least three rows to form sensitivity tertiles.")
    if not np.isfinite(scores_arr).all():
        raise ValueError("scores must be finite.")

    order = np.argsort(scores_arr, kind="mergesort")
    groups = np.array_split(order, len(SENSITIVITY_BUCKETS))
    return tuple(
        SensitivityBucket(
            name=name,
            row_indices=row_indices_arr[group].copy(),
            scores=scores_arr[group].copy(),
        )
        for name, group in zip(SENSITIVITY_BUCKETS, groups)
    )  # type: ignore[return-value]


def build_glm_sensitivity_buckets(
    *,
    u_ref: float | None = None,
    row_indices: Sequence[int] | np.ndarray | None = None,
) -> tuple[SensitivityBucket, SensitivityBucket, SensitivityBucket]:
    """Load GLM rows, score local price sensitivity, and return tertile buckets."""
    resolved_row_indices = (
        eligible_csv_row_indices("glm") if row_indices is None else np.asarray(row_indices, dtype=int)
    )
    resolved_u_ref = median_observed_u("glm") if u_ref is None else float(u_ref)
    acceptance_model, _ = load_model_artifacts("glm")
    x_frame = load_x_frame("glm", row_indices=resolved_row_indices)
    scores = glm_price_sensitivity_scores(
        acceptance_model,
        x_frame,
        u_ref=resolved_u_ref,
    )
    return split_sensitivity_tertiles(resolved_row_indices, scores)


__all__ = [
    "SENSITIVITY_BUCKETS",
    "SensitivityBucket",
    "build_glm_sensitivity_buckets",
    "glm_price_sensitivity_scores",
    "median_observed_u",
    "split_sensitivity_tertiles",
]
