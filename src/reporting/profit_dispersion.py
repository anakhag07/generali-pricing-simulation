"""Computation helpers for customer-level profit-dispersion curves."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from data.loader import ModelArtifactBundle
from data.monotone_spline_xgb import fit_monotone_churn_curve


U_GRID = np.linspace(0.0, 0.16, 161)
ANCHOR_U = np.linspace(0.0, 0.16, 17)
SPLINE_DENSE_GRID_SIZE = 500


@dataclass(frozen=True)
class ProfitDispersion:
    """Pointwise population summaries of customer profit."""

    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    mad: np.ndarray


def load_deterministic_sample_rows(
    diagnostics_path: str | Path,
    eligible_rows: Sequence[int],
    *,
    sample_size: int = 20_000,
    seed: int = 20260831,
) -> np.ndarray:
    """Load and verify the saved deterministic sample against its seed recipe."""
    path = Path(diagnostics_path)
    with np.load(path, allow_pickle=False) as diagnostics:
        if "row_indices" not in diagnostics:
            raise ValueError(f"{path} does not contain row_indices.")
        rows = np.asarray(diagnostics["row_indices"])

    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("row_indices must be a one-dimensional integer array.")
    rows = rows.astype(int, copy=False)
    if rows.size != int(sample_size) or np.unique(rows).size != rows.size:
        raise ValueError(
            f"Expected {int(sample_size):,} unique deterministic sample rows."
        )

    eligible = np.asarray(eligible_rows, dtype=int)
    if eligible.ndim != 1 or np.unique(eligible).size != eligible.size:
        raise ValueError("eligible_rows must be a one-dimensional array of unique rows.")
    if sample_size > eligible.size:
        raise ValueError("sample_size cannot exceed the number of eligible rows.")
    expected = np.sort(
        np.random.default_rng(int(seed)).choice(
            eligible,
            size=int(sample_size),
            replace=False,
        )
    )
    if not np.array_equal(rows, expected):
        raise ValueError(
            "Saved row_indices do not match the deterministic sampling recipe "
            f"for seed {int(seed)}."
        )
    return rows.copy()


def row_index_sha256(row_indices: Sequence[int]) -> str:
    """Return a platform-independent digest for a sequence of row positions."""
    rows = np.asarray(row_indices, dtype="<i8")
    if rows.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")
    return hashlib.sha256(rows.tobytes()).hexdigest()


def predict_acceptance_matrix(
    artifact: ModelArtifactBundle,
    frame: pd.DataFrame,
    u_values: Sequence[float],
) -> np.ndarray:
    """Evaluate an acceptance artifact for every customer and candidate action."""
    if artifact.probability_target != "acceptance":
        raise ValueError("Expected an artifact whose positive class is acceptance.")
    u = np.asarray(u_values, dtype=float)
    if u.ndim != 1 or u.size == 0 or not np.isfinite(u).all():
        raise ValueError("u_values must be a non-empty finite one-dimensional array.")

    raw = frame.copy()
    raw["U"] = 0.0
    model_frame = artifact.model_frame(raw)
    output = np.empty((len(frame), u.size), dtype=float)
    for index, proposed_u in enumerate(u):
        model_frame["U"] = float(proposed_u)
        probability = np.asarray(artifact.model.predict_proba(model_frame), dtype=float)
        output[:, index] = probability[:, 1]
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ValueError("Acceptance predictions must be finite probabilities.")
    return output


def predict_loss(artifact: ModelArtifactBundle, frame: pd.DataFrame) -> np.ndarray:
    """Evaluate one financial-loss artifact for every customer."""
    values = np.asarray(
        artifact.model.predict(artifact.model_frame(frame)),
        dtype=float,
    )
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("Loss predictions must be one finite value per customer.")
    return values


def _spline_acceptance_row(
    acceptance_at_anchors: np.ndarray,
    u_values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fitted = fit_monotone_churn_curve(
        ANCHOR_U,
        1.0 - acceptance_at_anchors,
        weights=weights,
        dense_grid_size=SPLINE_DENSE_GRID_SIZE,
    )
    return np.clip(1.0 - fitted.curve(u_values), 0.0, 1.0)


def exact_spline_acceptance_matrix(
    xgb_acceptance: ModelArtifactBundle,
    frame: pd.DataFrame,
    u_values: Sequence[float],
    weights: Sequence[float],
    *,
    n_jobs: int,
) -> np.ndarray:
    """Fit every spline from XGBoost anchors, with no raw-model fallback."""
    u = np.asarray(u_values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != ANCHOR_U.shape or np.any(weight_array < 0.0):
        raise ValueError("weights must contain one non-negative value per spline anchor.")
    if not np.isfinite(weight_array).all() or not np.any(weight_array > 0.0):
        raise ValueError("weights must be finite and contain a positive value.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs cannot be zero.")

    anchors = predict_acceptance_matrix(xgb_acceptance, frame, ANCHOR_U)

    def fit_one(index: int) -> np.ndarray:
        try:
            return _spline_acceptance_row(anchors[index], u, weight_array)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            raise RuntimeError(
                f"Exact spline fitting failed for sample position {index}; "
                "raw-XGBoost fallback is disabled."
            ) from error

    fitted = Parallel(n_jobs=int(n_jobs), prefer="threads")(
        delayed(fit_one)(index) for index in range(len(frame))
    )
    output = np.stack(fitted)
    if output.shape != (len(frame), u.size) or not np.isfinite(output).all():
        raise ValueError("Exact spline acceptance produced an invalid matrix.")
    return output


def customer_profit_matrix(
    acceptance: np.ndarray,
    premium: Sequence[float],
    predicted_loss: Sequence[float],
    u_values: Sequence[float],
) -> np.ndarray:
    """Return customer profit without negating the maximization quantity."""
    probability = np.asarray(acceptance, dtype=float)
    premium_array = np.asarray(premium, dtype=float)
    loss_array = np.asarray(predicted_loss, dtype=float)
    u = np.asarray(u_values, dtype=float)
    expected_shape = (premium_array.size, u.size)
    if probability.shape != expected_shape:
        raise ValueError(f"acceptance must have shape {expected_shape}.")
    if loss_array.shape != premium_array.shape:
        raise ValueError("predicted_loss must contain one value per premium.")
    if not all(
        np.isfinite(values).all()
        for values in (probability, premium_array, loss_array, u)
    ):
        raise ValueError("Profit inputs must be finite.")
    revenue = premium_array[:, None] * (1.0 + u[None, :])
    return probability * (revenue - loss_array[:, None])


def summarize_profit(profit: np.ndarray) -> ProfitDispersion:
    """Compute mean/population-SD and median/raw-MAD across customers."""
    values = np.asarray(profit, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("profit must be a non-empty two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("profit must contain only finite values.")
    median = np.median(values, axis=0)
    summary = ProfitDispersion(
        mean=np.mean(values, axis=0),
        std=np.std(values, axis=0, ddof=0),
        median=median,
        mad=np.median(np.abs(values - median[None, :]), axis=0),
    )
    if not all(np.isfinite(item).all() for item in summary.__dict__.values()):
        raise ValueError("Profit summaries must be finite.")
    return summary
