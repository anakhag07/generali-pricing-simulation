"""Reusable customer/action support computations for real-data analyses."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DEFAULT_ACTION_BANDWIDTH = 0.01
DEFAULT_NUMERIC_CLIP = 6.0


def mixed_customer_embedding(
    artifact: Any,
    frame: pd.DataFrame,
    *,
    numeric_clip: float = DEFAULT_NUMERIC_CLIP,
) -> np.ndarray:
    """Map raw rows to whitened numeric and exact-match categorical coordinates."""
    processor = artifact.preprocessor
    if processor is None:
        raise ValueError("Customer embedding requires an artifact preprocessor.")
    transformed = processor.transform(frame.loc[:, list(artifact.x_feature_cols)])
    numeric = transformed.loc[:, list(processor.numeric_feature_names_)].to_numpy(
        dtype=float
    )
    numeric = np.clip(numeric, -float(numeric_clip), float(numeric_clip))
    categorical = pd.get_dummies(
        frame.loc[:, list(processor.categorical_cols_)].astype("string"),
        dtype=float,
    ).to_numpy(dtype=float)
    categorical /= np.sqrt(2.0)
    return np.column_stack([numeric, categorical]).astype(np.float32)


def local_support_matrix(
    embedding: np.ndarray,
    observed_u: Sequence[float],
    u_grid: Sequence[float],
    *,
    n_neighbors: int,
    n_jobs: int = -1,
    action_bandwidth: float = DEFAULT_ACTION_BANDWIDTH,
    block_size: int = 200,
) -> np.ndarray:
    """Estimate Gaussian-kernel support over customer state and action."""
    coordinates = np.asarray(embedding, dtype=np.float32)
    historical = np.asarray(observed_u, dtype=np.float32)
    actions = np.asarray(u_grid, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("embedding must contain at least two customer rows.")
    if historical.shape != (len(coordinates),) or not np.isfinite(historical).all():
        raise ValueError("observed_u must contain one finite value per customer.")
    if actions.ndim != 1 or actions.size < 2 or not np.isfinite(actions).all():
        raise ValueError("u_grid must contain at least two finite actions.")
    if n_neighbors < 1 or action_bandwidth <= 0.0 or block_size < 1:
        raise ValueError("neighbors, action bandwidth, and block size must be positive.")

    neighbor_count = min(int(n_neighbors) + 1, len(coordinates))
    nearest = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="brute",
        metric="euclidean",
        n_jobs=int(n_jobs),
    ).fit(coordinates)
    distances, indices = nearest.kneighbors(coordinates)
    distances = distances[:, 1:].astype(np.float32)
    indices = indices[:, 1:]
    historical_neighbor_u = historical[indices]
    bandwidth_index = min(max(neighbor_count // 2 - 1, 0), distances.shape[1] - 1)
    state_bandwidth = np.maximum(distances[:, bandwidth_index], 1e-6)
    state_weights = np.exp(
        -0.5 * (distances / state_bandwidth[:, None]) ** 2
    ).astype(np.float32)

    support = np.empty((len(coordinates), len(actions)), dtype=np.float32)
    for start in range(0, len(coordinates), int(block_size)):
        stop = min(start + int(block_size), len(coordinates))
        action_distance = (
            historical_neighbor_u[start:stop, :, None] - actions[None, None, :]
        ) / float(action_bandwidth)
        action_weights = np.exp(-0.5 * action_distance**2)
        support[start:stop] = np.sum(
            state_weights[start:stop, :, None] * action_weights,
            axis=1,
        )
    return support


def normalized_coverage_widths(support: np.ndarray, *, scale: float) -> np.ndarray:
    """Convert each customer's support curve to a penalty in ``[0, scale]``."""
    values = np.asarray(support, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("support must be a 2D customer-by-action array.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("support must contain finite nonnegative values.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")
    row_max = np.max(values, axis=1, keepdims=True)
    if np.any(row_max <= 0.0):
        raise ValueError("Every customer must have positive support somewhere.")
    relative = values / row_max
    return float(scale) * (1.0 - relative)


__all__ = [
    "DEFAULT_ACTION_BANDWIDTH",
    "DEFAULT_NUMERIC_CLIP",
    "local_support_matrix",
    "mixed_customer_embedding",
    "normalized_coverage_widths",
]
