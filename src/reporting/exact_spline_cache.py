"""Validated cache construction for exact per-customer spline responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data.dataset_metadata import PREMIUM_COL
from experiments.provenance import array_sha256
from reporting.profit_dispersion import (
    exact_spline_acceptance_matrix,
    predict_loss,
    spline_anchor_weights,
)


def load_or_build_exact_spline_response_grid(
    *,
    cache_path: str | Path,
    frame: pd.DataFrame,
    row_indices: Sequence[int],
    eligible_rows: Sequence[int],
    action_grid: Sequence[float],
    acceptance_artifact: Any,
    loss_artifact: Any,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an exact matching cache or build acceptance and objective-cost grids."""
    path = Path(cache_path)
    rows = np.asarray(row_indices, dtype=int)
    eligible = np.asarray(eligible_rows, dtype=int)
    actions = np.asarray(action_grid, dtype=float)
    weights = spline_anchor_weights(eligible)
    identity = {
        "row_indices_sha256": array_sha256(rows.astype("<i8")),
        "eligible_rows_sha256": array_sha256(eligible.astype("<i8")),
        "action_grid_sha256": array_sha256(actions.astype("<f8")),
        "spline_weights_sha256": array_sha256(weights.astype("<f8")),
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            cached_identity = {
                key: str(cached[key].item()) if key in cached else ""
                for key in identity
            }
            acceptance = cached["acceptance"].astype(float)
            cost = cached["cost"].astype(float)
        if cached_identity == identity and acceptance.shape == cost.shape == (
            len(rows),
            len(actions),
        ):
            return acceptance, cost

    acceptance = exact_spline_acceptance_matrix(
        acceptance_artifact,
        frame,
        actions,
        weights,
        n_jobs=int(n_jobs),
    )
    loss = predict_loss(loss_artifact, frame)
    premium = frame[PREMIUM_COL].to_numpy(dtype=float)
    cost = acceptance * (
        loss[:, None] - premium[:, None] * (1.0 + actions[None, :])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        acceptance=acceptance.astype(np.float32),
        cost=cost.astype(np.float32),
        **{key: np.asarray(value) for key, value in identity.items()},
    )
    return acceptance, cost


__all__ = ["load_or_build_exact_spline_response_grid"]
