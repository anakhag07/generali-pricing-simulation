"""Tests for GLM sensitivity bucket construction."""

from __future__ import annotations

import numpy as np
import pytest


def test_split_sensitivity_tertiles_orders_low_medium_high() -> None:
    from experiments.sensitivity_buckets import split_sensitivity_tertiles

    row_indices = np.array([10, 11, 12, 13, 14, 15])
    scores = np.array([0.6, 0.1, 0.4, 0.2, 0.3, 0.5])

    low, medium, high = split_sensitivity_tertiles(row_indices, scores)

    assert low.name == "low"
    assert medium.name == "medium"
    assert high.name == "high"
    assert low.row_indices.tolist() == [11, 13]
    assert medium.row_indices.tolist() == [14, 12]
    assert high.row_indices.tolist() == [15, 10]
    assert np.max(low.scores) <= np.min(medium.scores)
    assert np.max(medium.scores) <= np.min(high.scores)


def test_split_sensitivity_tertiles_rejects_bad_inputs() -> None:
    from experiments.sensitivity_buckets import split_sensitivity_tertiles

    with pytest.raises(ValueError, match="same length"):
        split_sensitivity_tertiles(np.array([1, 2, 3]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="at least three"):
        split_sensitivity_tertiles(np.array([1, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="finite"):
        split_sensitivity_tertiles(np.array([1, 2, 3]), np.array([0.1, np.nan, 0.2]))


def test_glm_price_sensitivity_scores_are_finite() -> None:
    from data.loader import load_model_artifacts, load_x_frame, sample_csv_row_indices
    from experiments.sensitivity_buckets import glm_price_sensitivity_scores

    row_indices = sample_csv_row_indices("glm", n_rows=20, seed=123)
    x_frame = load_x_frame("glm", row_indices=row_indices)
    acceptance_model, _ = load_model_artifacts("glm")

    scores = glm_price_sensitivity_scores(acceptance_model, x_frame, u_ref=0.0)

    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)


def test_glm_price_sensitivity_matrix_matches_point_scores() -> None:
    from data.loader import load_model_artifacts, load_x_frame, sample_csv_row_indices
    from experiments.sensitivity_buckets import (
        glm_price_derivative_matrix,
        glm_price_sensitivity_matrix,
        glm_price_sensitivity_scores,
    )

    row_indices = sample_csv_row_indices("glm", n_rows=20, seed=456)
    x_frame = load_x_frame("glm", row_indices=row_indices)
    acceptance_model, _ = load_model_artifacts("glm")
    u_values = np.array([-0.3, 0.0, 0.3], dtype=float)

    matrix = glm_price_sensitivity_matrix(acceptance_model, x_frame, u_values=u_values)
    derivative_matrix = glm_price_derivative_matrix(
        acceptance_model,
        x_frame,
        u_values=u_values,
    )

    assert matrix.shape == (20, 3)
    assert derivative_matrix.shape == (20, 3)
    assert np.isfinite(matrix).all()
    assert np.isfinite(derivative_matrix).all()
    assert np.all(matrix >= 0.0)
    np.testing.assert_allclose(matrix, np.abs(derivative_matrix))
    for idx, u_ref in enumerate(u_values):
        point_scores = glm_price_sensitivity_scores(
            acceptance_model,
            x_frame,
            u_ref=float(u_ref),
        )
        np.testing.assert_allclose(matrix[:, idx], point_scores)


def test_glm_price_sensitivity_matrix_rejects_bad_u_values() -> None:
    from data.loader import load_model_artifacts, load_x_frame, sample_csv_row_indices
    from experiments.sensitivity_buckets import glm_price_derivative_matrix

    row_indices = sample_csv_row_indices("glm", n_rows=3, seed=789)
    x_frame = load_x_frame("glm", row_indices=row_indices)
    acceptance_model, _ = load_model_artifacts("glm")

    with pytest.raises(ValueError, match="at least one"):
        glm_price_derivative_matrix(acceptance_model, x_frame, u_values=[])
    with pytest.raises(ValueError, match="finite"):
        glm_price_derivative_matrix(acceptance_model, x_frame, u_values=[0.0, np.nan])


def test_build_glm_sensitivity_buckets_covers_rows() -> None:
    from data.loader import sample_csv_row_indices
    from experiments.sensitivity_buckets import build_glm_sensitivity_buckets

    row_indices = sample_csv_row_indices("glm", n_rows=30, seed=321)
    buckets = build_glm_sensitivity_buckets(row_indices=row_indices, u_ref=0.0)

    assert [bucket.name for bucket in buckets] == ["low", "medium", "high"]
    assert [bucket.row_indices.size for bucket in buckets] == [10, 10, 10]
    combined = np.concatenate([bucket.row_indices for bucket in buckets])
    assert set(combined.tolist()) == set(row_indices.tolist())
    assert np.max(buckets[0].scores) <= np.min(buckets[1].scores)
    assert np.max(buckets[1].scores) <= np.min(buckets[2].scores)
