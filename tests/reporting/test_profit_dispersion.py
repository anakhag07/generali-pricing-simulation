"""Tests for profit-dispersion computation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import ModelArtifactBundle
from reporting import profit_dispersion


class _AcceptanceModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        acceptance = np.clip(0.8 - frame["U"].to_numpy(dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - acceptance, acceptance])


def _acceptance_artifact() -> ModelArtifactBundle:
    return ModelArtifactBundle(
        model=_AcceptanceModel(),
        preprocessor=None,
        u_cols=("U",),
        x_feature_cols=("x",),
        probability_target="acceptance",
    )


def test_load_deterministic_sample_rows_verifies_seed_recipe(tmp_path) -> None:
    eligible = np.arange(100)
    expected = np.sort(np.random.default_rng(17).choice(eligible, 10, replace=False))
    diagnostics = tmp_path / "diagnostics.npz"
    np.savez_compressed(diagnostics, row_indices=expected)

    actual = profit_dispersion.load_deterministic_sample_rows(
        diagnostics,
        eligible,
        sample_size=10,
        seed=17,
    )

    np.testing.assert_array_equal(actual, expected)


def test_load_deterministic_sample_rows_rejects_different_sample(tmp_path) -> None:
    diagnostics = tmp_path / "diagnostics.npz"
    np.savez_compressed(diagnostics, row_indices=np.arange(10))

    with pytest.raises(ValueError, match="sampling recipe"):
        profit_dispersion.load_deterministic_sample_rows(
            diagnostics,
            np.arange(100),
            sample_size=10,
            seed=17,
        )


def test_predict_acceptance_matrix_uses_each_action() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]})

    result = profit_dispersion.predict_acceptance_matrix(
        _acceptance_artifact(),
        frame,
        [0.0, 0.1],
    )

    np.testing.assert_allclose(result, [[0.8, 0.7], [0.8, 0.7]])


def test_exact_spline_acceptance_has_no_raw_model_fallback(monkeypatch) -> None:
    def fail_fit(*args, **kwargs):
        raise ValueError("deliberate spline failure")

    monkeypatch.setattr(profit_dispersion, "_spline_acceptance_row", fail_fit)

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        profit_dispersion.exact_spline_acceptance_matrix(
            _acceptance_artifact(),
            pd.DataFrame({"x": [1.0]}),
            [0.0, 0.16],
            np.ones(profit_dispersion.ANCHOR_U.size),
            n_jobs=1,
        )


def test_customer_profit_matrix_keeps_maximization_sign() -> None:
    acceptance = np.array([[0.5, 0.25], [1.0, 0.5]])

    profit = profit_dispersion.customer_profit_matrix(
        acceptance,
        premium=[100.0, 200.0],
        predicted_loss=[60.0, 150.0],
        u_values=[0.0, 0.1],
    )

    np.testing.assert_allclose(profit, [[20.0, 12.5], [50.0, 35.0]])


def test_summarize_profit_uses_population_std_and_raw_mad() -> None:
    values = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [8.0, 30.0],
        ]
    )

    summary = profit_dispersion.summarize_profit(values)

    np.testing.assert_allclose(summary.mean, np.mean(values, axis=0))
    np.testing.assert_allclose(summary.std, np.std(values, axis=0, ddof=0))
    np.testing.assert_allclose(summary.median, [3.0, 20.0])
    np.testing.assert_allclose(summary.mad, [2.0, 10.0])
