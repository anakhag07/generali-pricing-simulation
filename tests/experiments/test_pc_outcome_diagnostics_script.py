from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import plot_pc_outcome_diagnostics as diagnostics


class _FakeModelBasedObjective:
    premium_col = 1

    def policy_theta_dim(self) -> int:
        return 2

    def _policy_features(self, x_batch: np.ndarray) -> np.ndarray:
        return np.column_stack([x_batch[:, 0], x_batch[:, 1] - x_batch[:, 0]])

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        features = self._policy_features(x_batch)
        return theta[0] + theta[1] * features[:, 0]

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        return np.clip(u, -0.5, 0.5)

    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        return 0.6 - 0.1 * u_arr + 0.01 * x_batch[:, 0]

    def _loss_prediction(self, x_batch: np.ndarray) -> np.ndarray:
        return 2.0 + x_batch[:, 1]


def test_safe_corr_returns_nan_for_constant_vector() -> None:
    corr = diagnostics._safe_corr(np.ones(3), np.array([1.0, 2.0, 3.0]))

    assert np.isnan(corr)


def test_safe_corr_returns_pearson_correlation() -> None:
    corr = diagnostics._safe_corr(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))

    assert corr == pytest.approx(1.0)


def test_load_summary_theta_reads_estimator_theta(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps({"estimators": {"first_order": {"theta": [0.2, -0.1]}}}),
        encoding="utf-8",
    )

    theta = diagnostics._load_summary_theta(path, "first_order")

    np.testing.assert_allclose(theta, np.array([0.2, -0.1]))


def test_build_diagnostic_data_uses_final_policy_quantities() -> None:
    config = SimpleNamespace(objective=_FakeModelBasedObjective())
    theta = np.array([0.1, 0.2], dtype=float)
    x_batch = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=float)

    data = diagnostics.build_diagnostic_data(config, theta, x_batch)

    np.testing.assert_allclose(data.components, np.array([[1.0, 2.0], [2.0, 3.0]]))
    np.testing.assert_allclose(data.u, np.array([0.3, 0.5]))
    np.testing.assert_allclose(data.loss, np.array([5.0, 7.0]))
    assert data.component_names == ("component_1", "component_2")


def test_write_diagnostic_plots_creates_expected_outputs(tmp_path) -> None:
    data = diagnostics.DiagnosticData(
        components=np.array([[0.0, 1.0], [1.0, 0.5], [2.0, 0.0]], dtype=float),
        component_names=("PC1", "PC2"),
        u=np.array([0.0, 0.1, 0.2], dtype=float),
        acceptance=np.array([0.8, 0.7, 0.6], dtype=float),
        loss=np.array([1.0, 2.0, 3.0], dtype=float),
        premium=np.array([10.0, 11.0, 12.0], dtype=float),
        per_sample_objective=np.array([-1.0, -2.0, -3.0], dtype=float),
    )

    outputs = diagnostics.write_diagnostic_plots(
        data,
        tmp_path,
        max_components=2,
        max_points=None,
        sample_seed=0,
    )

    assert {path.name for path in outputs} == {
        "pc_vs_acceptance.png",
        "pc_vs_loss.png",
        "pc_vs_u.png",
        "u_vs_acceptance.png",
        "pc_diagnostic_correlations.csv",
    }
    assert all(path.exists() for path in outputs)
