from __future__ import annotations

import matplotlib
import numpy as np

from objective import FixedRegressionObjective, LinearPolicy
from reporting.visualization import (
    _adaptive_contour_norm,
    _plot_policy_acceptance_histograms,
    _plot_policy_final_summary_metrics,
    _plot_policy_u_acceptance_histograms,
    _plot_policy_u_histograms,
)


class DummyAcceptanceObjective(FixedRegressionObjective):
    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        u_values = np.asarray(u_arr, dtype=float).reshape(-1)
        logits = 0.4 * x_arr[:, 0] - 0.8 * u_values
        return 1.0 / (1.0 + np.exp(-logits))


def _objective() -> DummyAcceptanceObjective:
    return DummyAcceptanceObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )


def _x_samples() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 120, dtype=float).reshape(-1, 1)


def _observed_u() -> np.ndarray:
    return np.linspace(-0.6, 0.6, 120, dtype=float)


def _theta_by_estimator() -> dict[str, np.ndarray]:
    return {
        "first_order": np.asarray([0.1, 0.2], dtype=float),
        "spsa": np.asarray([0.3, -0.1], dtype=float),
    }


def test_plot_policy_u_histograms_writes_renamed_distribution(tmp_path) -> None:
    _plot_policy_u_histograms(
        _observed_u(),
        _x_samples(),
        _objective(),
        _theta_by_estimator(),
        plot_dir=str(tmp_path),
    )

    assert (tmp_path / "u_histogram.png").is_file()
    assert not (tmp_path / "policy_u_histograms.png").exists()


def test_plot_policy_acceptance_histograms_writes_renamed_distribution(tmp_path) -> None:
    _plot_policy_acceptance_histograms(
        _observed_u(),
        _x_samples(),
        _objective(),
        _theta_by_estimator(),
        plot_dir=str(tmp_path),
    )

    assert (tmp_path / "acceptance_histograms.png").is_file()
    assert not (tmp_path / "policy_acceptance_histograms.png").exists()


def test_plot_policy_final_summary_metrics_writes_customer_iqr_chart(tmp_path) -> None:
    _plot_policy_final_summary_metrics(
        _x_samples(),
        _objective(),
        _theta_by_estimator(),
        {"first_order": 0.12, "spsa": 0.34},
        plot_dir=str(tmp_path),
    )

    assert (tmp_path / "final_summary_metrics.png").is_file()


def test_plot_policy_u_acceptance_histograms_writes_one_file_per_estimator(tmp_path) -> None:
    _plot_policy_u_acceptance_histograms(
        _x_samples(),
        _objective(),
        _theta_by_estimator(),
        plot_dir=str(tmp_path),
        acceptance_floor=0.7,
    )

    assert (tmp_path / "first_order.png").is_file()
    assert (tmp_path / "spsa.png").is_file()
    assert not (tmp_path / "policy_u_acceptance_histograms.png").exists()
    assert not (tmp_path / "policy_u_vs_acceptance_spread.png").exists()
    assert not (tmp_path / "policy_u_vs_objective.png").exists()


def test_adaptive_contour_norm_uses_linear_log_or_symlog_scales() -> None:
    assert _adaptive_contour_norm(np.asarray([[1.0, 2.0, 3.0]])) is None
    assert isinstance(
        _adaptive_contour_norm(np.asarray([[1.0, 250.0]], dtype=float)),
        matplotlib.colors.LogNorm,
    )
    assert isinstance(
        _adaptive_contour_norm(np.asarray([[-250.0, -1.0, 0.0, 100.0]], dtype=float)),
        matplotlib.colors.SymLogNorm,
    )
