from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from objective import FixedRegressionObjective, LinearPolicy
from reporting.visualization import (
    _adaptive_contour_norm,
    _plot_policy_acceptance_histograms,
    _plot_policy_final_summary_metrics,
    _plot_policy_u_acceptance_histograms,
    _plot_policy_u_histograms,
    plot_theta_objective_contours,
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


class DummyCategoricalDataFrameObjective:
    def _signal(self, x_batch: object) -> np.ndarray:
        if hasattr(x_batch, "loc"):
            values = x_batch.loc[:, "X_vehicle_power"].astype(str)
            return values.map(lambda val: 0.0 if val == "." else float(val)).to_numpy(dtype=float)
        return np.asarray(x_batch, dtype=float)[:, 0]

    def _clip_u(self, u_values: np.ndarray) -> np.ndarray:
        return np.asarray(u_values, dtype=float)

    def policy_value(self, theta: np.ndarray, x_batch: object) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        return theta_arr[0] + theta_arr[1] * self._signal(x_batch) / 100.0

    def _acceptance_proba(self, x_batch: object, u_arr: np.ndarray) -> np.ndarray:
        logits = 0.01 * self._signal(x_batch) - 0.5 * np.asarray(u_arr, dtype=float)
        return 1.0 / (1.0 + np.exp(-logits))

    def _value_batch(self, x_batch: object, u_arr: np.ndarray) -> np.ndarray:
        signal = self._signal(x_batch)
        u_values = np.asarray(u_arr, dtype=float)
        return self._acceptance_proba(x_batch, u_values) * (signal - 10.0 * (u_values + 1.0))

    def value_at_u(self, x_batch: object, u: float) -> float:
        u_values = np.full(self._signal(x_batch).shape[0], float(u), dtype=float)
        return float(np.mean(self._value_batch(x_batch, u_values)))

    def value(self, theta: np.ndarray, x_batch: object) -> float:
        u_values = self.policy_value(theta, x_batch)
        return float(np.mean(self._value_batch(x_batch, u_values)))


def _x_samples() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 120, dtype=float).reshape(-1, 1)


def _categorical_x_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "X_vehicle_power": [".", "105", "112", "68"],
            "X_vehicle_weight": [".", "1240", "1390", "1430"],
        }
    )


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


def test_plot_policy_u_acceptance_histograms_adds_objective_value_panel(monkeypatch, tmp_path) -> None:
    x_samples = _x_samples()
    objective = _objective()
    theta = np.asarray([0.1, 0.2], dtype=float)
    theta_by_estimator = {"first_order": theta}
    hist_values: list[np.ndarray] = []
    xlabel_values: list[str] = []
    original_hist = matplotlib.axes.Axes.hist
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel

    def record_hist(self, x, *args, **kwargs):
        hist_values.append(np.asarray(x, dtype=float))
        return original_hist(self, x, *args, **kwargs)

    def record_xlabel(self, xlabel, *args, **kwargs):
        xlabel_values.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", record_hist)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record_xlabel)

    _plot_policy_u_acceptance_histograms(
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir=str(tmp_path),
    )

    policy_u = objective.policy.value(theta, x_samples)
    expected_objective_values = objective._value_batch(x_samples, policy_u)
    assert (tmp_path / "first_order.png").is_file()
    np.testing.assert_allclose(hist_values[1], expected_objective_values)
    assert any("Objective contribution M(x, u)" in label for label in xlabel_values)


def test_policy_diagnostic_plots_preserve_categorical_dataframes(tmp_path) -> None:
    x_samples = _categorical_x_samples()
    observed_u = np.asarray([-0.1, 0.0, 0.1, 0.2], dtype=float)
    theta_by_estimator = {"first_order": np.asarray([0.05, 0.1], dtype=float)}
    objective = DummyCategoricalDataFrameObjective()

    _plot_policy_u_histograms(
        observed_u,
        x_samples,
        objective,  # type: ignore[arg-type]
        theta_by_estimator,
        plot_dir=str(tmp_path),
    )
    _plot_policy_acceptance_histograms(
        observed_u,
        x_samples,
        objective,  # type: ignore[arg-type]
        theta_by_estimator,
        plot_dir=str(tmp_path),
    )
    _plot_policy_final_summary_metrics(
        x_samples,
        objective,  # type: ignore[arg-type]
        theta_by_estimator,
        {"first_order": 0.01},
        plot_dir=str(tmp_path),
    )
    _plot_policy_u_acceptance_histograms(
        x_samples,
        objective,  # type: ignore[arg-type]
        theta_by_estimator,
        plot_dir=str(tmp_path / "u_acceptance"),
    )
    plot_theta_objective_contours(
        x_samples,
        objective,  # type: ignore[arg-type]
        np.asarray([0.05, 0.1], dtype=float),
        plot_dir=str(tmp_path),
        grid_size=3,
    )

    assert (tmp_path / "u_histogram.png").is_file()
    assert (tmp_path / "acceptance_histograms.png").is_file()
    assert (tmp_path / "final_summary_metrics.png").is_file()
    assert (tmp_path / "u_acceptance" / "first_order.png").is_file()
    assert (tmp_path / "theta_objective_contours.png").is_file()


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
