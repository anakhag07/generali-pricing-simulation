from __future__ import annotations

import matplotlib.axes
import numpy as np

from reporting.visualization import (
    _plot_policy_delta_u_by_elasticity,
    _plot_policy_delta_u_histograms,
)


class DummySensitivityObjective:
    def __init__(self) -> None:
        self.last_u_ref: np.ndarray | None = None

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        x_arr = np.asarray(x_batch, dtype=float)
        return theta_arr[0] + theta_arr[1] * x_arr[:, 0]

    def _clip_u(self, u_values: np.ndarray) -> np.ndarray:
        return np.asarray(u_values, dtype=float)

    def _d_acceptance_du_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        self.last_u_ref = np.asarray(u_arr, dtype=float).copy()
        x_arr = np.asarray(x_batch, dtype=float)
        return -(0.2 + 0.1 * x_arr[:, 0])


def test_delta_u_by_elasticity_plot_uses_absolute_sensitivity_and_reference_u(
    monkeypatch,
    tmp_path,
) -> None:
    x_samples = np.linspace(0.0, 1.0, 12, dtype=float).reshape(-1, 1)
    observed_u = np.linspace(-0.05, 0.05, 12, dtype=float)
    theta = np.asarray([0.1, 0.2], dtype=float)
    objective = DummySensitivityObjective()
    scatter_x_values: list[np.ndarray] = []
    scatter_y_values: list[np.ndarray] = []
    xlabel_values: list[str] = []
    ylabel_values: list[str] = []

    original_scatter = matplotlib.axes.Axes.scatter
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def record_scatter(self, x, y, *args, **kwargs):
        scatter_x_values.append(np.asarray(x, dtype=float))
        scatter_y_values.append(np.asarray(y, dtype=float))
        return original_scatter(self, x, y, *args, **kwargs)

    def record_xlabel(self, xlabel, *args, **kwargs):
        xlabel_values.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def record_ylabel(self, ylabel, *args, **kwargs):
        ylabel_values.append(str(ylabel))
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", record_ylabel)

    _plot_policy_delta_u_by_elasticity(
        observed_u,
        x_samples,
        objective,  # type: ignore[arg-type]
        {"first_order": theta},
        plot_dir=str(tmp_path),
    )

    assert (tmp_path / "delta_u_by_sensitivity.png").is_file()
    np.testing.assert_allclose(objective.last_u_ref, np.full(x_samples.shape[0], 0.08))
    np.testing.assert_allclose(
        scatter_x_values[0],
        np.abs(-(0.2 + 0.1 * x_samples[:, 0])),
    )
    np.testing.assert_allclose(
        scatter_y_values[0],
        objective.policy_value(theta, x_samples) - observed_u,
    )
    assert any("evaluated at u = 0.08" in label for label in xlabel_values)
    assert any(
        "Δu = optimized customer u - historical u" in label
        for label in ylabel_values
    )


def test_delta_u_histogram_plots_policy_minus_observed_u(monkeypatch, tmp_path) -> None:
    x_samples = np.linspace(0.0, 1.0, 12, dtype=float).reshape(-1, 1)
    observed_u = np.linspace(-0.05, 0.05, 12, dtype=float)
    theta = np.asarray([0.1, 0.2], dtype=float)
    objective = DummySensitivityObjective()
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

    _plot_policy_delta_u_histograms(
        observed_u,
        x_samples,
        objective,  # type: ignore[arg-type]
        {"first_order": theta},
        plot_dir=str(tmp_path),
    )

    assert (tmp_path / "delta_u_histogram.png").is_file()
    np.testing.assert_allclose(
        hist_values[0],
        objective.policy_value(theta, x_samples) - observed_u,
    )
    assert any("Δu = optimized customer u - historical u" in label for label in xlabel_values)
