from __future__ import annotations

import matplotlib.axes
import numpy as np

from reporting.visualization import _plot_policy_objective_contribution_summary


class DummyObjectiveContributionObjective:
    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float)
        x_arr = np.asarray(x_batch, dtype=float)
        return theta_arr[0] + theta_arr[1] * x_arr[:, 0]

    def _clip_u(self, u_values: np.ndarray) -> np.ndarray:
        return np.asarray(u_values, dtype=float)

    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        u_values = np.asarray(u_arr, dtype=float)
        logits = x_arr[:, 0] - u_values
        return 1.0 / (1.0 + np.exp(-logits))

    def _value_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        u_values = np.asarray(u_arr, dtype=float)
        return np.asarray([1.0, -2.0, 3.0, -4.0], dtype=float) + 0.0 * (
            x_arr[:, 0] + u_values
        )


def test_objective_contribution_summary_plots_expected_profit_and_acceptance(
    monkeypatch,
    tmp_path,
) -> None:
    x_samples = np.linspace(0.0, 1.0, 4, dtype=float).reshape(-1, 1)
    theta = np.asarray([0.2, 0.3], dtype=float)
    objective = DummyObjectiveContributionObjective()
    hist_values: list[np.ndarray] = []
    scatter_values: list[tuple[np.ndarray, np.ndarray]] = []
    xlabel_values: list[str] = []
    ylabel_values: list[str] = []

    original_hist = matplotlib.axes.Axes.hist
    original_scatter = matplotlib.axes.Axes.scatter
    original_set_xlabel = matplotlib.axes.Axes.set_xlabel
    original_set_ylabel = matplotlib.axes.Axes.set_ylabel

    def record_hist(self, x, *args, **kwargs):
        hist_values.append(np.asarray(x, dtype=float))
        return original_hist(self, x, *args, **kwargs)

    def record_scatter(self, x, y, *args, **kwargs):
        scatter_values.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        return original_scatter(self, x, y, *args, **kwargs)

    def record_xlabel(self, xlabel, *args, **kwargs):
        xlabel_values.append(str(xlabel))
        return original_set_xlabel(self, xlabel, *args, **kwargs)

    def record_ylabel(self, ylabel, *args, **kwargs):
        ylabel_values.append(str(ylabel))
        return original_set_ylabel(self, ylabel, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", record_hist)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", record_xlabel)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", record_ylabel)

    _plot_policy_objective_contribution_summary(
        x_samples,
        objective,  # type: ignore[arg-type]
        {"first_order": theta},
        plot_dir=str(tmp_path),
    )

    policy_u = objective.policy_value(theta, x_samples)
    expected_profit = -objective._value_batch(x_samples, policy_u)
    expected_acceptance = objective._acceptance_proba(x_samples, policy_u)
    assert (tmp_path / "objective_contribution_summary.png").is_file()
    np.testing.assert_allclose(hist_values[0], expected_profit)
    np.testing.assert_allclose(scatter_values[0][0], expected_acceptance)
    np.testing.assert_allclose(scatter_values[0][1], expected_profit)
    assert any("Expected profit contribution" in label for label in xlabel_values)
    assert any("Predicted acceptance probability" in label for label in xlabel_values)
    assert any("Expected profit contribution" in label for label in ylabel_values)
