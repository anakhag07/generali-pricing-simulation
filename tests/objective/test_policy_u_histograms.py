from __future__ import annotations

import numpy as np

from experiments.results import OptimizationTrace  # noqa: F401
from objective import FixedRegressionObjective, LinearPolicy
from reporting.visualization import (
    ESTIMATOR_STYLES,
    _LINE_ALPHA,
    _LINE_WIDTH,
    _SCATTER_ALPHA,
    _plot_policy_acceptance_histograms,
    _plot_policy_u_acceptance_histograms,
    _plot_policy_u_histograms,
    _plot_policy_u_vs_acceptance_spread,
    _plot_policy_u_vs_objective,
)


class DummyAxis:
    def __init__(self) -> None:
        self.hist_calls: list[dict[str, object]] = []
        self.fill_between_calls: list[dict[str, object]] = []
        self.scatter_calls: list[dict[str, object]] = []
        self.plot_calls: list[dict[str, object]] = []
        self.twin_axes: list[DummyAxis] = []

    def hist(self, *args, **kwargs) -> None:
        self.hist_calls.append({"args": args, "kwargs": kwargs})

    def fill_between(self, *args, **kwargs) -> None:
        self.fill_between_calls.append({"args": args, "kwargs": kwargs})

    def scatter(self, *args, **kwargs) -> None:
        self.scatter_calls.append({"args": args, "kwargs": kwargs})

    def plot(self, *args, **kwargs) -> None:
        self.plot_calls.append({"args": args, "kwargs": kwargs})

    def twinx(self) -> "DummyAxis":
        axis = DummyAxis()
        self.twin_axes.append(axis)
        return axis

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return None

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return None

    def set_title(self, *_args, **_kwargs) -> None:
        return None

    def set_ylim(self, *_args, **_kwargs) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None

    def grid(self, *_args, **_kwargs) -> None:
        return None


class DummyAcceptanceObjective(FixedRegressionObjective):
    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        u_values = np.asarray(u_arr, dtype=float).reshape(-1)
        logits = 0.4 * x_arr[:, 0] - 0.8 * u_values
        return 1.0 / (1.0 + np.exp(-logits))


class DummyFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_plot_policy_u_histograms_draws_observed_and_estimator_series(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr(
        "reporting.visualization._ensure_plot_dir",
        lambda _plot_dir: tmp_path,
    )
    monkeypatch.setattr(
        "reporting.visualization.plt.close",
        lambda *_args, **_kwargs: None,
    )

    objective = FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )
    observed_u = np.asarray([0.9, 1.0, 1.1], dtype=float)
    x_samples = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    theta_by_estimator = {
        "first_order": np.asarray([0.1, 0.2], dtype=float),
        "spsa": np.asarray([0.3, -0.1], dtype=float),
    }

    _plot_policy_u_histograms(
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir="unused",
    )

    assert len(dummy_ax.hist_calls) == 3
    observed_kwargs = dummy_ax.hist_calls[0]["kwargs"]
    assert observed_kwargs["label"] == "observed U"
    assert observed_kwargs["density"] is True
    assert observed_kwargs["alpha"] == _SCATTER_ALPHA

    first_order_kwargs = dummy_ax.hist_calls[1]["kwargs"]
    assert first_order_kwargs["label"] == ESTIMATOR_STYLES["first_order"]["label"]
    assert first_order_kwargs["color"] == ESTIMATOR_STYLES["first_order"]["color"]
    assert first_order_kwargs["histtype"] == "step"
    assert first_order_kwargs["alpha"] == _LINE_ALPHA
    assert first_order_kwargs["linewidth"] == _LINE_WIDTH

    spsa_kwargs = dummy_ax.hist_calls[2]["kwargs"]
    assert spsa_kwargs["label"] == ESTIMATOR_STYLES["spsa"]["label"]
    assert spsa_kwargs["color"] == ESTIMATOR_STYLES["spsa"]["color"]
    assert spsa_kwargs["histtype"] == "step"
    assert spsa_kwargs["alpha"] == _LINE_ALPHA
    assert spsa_kwargs["linewidth"] == _LINE_WIDTH


def test_plot_policy_u_vs_objective_draws_points_and_binned_lines(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr(
        "reporting.visualization._ensure_plot_dir",
        lambda _plot_dir: tmp_path,
    )
    monkeypatch.setattr(
        "reporting.visualization.plt.close",
        lambda *_args, **_kwargs: None,
    )

    objective = FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )
    observed_u = np.linspace(-0.6, 0.6, 120, dtype=float)
    x_samples = np.linspace(-1.0, 1.0, 120, dtype=float).reshape(-1, 1)
    theta_by_estimator = {
        "first_order": np.asarray([0.1, 0.2], dtype=float),
        "spsa": np.asarray([0.3, -0.1], dtype=float),
    }

    _plot_policy_u_vs_objective(
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir="unused",
    )

    assert len(dummy_ax.scatter_calls) == 3
    assert len(dummy_ax.plot_calls) == 3

    observed_scatter_kwargs = dummy_ax.scatter_calls[0]["kwargs"]
    assert observed_scatter_kwargs["color"] == "#969696"
    assert observed_scatter_kwargs["alpha"] == _SCATTER_ALPHA

    first_order_scatter_kwargs = dummy_ax.scatter_calls[1]["kwargs"]
    assert first_order_scatter_kwargs["color"] == ESTIMATOR_STYLES["first_order"]["color"]
    assert first_order_scatter_kwargs["marker"] == ESTIMATOR_STYLES["first_order"]["marker"]

    spsa_scatter_kwargs = dummy_ax.scatter_calls[2]["kwargs"]
    assert spsa_scatter_kwargs["color"] == ESTIMATOR_STYLES["spsa"]["color"]
    assert spsa_scatter_kwargs["marker"] == ESTIMATOR_STYLES["spsa"]["marker"]

    observed_line_kwargs = dummy_ax.plot_calls[0]["kwargs"]
    assert observed_line_kwargs["label"] == "observed U"
    assert observed_line_kwargs["color"] == "#636363"
    assert observed_line_kwargs["alpha"] == _LINE_ALPHA
    assert observed_line_kwargs["linewidth"] == _LINE_WIDTH

    first_order_line_kwargs = dummy_ax.plot_calls[1]["kwargs"]
    assert first_order_line_kwargs["label"] == ESTIMATOR_STYLES["first_order"]["label"]
    assert first_order_line_kwargs["color"] == ESTIMATOR_STYLES["first_order"]["color"]
    assert first_order_line_kwargs["alpha"] == _LINE_ALPHA
    assert first_order_line_kwargs["linewidth"] == _LINE_WIDTH

    spsa_line_kwargs = dummy_ax.plot_calls[2]["kwargs"]
    assert spsa_line_kwargs["label"] == ESTIMATOR_STYLES["spsa"]["label"]
    assert spsa_line_kwargs["color"] == ESTIMATOR_STYLES["spsa"]["color"]
    assert spsa_line_kwargs["alpha"] == _LINE_ALPHA
    assert spsa_line_kwargs["linewidth"] == _LINE_WIDTH


def test_plot_policy_acceptance_histograms_draws_observed_and_estimator_series(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr(
        "reporting.visualization._ensure_plot_dir",
        lambda _plot_dir: tmp_path,
    )
    monkeypatch.setattr(
        "reporting.visualization.plt.close",
        lambda *_args, **_kwargs: None,
    )

    objective = DummyAcceptanceObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )
    observed_u = np.asarray([0.9, 1.0, 1.1], dtype=float)
    x_samples = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    theta_by_estimator = {
        "first_order": np.asarray([0.1, 0.2], dtype=float),
        "spsa": np.asarray([0.3, -0.1], dtype=float),
    }

    _plot_policy_acceptance_histograms(
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir="unused",
    )

    assert len(dummy_ax.hist_calls) == 3
    assert dummy_ax.hist_calls[0]["kwargs"]["label"] == "observed U"
    assert dummy_ax.hist_calls[1]["kwargs"]["label"] == ESTIMATOR_STYLES["first_order"]["label"]
    assert dummy_ax.hist_calls[2]["kwargs"]["label"] == ESTIMATOR_STYLES["spsa"]["label"]


def test_plot_policy_u_acceptance_histograms_draws_customer_histogram_and_acceptance(monkeypatch, tmp_path) -> None:
    hist_ax = DummyAxis()
    scatter_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, np.asarray([[hist_ax, scatter_ax]], dtype=object)),
    )
    monkeypatch.setattr(
        "reporting.visualization._ensure_plot_dir",
        lambda _plot_dir: tmp_path,
    )
    monkeypatch.setattr(
        "reporting.visualization.plt.close",
        lambda *_args, **_kwargs: None,
    )

    objective = DummyAcceptanceObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )
    x_samples = np.linspace(-1.0, 1.0, 120, dtype=float).reshape(-1, 1)
    theta_by_estimator = {"first_order": np.asarray([0.1, 0.2], dtype=float)}

    _plot_policy_u_acceptance_histograms(
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir="unused",
    )

    assert len(hist_ax.hist_calls) == 1
    assert hist_ax.hist_calls[0]["kwargs"]["label"] == "customers"
    assert len(hist_ax.twin_axes) == 1
    assert hist_ax.twin_axes[0].plot_calls[0]["kwargs"]["label"] == "bin mean acceptance"
    assert len(scatter_ax.scatter_calls) == 1
    assert len(scatter_ax.plot_calls) == 1
    assert scatter_ax.plot_calls[0]["kwargs"]["label"] == "bin mean acceptance"


def test_plot_policy_u_vs_acceptance_spread_draws_quantile_bands_and_points(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr(
        "reporting.visualization._ensure_plot_dir",
        lambda _plot_dir: tmp_path,
    )
    monkeypatch.setattr(
        "reporting.visualization.plt.close",
        lambda *_args, **_kwargs: None,
    )

    objective = DummyAcceptanceObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.8,
        beta_3=[0.1],
        beta_4=0.3,
    )
    observed_u = np.linspace(-0.6, 0.6, 120, dtype=float)
    x_samples = np.linspace(-1.0, 1.0, 120, dtype=float).reshape(-1, 1)
    theta_by_estimator = {
        "first_order": np.asarray([0.1, 0.2], dtype=float),
        "spsa": np.asarray([0.3, -0.1], dtype=float),
    }

    _plot_policy_u_vs_acceptance_spread(
        observed_u,
        x_samples,
        objective,
        theta_by_estimator,
        plot_dir="unused",
    )

    assert len(dummy_ax.fill_between_calls) == 2
    assert len(dummy_ax.scatter_calls) == 3
    assert len(dummy_ax.plot_calls) == 4
    assert dummy_ax.fill_between_calls[0]["kwargs"]["label"] == "fixed-u 5-95% across X"
    assert dummy_ax.plot_calls[0]["kwargs"]["label"] == "fixed-u median across X"
    assert dummy_ax.plot_calls[1]["kwargs"]["label"] == "observed U"
