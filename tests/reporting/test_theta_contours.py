from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.reporting.plots import PlotReporter, _contour_grid_size, _contour_x_samples
from experiments.results import EstimatorResult, ExperimentResult
from experiments.results import OptimizationTrace
from objective import FixedRegressionObjective, LinearPolicy
from reporting.visualization import (
    plot_theta_objective_contours,
    select_theta_axes_max_variance,
    theta_objective_contour_grid,
)


class DummyColorbar:
    def __init__(self) -> None:
        self.label: str | None = None

    def set_label(self, label: str) -> None:
        self.label = label


class DummyAxis:
    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []
        self.scatter_calls: list[dict[str, object]] = []

    def contourf(self, *_args, **_kwargs) -> str:
        return "contour"

    def contour(self, *_args, **_kwargs) -> None:
        return None

    def plot(self, *_args, **kwargs) -> None:
        self.plot_calls.append(kwargs)

    def scatter(self, *_args, **kwargs) -> None:
        self.scatter_calls.append(kwargs)

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return None

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return None

    def set_title(self, *_args, **_kwargs) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None


class DummyFigure:
    def __init__(self) -> None:
        self.colorbar_obj = DummyColorbar()

    def colorbar(self, *_args, **_kwargs) -> DummyColorbar:
        return self.colorbar_obj

    def tight_layout(self) -> None:
        return None

    def savefig(self, *_args, **_kwargs) -> None:
        return None


def _build_theta_objective() -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.4],
        beta_4=0.5,
    )


def test_theta_objective_contour_grid_shapes() -> None:
    x_samples = np.array([[1.0, -1.0]], dtype=float)
    theta_base = np.asarray([0.1, 0.2, 0.3], dtype=float)
    objective = _build_theta_objective()

    grid_x, grid_y, objective_grid = theta_objective_contour_grid(
        x_samples,
        objective,
        theta_base,
        axis_indices=(0, 1),
        theta_refs=[theta_base, theta_base + 0.05],
        grid_size=20,
    )

    assert grid_x.shape == (20, 20)
    assert grid_y.shape == (20, 20)
    assert objective_grid.shape == (20, 20)


def test_theta_objective_contour_grid_rejects_invalid_axes() -> None:
    x_samples = np.array([[1.0, 0.5]], dtype=float)
    theta_base = np.asarray([0.1, 0.2, 0.3], dtype=float)
    objective = _build_theta_objective()

    with pytest.raises(ValueError, match="distinct"):
        theta_objective_contour_grid(
            x_samples,
            objective,
            theta_base,
            axis_indices=(1, 1),
        )

    with pytest.raises(ValueError, match="valid indices"):
        theta_objective_contour_grid(
            x_samples,
            objective,
            theta_base,
            axis_indices=(0, 5),
        )


def test_select_theta_axes_max_variance_orders_by_variance() -> None:
    theta_points = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([1.0, 2.0, 0.5], dtype=float),
        np.array([2.0, 4.0, 1.0], dtype=float),
        np.array([3.0, 6.0, 1.5], dtype=float),
    ]
    axis_indices = select_theta_axes_max_variance(theta_points)
    assert axis_indices == (1, 0)


class DummyModelBasedObjective:
    acceptance_model = object()
    loss_model = object()

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del theta
        return float(np.mean(x_batch))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del x_batch
        return np.zeros_like(theta, dtype=float)


def test_model_based_contour_samples_are_deterministic_and_capped() -> None:
    x_samples = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
    sampled = _contour_x_samples(x_samples, DummyModelBasedObjective(), max_rows=200)

    assert sampled.shape == (200, 3)
    np.testing.assert_allclose(sampled[0], x_samples[0])
    np.testing.assert_allclose(sampled[-1], x_samples[-1])


def test_non_model_based_contour_samples_use_full_data() -> None:
    x_samples = np.arange(1000 * 2, dtype=float).reshape(1000, 2)
    sampled = _contour_x_samples(x_samples, _build_theta_objective(), max_rows=200)

    assert sampled is x_samples


def test_model_based_contour_grid_size_is_lowered() -> None:
    assert _contour_grid_size(DummyModelBasedObjective()) == 20
    assert _contour_grid_size(_build_theta_objective()) == 60


def _trace(theta_values: list[np.ndarray]) -> OptimizationTrace:
    return OptimizationTrace(
        steps=[0, 1],
        u_values=[0.0, 0.0],
        objective_values=[1.0, 0.8],
        u_grad_estimates=[0.1, 0.1],
        theta_values=theta_values,
    )


def test_plot_reporter_subsamples_model_based_contours_and_writes_timings(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, np.ndarray] = {}

    def no_op(*_args, **_kwargs) -> None:
        return None

    def capture_contour(x_samples, *_args, **kwargs) -> None:
        captured["x_samples"] = np.asarray(x_samples, dtype=float)
        captured["grid_size"] = kwargs["grid_size"]

    monkeypatch.setattr("experiments.reporting.plots.plot_loss_curves", no_op)
    monkeypatch.setattr("experiments.reporting.plots.plot_gradient_norms", no_op)
    monkeypatch.setattr("experiments.reporting.plots.plot_step_sizes", no_op)
    monkeypatch.setattr("experiments.reporting.plots.plot_theta_objective_contours", capture_contour)

    x_samples = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
    theta0 = np.asarray([0.0, 0.0], dtype=float)
    config = ExperimentConfig(
        state_dim=3,
        objective=DummyModelBasedObjective(),
        theta0=theta0,
        n_samples=x_samples.shape[0],
        x_fixed=x_samples,
        step_rule="constant",
        perturbation_space="theta",
        t_steps=1,
        plot=True,
        enabled_estimators=("first_order",),
    )
    trace = _trace([theta0, np.asarray([0.1, -0.1], dtype=float)])
    result = ExperimentResult(
        config=config,
        x_samples=x_samples,
        initial_value=0.0,
        results={
            "first_order": EstimatorResult(
                theta=np.asarray([0.1, -0.1], dtype=float),
                u=0.0,
                value=0.0,
                time=0.0,
            )
        },
        traces={"first_order": trace},
    )
    run_context = RunContext(
        experiment_name="test",
        run_id="run",
        run_dir=tmp_path,
        plots_dir=tmp_path / "plots",
        started_at=datetime(2026, 1, 1),
    )

    PlotReporter().on_end(run_context, result)

    assert captured["x_samples"].shape == (200, 3)
    assert captured["grid_size"] == 20
    timings = json.loads((run_context.plots_dir / "plot_timings.json").read_text(encoding="utf-8"))
    assert timings["loss_curves"] >= 0.0
    assert timings["gradient_norms"] >= 0.0
    assert timings["theta_objective_contours"] >= 0.0


def test_plot_theta_objective_contours_keeps_path_markers_and_special_overlays(
    monkeypatch, tmp_path
) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr("reporting.visualization._ensure_plot_dir", lambda _plot_dir: tmp_path)
    monkeypatch.setattr("reporting.visualization.plt.close", lambda *_args, **_kwargs: None)

    theta_base = np.asarray([0.0, 0.0, 0.0], dtype=float)
    x_samples = np.array([[1.0, -1.0], [0.25, 0.5]], dtype=float)
    objective = _build_theta_objective()

    plot_theta_objective_contours(
        x_samples=x_samples,
        objective=objective,
        theta_base=theta_base,
        plot_dir="unused",
        axis_indices=(0, 1),
        theta_refs=[theta_base],
        theta_points=[
            (theta_base, "initial"),
            (np.asarray([0.4, -0.3, 0.2], dtype=float), "first-order final point"),
        ],
        traces={
            "first_order": _trace([theta_base, np.asarray([0.4, -0.3, 0.2], dtype=float)]),
            "finite_difference": _trace([theta_base, np.asarray([0.2, 0.1, 0.0], dtype=float)]),
            "stein_difference": _trace([theta_base, np.asarray([-0.2, 0.3, 0.1], dtype=float)]),
            "spsa": _trace([theta_base, np.asarray([0.1, -0.2, -0.1], dtype=float)]),
        },
        grid_size=5,
    )

    assert [call["label"] for call in dummy_ax.plot_calls] == [
        "first-order",
        "finite-difference",
        "stein-difference",
        "SPSA",
    ]
    assert [call["marker"] for call in dummy_ax.plot_calls] == ["X", "P", "^", "D"]

    assert [call["label"] for call in dummy_ax.scatter_calls] == [
        "initial",
        "first-order final point",
    ]

    initial_call, final_call = dummy_ax.scatter_calls
    assert initial_call["marker"] == "o"
    assert initial_call["facecolors"] == "white"
    assert initial_call["edgecolors"] == "#111111"
    assert initial_call["zorder"] == 6

    assert final_call["marker"] == "X"
    assert final_call["color"] == "black"
    assert final_call["edgecolors"] == "black"
    assert final_call["s"] == 120.0
    assert final_call["zorder"] == 7
