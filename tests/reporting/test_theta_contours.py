from __future__ import annotations

import numpy as np
import pytest

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


def _trace(theta_values: list[np.ndarray]) -> OptimizationTrace:
    return OptimizationTrace(
        steps=[0, 1],
        u_values=[0.0, 0.0],
        objective_values=[1.0, 0.8],
        u_grad_estimates=[0.1, 0.1],
        theta_values=theta_values,
    )


def test_plot_theta_objective_contours_uses_path_only_legend_and_special_markers(
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
    assert all("marker" not in call for call in dummy_ax.plot_calls)

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
    assert final_call["s"] == 140.0
    assert final_call["zorder"] == 7
