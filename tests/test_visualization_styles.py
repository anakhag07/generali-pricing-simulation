from __future__ import annotations

from typing import cast

from experiments.results import OptimizationTrace
from reporting.visualization import ESTIMATOR_STYLES, plot_loss_curves


class DummyAxis:
    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []

    def plot(self, *_args, **kwargs) -> None:
        self.plot_calls.append(kwargs)

    def set_ylabel(self, *_args, **_kwargs) -> None:
        return None

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return None

    def legend(self, *_args, **_kwargs) -> None:
        return None

    def grid(self, *_args, **_kwargs) -> None:
        return None


class DummyFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_estimator_styles_use_distinct_markers() -> None:
    ordered_names = ("first_order", "finite_difference", "gauss_stein", "stein_difference", "spsa")
    markers = [ESTIMATOR_STYLES[name]["marker"] for name in ordered_names]
    assert len(markers) == len(set(markers))


def test_plot_loss_curves_uses_marker_and_darker_lines(monkeypatch, tmp_path) -> None:
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

    trace = OptimizationTrace(
        steps=[0, 1, 2, 3],
        u_values=[0.0, 0.0, 0.0, 0.0],
        objective_values=[1.0, 0.8, 0.6, 0.5],
        u_grad_estimates=[0.1, 0.1, 0.1, 0.1],
    )

    plot_loss_curves({"gauss_stein": trace}, plot_dir="unused")

    assert len(dummy_ax.plot_calls) == 1
    kwargs = dummy_ax.plot_calls[0]
    alpha = cast(float, kwargs["alpha"])
    linewidth = cast(float, kwargs["linewidth"])
    assert kwargs["marker"] == ESTIMATOR_STYLES["gauss_stein"]["marker"]
    assert alpha == 0.6
    assert linewidth >= 1.8
