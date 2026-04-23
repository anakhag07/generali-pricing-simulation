from __future__ import annotations

import numpy as np

from experiments.results import ConstantBaselineResult, OptimizationTrace
from reporting.visualization import plot_loss_curves, plot_objective_u_slice


class DummyAxis:
    def __init__(self) -> None:
        self.plot_calls: list[dict[str, object]] = []
        self.axhline_calls: list[dict[str, object]] = []
        self.axvline_calls: list[dict[str, object]] = []
        self.scatter_calls: list[dict[str, object]] = []

    def plot(self, *_args, **kwargs) -> None:
        self.plot_calls.append(kwargs)

    def axhline(self, *_args, **kwargs) -> None:
        self.axhline_calls.append(kwargs)

    def axvline(self, *_args, **kwargs) -> None:
        self.axvline_calls.append(kwargs)

    def scatter(self, *_args, **kwargs) -> None:
        self.scatter_calls.append(kwargs)

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


class QuadraticObjective:
    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        del x_batch
        return float((u - 0.1) ** 2)


def _trace() -> OptimizationTrace:
    return OptimizationTrace(
        steps=[0, 1, 2],
        u_values=[0.0, 0.1, 0.2],
        objective_values=[1.0, 0.8, 0.6],
        u_grad_estimates=[0.1, 0.1, 0.1],
    )


def test_plot_loss_curves_draws_constant_u_baselines(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr("reporting.visualization._ensure_plot_dir", lambda _plot_dir: tmp_path)
    monkeypatch.setattr("reporting.visualization.plt.close", lambda *_args, **_kwargs: None)

    plot_loss_curves(
        {"first_order": _trace()},
        plot_dir="unused",
        constant_u_baselines=(
            ConstantBaselineResult(u=-0.3, value=1.3),
            ConstantBaselineResult(u=0.0, value=1.0),
            ConstantBaselineResult(u=0.2, value=0.9),
        ),
    )

    assert len(dummy_ax.axhline_calls) == 3
    assert {call["label"] for call in dummy_ax.axhline_calls} == {
        "const u=0.20 (best)",
        "const u=0.00 (rank 2)",
        "const u=-0.30 (rank 3)",
    }
    colors = [call["color"] for call in dummy_ax.axhline_calls]
    assert len({tuple(color) for color in colors}) == 3


def test_plot_objective_u_slice_draws_constant_u_baseline_markers(monkeypatch, tmp_path) -> None:
    dummy_ax = DummyAxis()
    dummy_fig = DummyFigure()

    monkeypatch.setattr(
        "reporting.visualization.plt.subplots",
        lambda *_args, **_kwargs: (dummy_fig, dummy_ax),
    )
    monkeypatch.setattr("reporting.visualization._ensure_plot_dir", lambda _plot_dir: tmp_path)
    monkeypatch.setattr("reporting.visualization.plt.close", lambda *_args, **_kwargs: None)

    plot_objective_u_slice(
        x_samples=np.zeros((4, 2), dtype=float),
        objective=QuadraticObjective(),
        traces={"first_order": _trace()},
        plot_dir="unused",
        constant_u_baselines=(
            ConstantBaselineResult(u=-0.3, value=0.16),
            ConstantBaselineResult(u=0.0, value=0.01),
            ConstantBaselineResult(u=0.2, value=0.01),
        ),
    )

    assert len(dummy_ax.axvline_calls) == 3
    baseline_labels = [call["label"] for call in dummy_ax.scatter_calls if "label" in call and str(call["label"]).startswith("const u=")]
    assert baseline_labels == ["const u=0.00 (best)", "const u=0.20 (rank 2)", "const u=-0.30 (rank 3)"]
    colors = [call["color"] for call in dummy_ax.axvline_calls]
    assert len({tuple(color) for color in colors}) == 3
