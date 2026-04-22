from __future__ import annotations

from typing import cast

from experiments.results import OptimizationTrace
from reporting.visualization import plot_step_sizes


class DummyAxis:
    def __init__(self) -> None:
        self._yscale = "linear"
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

    def set_yscale(self, scale: str) -> None:
        self._yscale = scale

    def get_yscale(self) -> str:
        return self._yscale


class DummyFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_plot_step_sizes_uses_log_scale(monkeypatch, tmp_path) -> None:
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
        steps=[0, 1, 2],
        u_values=[0.0, 0.0, 0.0],
        objective_values=[0.0, 0.0, 0.0],
        u_grad_estimates=[0.0, 0.0, 0.0],
        step_sizes=[1e-2, 1e-3, 1e-4],
    )

    plot_step_sizes({"first_order": trace}, plot_dir="unused")

    assert dummy_ax.get_yscale() == "log"
    assert len(dummy_ax.plot_calls) == 1
    kwargs = dummy_ax.plot_calls[0]
    alpha = cast(float, kwargs["alpha"])
    assert kwargs["marker"] == "o"
    assert alpha == 0.5
