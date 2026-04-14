from __future__ import annotations

import numpy as np

from experiments.results import OptimizationTrace  # noqa: F401
from objective import FixedRegressionObjective, LinearPolicy
from reporting.visualization import ESTIMATOR_STYLES, _plot_policy_u_histograms


class DummyAxis:
    def __init__(self) -> None:
        self.hist_calls: list[dict[str, object]] = []

    def hist(self, *args, **kwargs) -> None:
        self.hist_calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return None

    def set_ylabel(self, *_args, **_kwargs) -> None:
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

    first_order_kwargs = dummy_ax.hist_calls[1]["kwargs"]
    assert first_order_kwargs["label"] == ESTIMATOR_STYLES["first_order"]["label"]
    assert first_order_kwargs["color"] == ESTIMATOR_STYLES["first_order"]["color"]
    assert first_order_kwargs["histtype"] == "step"

    spsa_kwargs = dummy_ax.hist_calls[2]["kwargs"]
    assert spsa_kwargs["label"] == ESTIMATOR_STYLES["spsa"]["label"]
    assert spsa_kwargs["color"] == ESTIMATOR_STYLES["spsa"]["color"]
    assert spsa_kwargs["histtype"] == "step"
