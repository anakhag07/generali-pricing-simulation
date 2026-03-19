"""Tests for W&B reporter streaming and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0, default_policy
from experiments.reporters import ReporterStack, RunContext, WandbReporter
from experiments.results import EstimatorResult, ExperimentResult
from objective import FixedRegressionObjective


@dataclass
class _FakeImage:
    path: str


class _FakeWandb:
    def __init__(self) -> None:
        self.init_calls: list[dict] = []
        self.log_calls: list[tuple[dict, int | None]] = []
        self.define_metric_calls: list[tuple[str, dict]] = []
        self.finish_calls = 0

    def init(self, **kwargs: object) -> object:
        self.init_calls.append(dict(kwargs))
        return object()

    def log(self, payload: dict, step: int | None = None) -> None:
        self.log_calls.append((dict(payload), step))

    def finish(self) -> None:
        self.finish_calls += 1

    def define_metric(self, name: str, **kwargs: object) -> None:
        self.define_metric_calls.append((name, dict(kwargs)))

    def Image(self, path: str) -> _FakeImage:  # noqa: N802
        return _FakeImage(path=path)


def _build_config(**overrides: object) -> ExperimentConfig:
    objective = FixedRegressionObjective.from_parameters(
        policy=default_policy(1),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    kwargs = {
        "state_dim": 1,
        "objective": objective,
        "theta0": default_theta0(1),
        "n_samples": 2,
        "step_rule": "constant",
        "plot": False,
        "wandb_enabled": True,
        "wandb_project": "unit-tests",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _build_run_context(tmp_path: Path) -> RunContext:
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        experiment_name="wandb-test",
        run_id="20260101_000000",
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
        started_at=datetime(2026, 1, 1, 0, 0, 0),
    )


def _build_result(config: ExperimentConfig) -> ExperimentResult:
    x_samples = np.array([[0.1]], dtype=float)
    results = {
        "first_order": EstimatorResult(
            theta=np.asarray([0.1, 0.2], dtype=float),
            u=0.55,
            value=-0.12,
            time=0.04,
        ),
        "stein_difference": EstimatorResult(
            theta=np.asarray([0.15, 0.25], dtype=float),
            u=0.56,
            value=-0.11,
            time=0.05,
        ),
        "spsa": EstimatorResult(
            theta=np.asarray([0.2, 0.3], dtype=float),
            u=0.57,
            value=-0.10,
            time=0.06,
        ),
    }
    return ExperimentResult(
        config=config,
        x_samples=x_samples,
        initial_value=0.0,
        results=results,
        traces={},
    )


def test_wandb_reporter_streams_and_summarizes(tmp_path: Path, monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = _build_config(wandb_log_plots=False)
    run_context = _build_run_context(tmp_path)
    result = _build_result(config)

    reporter = WandbReporter()
    reporter.on_start(run_context, config)
    reporter.log_step("first_order", 0, 0.53, -0.08, grad_norm=0.2)
    reporter.log_step("first_order", 1, 0.52, -0.09, grad_norm=0.1, step_size=0.01)
    reporter.on_end(run_context, result)

    assert len(fake_wandb.init_calls) == 1
    init_payload = fake_wandb.init_calls[0]
    assert init_payload["project"] == "unit-tests"
    assert init_payload["config"]["n_grad_samples"] == 64

    defined = {name: kwargs for name, kwargs in fake_wandb.define_metric_calls}
    assert "curve/first_order/step" in defined
    assert defined["curve/first_order/objective"]["step_metric"] == "curve/first_order/step"
    assert defined["curve/first_order/u"]["step_metric"] == "curve/first_order/step"

    curve_payloads = [payload for payload, _ in fake_wandb.log_calls if "curve/first_order/objective" in payload]
    assert len(curve_payloads) == 2
    assert "curve/first_order/theta_grad_norm" in curve_payloads[0]
    assert "curve/first_order/step_size" in curve_payloads[1]

    final_payloads = [payload for payload, _ in fake_wandb.log_calls if "final/first_order/value" in payload]
    assert len(final_payloads) == 1
    assert "final/spsa/value" in final_payloads[0]
    assert fake_wandb.finish_calls == 1


def test_wandb_reporter_allowlist_filters_metrics(tmp_path: Path, monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = _build_config(wandb_estimator_allowlist=("spsa",), wandb_log_plots=False)
    run_context = _build_run_context(tmp_path)
    result = _build_result(config)

    reporter = ReporterStack([WandbReporter()])
    reporter.on_start(run_context, config)
    reporter.log_step("first_order", 0, 0.5, -0.1, grad_norm=0.3)
    reporter.log_step("spsa", 0, 0.6, -0.2, grad_norm=0.4)
    reporter.on_end(run_context, result)

    flattened_keys = {
        key
        for payload, _ in fake_wandb.log_calls
        for key in payload.keys()
    }
    defined_metrics = {name for name, _ in fake_wandb.define_metric_calls}
    assert "curve/first_order/step" not in defined_metrics
    assert "curve/spsa/step" in defined_metrics
    assert "curve/first_order/objective" not in flattened_keys
    assert "curve/spsa/objective" in flattened_keys
    assert "final/first_order/value" not in flattened_keys
    assert "final/spsa/value" in flattened_keys


def test_wandb_reporter_logs_plot_images(tmp_path: Path, monkeypatch) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = _build_config(wandb_log_plots=True)
    run_context = _build_run_context(tmp_path)
    run_context.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = run_context.plots_dir / "loss_curves.png"
    plot_path.write_bytes(b"png")

    reporter = WandbReporter()
    reporter.on_start(run_context, config)
    reporter.on_end(run_context, _build_result(config))

    plot_payloads = [payload for payload, _ in fake_wandb.log_calls if "plots/loss_curves" in payload]
    assert len(plot_payloads) == 1
    image = plot_payloads[0]["plots/loss_curves"]
    assert isinstance(image, _FakeImage)
    assert image.path.endswith("loss_curves.png")


def test_wandb_reporter_accepts_stein_difference_alias_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    config = _build_config(
        enabled_estimators=("stein-difference",),
        wandb_estimator_allowlist=("stein-difference",),
        wandb_log_plots=False,
    )
    run_context = _build_run_context(tmp_path)
    result = _build_result(config)

    reporter = ReporterStack([WandbReporter()])
    reporter.on_start(run_context, config)
    reporter.log_step("stein_difference", 0, 0.56, -0.11, grad_norm=0.25)
    reporter.on_end(run_context, result)

    flattened_keys = {
        key
        for payload, _ in fake_wandb.log_calls
        for key in payload.keys()
    }
    defined_metrics = {name for name, _ in fake_wandb.define_metric_calls}
    assert "curve/stein_difference/step" in defined_metrics
    assert "curve/stein_difference/objective" in flattened_keys
    assert "final/stein_difference/value" in flattened_keys
