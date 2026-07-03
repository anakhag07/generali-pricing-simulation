from __future__ import annotations

from dataclasses import dataclass

import experiments.execution as execution
from experiments.configs import get_config
from experiments.execution import default_reporter_stack, execute_experiment_run
from experiments.reporting import (
    ConsoleReporter,
    FileStepLogger,
    JsonReporter,
    PlotReporter,
    PolicyArtifactReporter,
    ReporterStack,
    WandbReporter,
)


def test_default_reporter_stack_uses_expected_order() -> None:
    config = get_config("planted_logistic_base", overrides={"wandb_enabled": False})

    stack = default_reporter_stack(config)

    assert isinstance(stack, ReporterStack)
    assert [type(reporter) for reporter in stack._reporters] == [
        ConsoleReporter,
        FileStepLogger,
        PolicyArtifactReporter,
        JsonReporter,
        PlotReporter,
    ]


def test_default_reporter_stack_appends_wandb_when_enabled() -> None:
    config = get_config(
        "planted_logistic_base",
        overrides={"wandb_enabled": True, "wandb_project": "test-project"},
    )

    stack = default_reporter_stack(config)

    assert isinstance(stack._reporters[-1], WandbReporter)


def test_default_reporter_stack_uses_supplied_json_reporter() -> None:
    config = get_config("planted_logistic_base", overrides={"wandb_enabled": False})
    custom_json = JsonReporter(summary_name="summary-seed-7.json")

    stack = default_reporter_stack(config, json_reporter=custom_json)

    # Ordering contract preserved: injected reporter occupies the JsonReporter slot,
    # right after PolicyArtifactReporter.
    assert [type(reporter) for reporter in stack._reporters] == [
        ConsoleReporter,
        FileStepLogger,
        PolicyArtifactReporter,
        JsonReporter,
        PlotReporter,
    ]
    assert stack._reporters[3] is custom_json


def test_default_reporter_stack_omits_plots_when_include_plots_false() -> None:
    config = get_config("planted_logistic_base", overrides={"wandb_enabled": False})

    stack = default_reporter_stack(config, include_plots=False)

    assert [type(reporter) for reporter in stack._reporters] == [
        ConsoleReporter,
        FileStepLogger,
        PolicyArtifactReporter,
        JsonReporter,
    ]


def test_default_reporter_stack_keeps_plots_before_wandb_without_plots() -> None:
    config = get_config(
        "planted_logistic_base",
        overrides={"wandb_enabled": True, "wandb_project": "test-project"},
    )

    stack = default_reporter_stack(config, include_plots=False)

    # Plots dropped, but W&B stays last so the plots-before-wandb contract holds.
    assert not any(isinstance(reporter, PlotReporter) for reporter in stack._reporters)
    assert isinstance(stack._reporters[-1], WandbReporter)


@dataclass(frozen=True)
class _FakeRunContext:
    experiment_name: str


class _FakeReporterStack:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def on_start(self, run_context, config) -> None:
        self.events.append(("start", run_context.experiment_name))

    def on_end(self, run_context, result) -> None:
        self.events.append(("end", run_context.experiment_name))


def test_execute_experiment_run_returns_context_and_result(monkeypatch) -> None:
    config = get_config("planted_logistic_base", overrides={"plot": False, "wandb_enabled": False})
    reporter_stack = _FakeReporterStack()
    fake_result = object()

    monkeypatch.setattr(
        execution,
        "create_run_context",
        lambda name, runs_root="outputs": _FakeRunContext(experiment_name=name),
    )
    monkeypatch.setattr(execution, "run_experiment", lambda cfg, step_reporter=None: fake_result)

    executed = execute_experiment_run(
        "one-run",
        config,
        runs_root="unused",
        reporter_stack_factory=lambda cfg: reporter_stack,
    )

    assert executed.name == "one-run"
    assert executed.config is config
    assert executed.result is fake_result
    assert executed.run_context.experiment_name == "one-run"
    assert reporter_stack.events == [("start", "one-run"), ("end", "one-run")]
