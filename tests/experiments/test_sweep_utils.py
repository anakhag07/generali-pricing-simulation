from __future__ import annotations

from dataclasses import dataclass

import pytest

from experiments.configs import get_config
import experiments.sweep_utils as sweep_utils
from experiments.sweep_utils import (
    apply_config_overrides,
    expand_override_grid,
    generate_sweep_runs,
    make_display_name,
    make_sweep_name,
    run_preset_sweep,
)


def test_expand_override_grid_cartesian_product() -> None:
    grid = {
        "seed": [7, 11],
        "sigma": [0.03, 0.05],
    }
    combos = expand_override_grid(grid)
    assert len(combos) == 4
    assert {tuple(sorted(combo.items())) for combo in combos} == {
        (("seed", 7), ("sigma", 0.03)),
        (("seed", 7), ("sigma", 0.05)),
        (("seed", 11), ("sigma", 0.03)),
        (("seed", 11), ("sigma", 0.05)),
    }


def test_apply_config_overrides_updates_fields() -> None:
    base = get_config("fixed_regression_base")
    updated = apply_config_overrides(base, {"seed": 13, "sigma": 0.02, "plot": False})
    assert updated.seed == 13
    assert updated.sigma == 0.02
    assert updated.plot is False
    assert base.seed == 7


def test_apply_config_overrides_rejects_unknown_field() -> None:
    base = get_config("fixed_regression_base")
    with pytest.raises(ValueError, match="Unknown config override fields"):
        apply_config_overrides(base, {"does_not_exist": 1})


def test_generate_sweep_runs_from_grid() -> None:
    runs = generate_sweep_runs(
        base_preset="planted_logistic_base",
        override_grid={"seed": [7, 8], "t_steps": [10]},
    )
    assert len(runs) == 2
    run_name_1, config_1, override_1 = runs[0]
    run_name_2, config_2, override_2 = runs[1]
    assert run_name_1 == "seed-7__t_steps-10"
    assert run_name_2 == "seed-8__t_steps-10"
    assert config_1.t_steps == 10
    assert config_2.t_steps == 10
    assert override_1["seed"] == 7
    assert override_2["seed"] == 8


def test_generate_sweep_runs_accepts_real_data_factory_overrides() -> None:
    runs = generate_sweep_runs(
        base_preset="real_data_glm_base",
        override_list=[
            {
                "policy_kind": "linear",
                "feature_order": "quadratic",
                "n_samples": 20,
                "plot": False,
                "wandb_enabled": False,
            }
        ],
        display_keys=("policy_kind", "feature_order"),
    )
    run_name, config, override = runs[0]
    assert run_name == "policy_kind-linear__feature_order-quadratic"
    assert config.n_samples == 20
    assert override["policy_kind"] == "linear"


def test_make_sweep_name_is_deterministic() -> None:
    name = make_sweep_name("fixed_regression_base", 3, {"sigma": 0.05, "seed": 7})
    assert name == "fixed_regression_base__sweep_003__seed-7__sigma-0.05"


def test_make_display_name_uses_selected_keys_in_order() -> None:
    name = make_display_name(
        "fixed_regression_base",
        3,
        {"sigma": 0.05, "seed": 7, "n_grad_samples": 64},
        display_keys=("sigma", "n_grad_samples"),
    )
    assert name == "sigma-0.05__ngrad-64"


@dataclass(frozen=True)
class _FakeRunContext:
    experiment_name: str


class _FakeReporterStack:
    def __init__(self, reporters) -> None:
        self.reporters = reporters

    def on_start(self, run_context, config) -> None:
        return None

    def on_end(self, run_context, result) -> None:
        return None


def test_run_preset_sweep_uses_project_name_as_runs_root(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    def fake_create_run_context(experiment_name: str, runs_root: str = "outputs") -> _FakeRunContext:
        captured["experiment_name"] = experiment_name
        captured["runs_root"] = runs_root
        return _FakeRunContext(experiment_name=experiment_name)

    monkeypatch.setattr(sweep_utils, "create_run_context", fake_create_run_context)
    monkeypatch.setattr(sweep_utils, "ConsoleReporter", lambda verbose=False: object())
    monkeypatch.setattr(sweep_utils, "FileStepLogger", lambda: object())
    monkeypatch.setattr(sweep_utils, "JsonReporter", lambda: object())
    monkeypatch.setattr(sweep_utils, "PlotReporter", lambda: object())
    monkeypatch.setattr(sweep_utils, "WandbReporter", lambda: object())
    monkeypatch.setattr(sweep_utils, "ReporterStack", _FakeReporterStack)
    monkeypatch.setattr(sweep_utils, "run_experiment", lambda config, step_reporter=None: object())

    run_preset_sweep(
        base_preset="planted_logistic_base",
        override_list=[{"sigma": 0.03, "n_grad_samples": 64, "wandb_enabled": True}],
        runs_root=str(tmp_path),
        project_name="one_project",
        display_keys=("sigma", "n_grad_samples"),
    )

    assert captured["experiment_name"] == "sigma-0.03__ngrad-64"
    assert captured["runs_root"] == str(tmp_path / "one_project")
