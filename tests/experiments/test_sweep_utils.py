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


def test_generate_sweep_runs_accepts_quadratic_dimension_axis() -> None:
    runs = generate_sweep_runs(
        base_preset="quadratic_base",
        override_grid={"dimension": [2, 5], "plot": [False]},
        display_keys=("dimension",),
    )

    assert [run_name for run_name, _, _ in runs] == ["dimension-2", "dimension-5"]
    assert [config.objective.dimension for _, config, _ in runs] == [2, 5]
    assert [config.theta0.shape for _, config, _ in runs] == [(2,), (5,)]


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


def test_generate_sweep_runs_accepts_explicit_run_name() -> None:
    runs = generate_sweep_runs(
        base_preset="planted_logistic_base",
        override_list=[{"_run_name": "noise-std-0.25", "sigma": 0.03}],
    )

    run_name, config, override = runs[0]

    assert run_name == "noise-std-0.25"
    assert config.sigma == 0.03
    assert override == {"sigma": 0.03}


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


@dataclass(frozen=True)
class _FakeExecutedRun:
    result: object
    run_context: _FakeRunContext


def test_run_preset_sweep_uses_project_name_as_runs_root(monkeypatch, tmp_path) -> None:
    captured: dict[str, str] = {}

    def fake_execute_experiment_run(run_name: str, config, *, runs_root, run_metadata=None):
        captured["run_name"] = run_name
        captured["runs_root"] = runs_root
        captured["run_metadata"] = run_metadata
        return _FakeExecutedRun(
            result=object(),
            run_context=_FakeRunContext(experiment_name=run_name),
        )

    monkeypatch.setattr(sweep_utils, "execute_experiment_run", fake_execute_experiment_run)

    results = run_preset_sweep(
        base_preset="planted_logistic_base",
        override_list=[{"sigma": 0.03, "n_grad_samples": 64, "wandb_enabled": True}],
        runs_root=str(tmp_path),
        project_name="one_project",
        display_keys=("sigma", "n_grad_samples"),
    )

    assert captured["run_name"] == "sigma-0.03__ngrad-64"
    assert captured["runs_root"] == tmp_path / "one_project"
    assert captured["run_metadata"] == {
        "preset_name": "planted_logistic_base",
        "variant_name": "sigma-0.03__ngrad-64",
        "overrides": {"sigma": 0.03, "n_grad_samples": 64, "wandb_enabled": True},
    }
    assert len(results) == 1
    assert results[0].run_name == "sigma-0.03__ngrad-64"
    assert results[0].overrides["sigma"] == 0.03
    assert results[0].run_context.experiment_name == "sigma-0.03__ngrad-64"
