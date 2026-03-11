from __future__ import annotations

import pytest

from experiments.configs import get_config
from experiments.sweep_utils import (
    apply_config_overrides,
    expand_override_grid,
    generate_sweep_runs,
    make_sweep_name,
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
    assert run_name_1.startswith("planted_logistic_base__sweep_001")
    assert run_name_2.startswith("planted_logistic_base__sweep_002")
    assert config_1.t_steps == 10
    assert config_2.t_steps == 10
    assert override_1["seed"] == 7
    assert override_2["seed"] == 8


def test_make_sweep_name_is_deterministic() -> None:
    name = make_sweep_name("fixed_regression_base", 3, {"sigma": 0.05, "seed": 7})
    assert name == "fixed_regression_base__sweep_003__seed-7__sigma-0.05"
