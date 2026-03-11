from __future__ import annotations

from dataclasses import replace

from experiments.configs import get_config
from experiments.run import run_experiment


def test_run_experiment_fixed_regression_base_smoke() -> None:
    config = replace(
        get_config("fixed_regression_base"),
        n_samples=2,
        t_steps=1,
        plot=False,
        wandb_enabled=False,
    )
    result = run_experiment(config)
    assert isinstance(result.initial_value, float)
    assert "first_order" in result.results
    assert "zeroth_order" in result.results
    assert "spsa" in result.results
    assert isinstance(result.results["first_order"].u, float)
    assert isinstance(result.results["zeroth_order"].u, float)
    assert isinstance(result.results["spsa"].u, float)
