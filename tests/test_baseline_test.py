from __future__ import annotations

from experiments.configs import get_config
from experiments.run import run_experiment


def test_run_experiment_baseline_test() -> None:
    config = get_config("baseline_test")
    result = run_experiment(config)
    assert isinstance(result.initial_value, float)
    assert "first_order" in result.results
    assert "zeroth_order" in result.results
    assert "lbfgs" in result.results
    assert isinstance(result.results["first_order"].u, float)
    assert isinstance(result.results["zeroth_order"].u, float)
    assert isinstance(result.results["lbfgs"].u, float)
