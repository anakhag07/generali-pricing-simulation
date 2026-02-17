from __future__ import annotations

from experiments.configs import get_config
from experiments.run import run_experiment


def test_run_experiment_smoke() -> None:
    config = get_config("smoke")
    value, u_first, u_zero, u_lbfgs = run_experiment(config)
    assert isinstance(value, float)
    assert isinstance(u_first, float)
    assert isinstance(u_zero, float)
    assert isinstance(u_lbfgs, float)
