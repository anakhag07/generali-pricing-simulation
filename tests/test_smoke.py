from __future__ import annotations

from experiments.runner import ExperimentConfig, run_demo


def test_run_demo_smoke() -> None:
    config = ExperimentConfig(t_steps=1, n_samples=2)
    value, u_first, u_zero = run_demo(config)
    assert isinstance(value, float)
    assert isinstance(u_first, float)
    assert isinstance(u_zero, float)
