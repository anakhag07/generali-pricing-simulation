from __future__ import annotations

from objective.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec
from experiments.run import run_experiment


def test_run_experiment_single_estimator() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        seed=3,
        state_dim=1,
        objective_model=objective_model,
        policy_spec=default_policy_spec(1),
        n_samples=1,
        step_rule="constant",
        t_steps=1,
        step_size=0.01,
        n_grad_samples=1,
        plot=False,
        enabled_estimators=("first_order",),
    )
    result = run_experiment(config)
    assert isinstance(result.initial_value, float)
    assert "first_order" in result.results
    assert isinstance(result.results["first_order"].u, float)
    assert "zeroth_order" not in result.results
    assert "lbfgs" not in result.results


def test_run_experiment_spsa_only() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        seed=3,
        state_dim=1,
        objective_model=objective_model,
        policy_spec=default_policy_spec(1),
        n_samples=1,
        step_rule="constant",
        t_steps=2,
        step_size=0.01,
        n_grad_samples=2,
        plot=False,
        enabled_estimators=("spsa",),
    )
    result = run_experiment(config)
    assert "spsa" in result.results
    assert isinstance(result.results["spsa"].u, float)
    assert "first_order" not in result.results
    assert "zeroth_order" not in result.results
    assert "lbfgs" not in result.results
