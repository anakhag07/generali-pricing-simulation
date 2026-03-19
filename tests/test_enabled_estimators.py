from __future__ import annotations

from objective import FixedRegressionObjective, SoftmaxPolicy
from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0, default_policy
from experiments.run import run_experiment


def test_run_experiment_single_estimator() -> None:
    objective = FixedRegressionObjective.from_parameters(
        policy=default_policy(1),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        seed=3,
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=1,
        step_rule="l-bfgs-b",
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
    assert "gauss_stein" not in result.results
    assert "lbfgs" not in result.results


def test_run_experiment_spsa_only() -> None:
    objective = FixedRegressionObjective.from_parameters(
        policy=default_policy(1),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        seed=3,
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=1,
        step_rule="l-bfgs-b",
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
    assert "gauss_stein" not in result.results
    assert "lbfgs" not in result.results


def test_run_experiment_stein_difference_only() -> None:
    objective = FixedRegressionObjective.from_parameters(
        policy=default_policy(1),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        seed=3,
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=1,
        step_rule="l-bfgs-b",
        t_steps=2,
        step_size=0.01,
        n_grad_samples=2,
        plot=False,
        enabled_estimators=("stein_difference",),
    )
    result = run_experiment(config)
    assert "stein_difference" in result.results
    assert isinstance(result.results["stein_difference"].u, float)
    assert "first_order" not in result.results
    assert "gauss_stein" not in result.results
    assert "spsa" not in result.results
