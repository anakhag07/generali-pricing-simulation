from __future__ import annotations

from dataclasses import replace

import pytest

from data.loader import load_mean_observed_acceptance
from experiments.configs import get_config
from experiments.run import run_experiment
from objective import value_for_reporting


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
    for estimator in config.enabled_estimators:
        assert estimator in result.results
        assert isinstance(result.results[estimator].u, float)


def test_run_experiment_lagrangian_glm_smoke_reports_raw_objective() -> None:
    config = get_config(
        "real_data_glm_base",
        overrides={
            "policy_kind": "softmax",
            "n_samples": 30,
            "t_steps": 1,
            "n_grad_samples": 2,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
            "acceptance_floor": load_mean_observed_acceptance("glm"),
            "lagrangian_lambda": 2.0,
            "enabled_estimators": (
                "first_order",
                "finite_difference",
                "gauss_stein",
                "spsa",
                "stein_difference",
            ),
        },
    )
    result = run_experiment(config)

    assert result.config.lagrangian_lambda == pytest.approx(2.0)
    assert result.initial_value == pytest.approx(
        value_for_reporting(result.config.objective, result.config.theta0, result.x_samples)
    )
    for name, estimator_result in result.results.items():
        assert estimator_result.value == pytest.approx(
            value_for_reporting(result.config.objective, estimator_result.theta, result.x_samples)
        )
        assert estimator_result.mean_acceptance is not None
