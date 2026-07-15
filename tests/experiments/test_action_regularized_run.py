"""Runner-level tests for action-regularized objectives."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.run import run_experiment
from objective import FixedRegressionObjective, LinearPolicy
from objective.utils import value_for_reporting


def test_run_experiment_optimizes_regularized_objective_but_reports_raw_value() -> None:
    policy = LinearPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.4],
        beta_2=-0.8,
        beta_3=[0.3],
        beta_4=0.5,
    )
    theta0 = np.asarray([0.05, 0.1], dtype=float)
    x_fixed = np.asarray([[-1.0], [-0.2], [0.4], [1.0]], dtype=float)
    config = ExperimentConfig(
        state_dim=1,
        n_samples=x_fixed.shape[0],
        objective=objective,
        theta0=theta0,
        x_fixed=x_fixed,
        step_rule="constant",
        perturbation_space="theta",
        enabled_estimators=("first_order",),
        t_steps=1,
        step_size=0.01,
        batch_size=2,
        proximal_weight=0.5,
        u_reference=np.asarray([-0.3, -0.1, 0.1, 0.3], dtype=float),
        plot=False,
    )

    result = run_experiment(config)
    trace = result.traces["first_order"]
    estimator = result.results["first_order"]

    assert trace.proximal_penalty_values is not None
    assert len(trace.proximal_penalty_values) == len(trace.steps)
    assert estimator.value == pytest.approx(
        value_for_reporting(objective, estimator.theta, result.x_samples)
    )
