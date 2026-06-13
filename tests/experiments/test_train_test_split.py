from __future__ import annotations

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0, default_policy
from experiments.run import run_experiment
from objective import FixedRegressionObjective
from objective.utils import value_for_reporting


def _objective() -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=default_policy(1),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )


def _config(**overrides: object) -> ExperimentConfig:
    x_fixed = np.linspace(-1.0, 1.0, 10, dtype=float).reshape(-1, 1)
    payload = {
        "seed": 3,
        "state_dim": 1,
        "objective": _objective(),
        "theta0": default_theta0(1),
        "n_samples": x_fixed.shape[0],
        "x_fixed": x_fixed,
        "step_rule": "constant",
        "perturbation_space": "theta",
        "t_steps": 1,
        "step_size": 0.01,
        "n_grad_samples": 1,
        "plot": False,
        "enabled_estimators": ("first_order",),
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_default_split_preserves_full_training_batch() -> None:
    config = _config()

    result = run_experiment(config)

    assert result.x_samples.shape == (10, 1)
    assert result.x_test is None
    np.testing.assert_array_equal(result.train_indices, np.arange(10, dtype=int))
    assert result.test_indices is not None
    assert result.test_indices.size == 0
    assert result.test_metrics == {}
    assert result.train_metrics["first_order"].n_samples == 10


def test_train_test_split_optimizes_train_and_evaluates_test() -> None:
    config = _config(train_fraction=0.6, test_fraction=0.4)

    result = run_experiment(config)
    estimator = result.results["first_order"]

    assert result.x_samples.shape == (6, 1)
    assert result.x_test is not None
    assert result.x_test.shape == (4, 1)
    assert result.train_indices is not None
    assert result.test_indices is not None
    assert set(result.train_indices.tolist()).isdisjoint(result.test_indices.tolist())
    assert sorted([*result.train_indices.tolist(), *result.test_indices.tolist()]) == list(range(10))

    train_metrics = result.train_metrics["first_order"]
    test_metrics = result.test_metrics["first_order"]
    assert estimator.value == pytest.approx(train_metrics.objective_value)
    assert estimator.u == pytest.approx(train_metrics.mean_u)
    assert train_metrics.objective_value == pytest.approx(
        value_for_reporting(result.config.objective, estimator.theta, result.x_samples)
    )
    assert test_metrics.objective_value == pytest.approx(
        value_for_reporting(result.config.objective, estimator.theta, result.x_test)
    )
    assert test_metrics.n_samples == 4


def test_train_test_split_preserves_source_row_indices() -> None:
    source_row_indices = np.arange(100, 110, dtype=int)
    config = _config(
        train_fraction=0.7,
        test_fraction=0.3,
        x_fixed_row_indices=source_row_indices,
    )

    result = run_experiment(config)

    assert result.train_row_indices is not None
    assert result.test_row_indices is not None
    np.testing.assert_array_equal(result.train_row_indices, source_row_indices[result.train_indices])
    np.testing.assert_array_equal(result.test_row_indices, source_row_indices[result.test_indices])
