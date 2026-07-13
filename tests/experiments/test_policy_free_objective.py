from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.reporting.json_summary import build_summary_payload
from experiments.run import run_experiment
from objective import QuadraticObjective


def _config(*, dimension: int = 3, theta0: np.ndarray | None = None) -> ExperimentConfig:
    if theta0 is None:
        theta0 = np.ones(dimension, dtype=float) / np.sqrt(dimension)
    return ExperimentConfig(
        state_dim=1,
        n_samples=1,
        x_fixed=np.zeros((1, 1), dtype=float),
        objective=QuadraticObjective(dimension=dimension),
        theta0=theta0,
        step_rule="l-bfgs-b",
        perturbation_space="theta",
        enabled_estimators=("first_order", "finite_difference"),
        t_steps=20,
        sigma=1e-4,
        plot=False,
        wandb_enabled=False,
    )


def test_policy_free_quadratic_run_converges_and_reports_no_actions(tmp_path) -> None:
    result = run_experiment(_config())

    for estimator_result in result.results.values():
        assert np.linalg.norm(estimator_result.theta) < 1e-6
        assert estimator_result.value < 1e-12
        assert estimator_result.u is None
    for evaluation in result.train_metrics.values():
        assert evaluation.objective_value < 1e-12
        assert evaluation.mean_u is None
        assert evaluation.u_q25 is None
        assert evaluation.u_q75 is None

    context = RunContext(
        experiment_name="quadratic",
        run_id="test",
        run_dir=tmp_path,
        plots_dir=tmp_path / "plots",
        started_at=datetime(2026, 1, 1),
    )
    payload = build_summary_payload(context, result)
    assert payload["estimators"]["first_order"]["final_u"] is None
    assert payload["estimators"]["first_order"]["train"]["mean_u"] is None
    json.dumps(payload, allow_nan=False)


def test_policy_free_objective_dimension_controls_default_initialization() -> None:
    config = _config(dimension=4, theta0=np.ones(4))
    config = replace(config, theta0=None, t_steps=1)

    result = run_experiment(config)

    assert result.config.theta0 is not None
    assert result.config.theta0.shape == (4,)


def test_policy_free_objective_validates_theta_dimension() -> None:
    with pytest.raises(ValueError, match="objective requires 3"):
        _config(dimension=3, theta0=np.ones(2))


def test_policy_free_objective_rejects_action_only_estimators() -> None:
    with pytest.raises(ValueError, match="requires an objective with a policy"):
        ExperimentConfig(
            state_dim=1,
            n_samples=1,
            objective=QuadraticObjective(dimension=2),
            theta0=np.ones(2),
            step_rule="l-bfgs-b",
            perturbation_space="theta",
            enabled_estimators=("constant",),
        )


def test_quadratic_config_serializes_dimension() -> None:
    assert _config(dimension=5).to_dict()["objective"] == {
        "type": "QuadraticObjective",
        "dimension": 5,
    }
