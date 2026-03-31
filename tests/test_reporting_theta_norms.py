from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0
from experiments.reporters import RunContext, _build_summary_payload
from experiments.results import EstimatorResult, ExperimentResult, OptimizationTrace
from objective import FixedRegressionObjective, LinearPolicy
from reporting.logging import log_summary


def _build_result() -> ExperimentResult:
    policy = LinearPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.asarray([0.2], dtype=float),
        beta_2=-0.8,
        beta_3=np.asarray([0.1], dtype=float),
        beta_4=0.3,
    )
    theta0 = np.asarray([0.1, 0.2], dtype=float)
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=1,
        step_rule="constant",
        perturbation_space="theta",
        t_steps=3,
        plot=False,
        enabled_estimators=("first_order",),
    )
    trace = OptimizationTrace(
        steps=[0, 1],
        u_values=[0.1, 0.2],
        objective_values=[1.0, 0.8],
        u_grad_estimates=[0.4, 0.2],
        theta_values=[theta0.copy(), np.asarray([0.3, -0.2], dtype=float)],
        optimizer_status=0,
        optimizer_message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )
    result = ExperimentResult(
        config=config,
        x_samples=np.array([[0.5]], dtype=float),
        initial_value=1.0,
        results={
            "first_order": EstimatorResult(
                theta=np.asarray([0.3, -0.2], dtype=float),
                u=0.2,
                value=0.8,
                time=0.01,
            )
        },
        traces={"first_order": trace},
    )
    return result


def test_log_summary_prints_theta_norms(capsys) -> None:
    result = _build_result()
    log_summary(result)
    captured = capsys.readouterr().out
    assert "Final theta norms (first-order):" in captured
    assert "||theta||_2=" in captured
    assert "||theta-theta0||_2=" in captured


def test_summary_payload_contains_theta_norms(tmp_path: Path) -> None:
    result = _build_result()
    run_context = RunContext(
        experiment_name="test",
        run_id="20260310_000000",
        run_dir=tmp_path / "run",
        plots_dir=tmp_path / "run" / "plots",
        started_at=datetime(2026, 3, 10, 0, 0, 0),
    )

    payload = _build_summary_payload(run_context, result)
    estimator_payload = payload["estimators"]["first_order"]
    assert "theta_l2_norm" in estimator_payload
    assert "theta_delta_l2_norm" in estimator_payload
    assert estimator_payload["optimizer_status"] == 0
    assert "CONVERGENCE" in estimator_payload["optimizer_message"]
    assert float(estimator_payload["theta_l2_norm"]) > 0.0
    assert float(estimator_payload["theta_delta_l2_norm"]) > 0.0
