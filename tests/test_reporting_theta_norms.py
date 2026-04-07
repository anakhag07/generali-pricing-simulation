from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0
from experiments.reporters import RunContext, _build_summary_payload
from experiments.results import EstimatorResult, ExperimentResult, OptimizationTrace
from objective import FixedRegressionObjective, LinearPolicy, ModelBasedObjective, SoftmaxPolicy
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


def _build_model_based_result() -> ExperimentResult:
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )

    acceptance_model, loss_model = load_model_artifacts("glm")
    objective = ModelBasedObjective(
        policy=SoftmaxPolicy(),
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=extract_glm_u_coef(acceptance_model),
    )
    theta0 = np.array([0.4] + [0.01] * 12, dtype=float)
    config = ExperimentConfig(
        state_dim=12,
        objective=objective,
        theta0=theta0,
        n_samples=5,
        x_fixed=load_x_array("glm", n_rows=5),
        step_rule="constant",
        perturbation_space="u",
        t_steps=3,
        plot=False,
        enabled_estimators=("first_order",),
    )
    trace = OptimizationTrace(
        steps=[0, 1],
        u_values=[1.1, 1.12],
        objective_values=[-20.0, -21.0],
        u_grad_estimates=[0.3, 0.2],
        theta_values=[theta0.copy(), theta0.copy()],
        optimizer_status=0,
        optimizer_message="ok",
    )
    return ExperimentResult(
        config=config,
        x_samples=config.x_fixed,
        initial_value=-20.0,
        results={
            "first_order": EstimatorResult(
                theta=theta0.copy(),
                u=1.12,
                value=-21.0,
                time=0.01,
            )
        },
        traces={"first_order": trace},
    )


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


def test_log_summary_prints_model_coefficients(capsys) -> None:
    result = _build_model_based_result()
    log_summary(result)
    captured = capsys.readouterr().out
    assert "Objective: f(u; x) = p_acc(x, u) * (loss_hat(x) - u * premium(x))" in captured
    assert "p_churn(x, u) = sigmoid(beta_0 + beta_x^T x_acc + beta_u * u)" in captured
    assert "p_acc(x, u) = 1 - p_churn(x, u)" in captured
    assert "loss_hat(x) = gamma_0 + gamma_x^T x_loss" in captured
    assert "beta_x =" in captured
    assert "beta_u =" in captured
    assert "gamma_x =" in captured
    assert "gamma_0 =" in captured


def test_summary_payload_contains_model_coefficients(tmp_path: Path) -> None:
    result = _build_model_based_result()
    run_context = RunContext(
        experiment_name="glm-test",
        run_id="20260310_000000",
        run_dir=tmp_path / "run",
        plots_dir=tmp_path / "run" / "plots",
        started_at=datetime(2026, 3, 10, 0, 0, 0),
    )

    payload = _build_summary_payload(run_context, result)
    coeffs = payload["model_coefficients"]
    assert payload["model_formulas"]["acceptance"] == "p_acc(x, u) = 1 - p_churn(x, u)"
    assert set(coeffs) == {"churn", "loss"}
    assert len(coeffs["churn"]["x_feature_names"]) == len(coeffs["churn"]["x_coef"])
    assert len(coeffs["loss"]["x_feature_names"]) == len(coeffs["loss"]["x_coef"])
    assert "u_coef" in coeffs["churn"]
    assert "u_coef" not in coeffs["loss"]
