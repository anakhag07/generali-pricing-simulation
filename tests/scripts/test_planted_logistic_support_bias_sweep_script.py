from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.configs import get_config
from objective.objectives import BiasedObjective, UpperSupportHingeBias
from scripts import run_planted_logistic_support_bias_sweep as script


def _args(**overrides: object) -> SimpleNamespace:
    payload = {
        "seed": 7,
        "n_samples": 8,
        "t_steps": 2,
        "lambda_bias": [0.0, 0.05],
        "support_radius": [0.05, 0.1],
        "smooth_tau": None,
        "project_name": script.PROJECT_NAME,
        "output_dir": None,
        "per_run_plots": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_support_sweep_axes_and_run_names() -> None:
    assert script.LAMBDA_BIAS_VALUES == (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
    assert script.SUPPORT_RADII == (0.02, 0.05, 0.1, 0.2)
    assert script._run_name(0.05, 0.1) == "lambda-0p05__support-radius-0p1"


def test_biased_config_uses_upper_support_hinge_bias() -> None:
    config = script._biased_config(0.05, 0.1, _args(smooth_tau=0.01))

    assert config.enabled_estimators == (script.ESTIMATOR,)
    assert isinstance(config.objective, BiasedObjective)
    assert isinstance(config.objective.bias, UpperSupportHingeBias)
    assert config.objective.bias.lambda_bias == 0.05
    assert config.objective.bias.support_center == 0.1
    assert config.objective.bias.support_radius == 0.1
    assert config.objective.bias.support_upper == 0.2
    assert config.objective.bias.smooth_tau == 0.01


def test_row_for_variant_reports_support_metrics() -> None:
    base_objective = get_config(script.BASE_PRESET).objective
    biased_objective = BiasedObjective(
        base_objective,
        bias=UpperSupportHingeBias(lambda_bias=0.1, support_center=0.1, support_radius=0.05),
    )
    x = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]], dtype=float)
    oracle_theta = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
    biased_theta = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    oracle_executed = SimpleNamespace(
        result=SimpleNamespace(
            config=SimpleNamespace(objective=base_objective),
            x_samples=x,
            results={script.ESTIMATOR: SimpleNamespace(theta=oracle_theta)},
        ),
        run_context=SimpleNamespace(run_dir="oracle-dir"),
    )
    biased_executed = SimpleNamespace(
        result=SimpleNamespace(
            config=SimpleNamespace(objective=biased_objective),
            x_samples=x,
            results={script.ESTIMATOR: SimpleNamespace(theta=biased_theta)},
            traces={script.ESTIMATOR: SimpleNamespace(optimizer_success=True, optimizer_status=0)},
        ),
        run_context=SimpleNamespace(run_dir="biased-dir"),
    )

    row = script._row_for_variant(0.1, 0.05, oracle_executed, biased_executed)

    assert row["support_upper"] == pytest.approx(0.15)
    assert row["support_violation_rate"] == pytest.approx(1.0)
    assert row["mean_support_excess"] > 0.0
    assert row["optimism_gap"] == pytest.approx(-0.1 * row["mean_support_excess"])
    assert row["theta_l2_from_oracle"] == pytest.approx(np.linalg.norm(biased_theta - oracle_theta))


def test_write_outputs_writes_csv_and_plots(tmp_path) -> None:
    rows = []
    for support_radius in (0.05, 0.1):
        for lambda_bias in (0.0, 0.05):
            rows.append(
                {
                    "lambda_bias": lambda_bias,
                    "support_radius": support_radius,
                    "support_upper": 0.1 + support_radius,
                    "smooth_tau": "",
                    "true_objective_at_oracle": 1.0,
                    "true_objective_at_biased_solution": 1.0 + lambda_bias,
                    "true_gap": lambda_bias,
                    "surrogate_objective_at_biased_solution": 1.0,
                    "optimism_gap": -lambda_bias,
                    "mean_action_oracle": 0.1,
                    "mean_action_biased_solution": 0.2,
                    "support_violation_rate": 1.0,
                    "mean_support_excess": lambda_bias,
                    "max_support_excess": lambda_bias,
                    "theta_l2_from_oracle": lambda_bias,
                    "optimizer_success": True,
                    "optimizer_status": 0,
                    "oracle_run_dir": "oracle",
                    "biased_run_dir": "biased",
                }
            )

    output_path = script._write_outputs(tmp_path, rows)

    assert output_path.name == script.OUTPUT_CSV
    assert (tmp_path / "plots" / "true_gap_heatmap.png").exists()
    assert (tmp_path / "plots" / "mean_support_excess_heatmap.png").exists()
    assert (tmp_path / "plots" / "true_gap_vs_support_excess.png").exists()
