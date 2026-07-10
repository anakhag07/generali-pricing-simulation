from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.configs import get_config
from objective.objectives import BiasedObjective
from scripts import run_planted_logistic_action_bias_sweep as script


def _args(**overrides: object) -> SimpleNamespace:
    payload = {
        "seed": 7,
        "n_samples": 8,
        "t_steps": 2,
        "lambda_bias": [0.0, 0.01],
        "project_name": script.PROJECT_NAME,
        "output_dir": None,
        "per_run_plots": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_lambda_grid_and_run_names() -> None:
    assert script.LAMBDA_BIAS_VALUES == (0.0, 0.01, 0.05, 0.1, 0.2)
    assert script._run_name(0.01) == "lambda-bias-0p01"
    assert script._run_name(0.2) == "lambda-bias-0p2"


def test_biased_config_uses_biased_objective() -> None:
    config = script._biased_config(0.05, _args())

    assert config.enabled_estimators == (script.ESTIMATOR,)
    assert isinstance(config.objective, BiasedObjective)
    assert config.objective.lambda_bias == 0.05
    assert config.n_samples == 8
    assert config.t_steps == 2


def test_row_for_lambda_uses_signed_optimism_gap() -> None:
    base_objective = get_config(script.BASE_PRESET).objective
    biased_objective = BiasedObjective(base_objective, lambda_bias=0.1)
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
        ),
        run_context=SimpleNamespace(run_dir="biased-dir"),
    )

    row = script._row_for_lambda(0.1, oracle_executed, biased_executed)

    assert row["optimism_gap"] == pytest.approx(
        row["surrogate_objective_at_biased_solution"] - row["true_objective_at_biased_solution"]
    )
    assert row["optimism_gap"] == pytest.approx(-0.1 * row["mean_action_biased_solution"])
    assert row["oracle_run_dir"] == "oracle-dir"
    assert row["biased_run_dir"] == "biased-dir"


def test_write_rows_writes_expected_csv(tmp_path) -> None:
    row = {field: 0.0 for field in script.FIELDNAMES}
    row["oracle_run_dir"] = "oracle"
    row["biased_run_dir"] = "biased"

    output_path = script._write_rows(tmp_path, [row])

    text = output_path.read_text(encoding="utf-8")
    assert output_path.name == script.OUTPUT_CSV
    assert "lambda_bias" in text
    assert "optimism_gap" in text
