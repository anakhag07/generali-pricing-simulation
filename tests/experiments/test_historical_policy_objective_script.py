from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_historical_policy_objective as script


class _ConstantPolicyObjective:
    def policy_value(self, theta: np.ndarray, x_batch: object) -> np.ndarray:
        return np.full(int(x_batch.shape[0]), float(theta[0]), dtype=float)


def _config(n_rows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        objective=_ConstantPolicyObjective(),
        x_fixed=pd.DataFrame({"x": np.arange(n_rows, dtype=float)}),
    )


def _historical_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dummy_id": [101, 102],
            "id": [201, 202],
            "U": [0.1, 0.2],
            "is_churn": [0, 1],
            "Y_G_Loss": [100.0, 200.0],
            "X_policy_premium": [10.0, 20.0],
        }
    )


def test_evaluate_historical_policy_objective_uses_one_minus_is_churn() -> None:
    evaluation = script.evaluate_historical_policy_objective(
        config=_config(),
        theta=np.asarray([0.5], dtype=float),
        row_indices=np.asarray([5, 6], dtype=int),
        historical_rows=_historical_rows(),
        estimator="first_order",
    )

    np.testing.assert_allclose(evaluation.policy_u, [0.5, 0.5])
    np.testing.assert_allclose(evaluation.historical_acceptance, [1.0, 0.0])
    np.testing.assert_allclose(evaluation.policy_revenue, [15.0, 30.0])
    np.testing.assert_allclose(evaluation.objective_contribution, [85.0, 0.0])

    summary = script.evaluation_summary(evaluation)
    assert summary["mean_objective"] == pytest.approx(42.5)
    assert summary["mean_historical_acceptance"] == pytest.approx(0.5)


def test_write_outputs_includes_summary_and_per_row_csv(tmp_path) -> None:
    evaluation = script.evaluate_historical_policy_objective(
        config=_config(),
        theta=np.asarray([0.5], dtype=float),
        row_indices=np.asarray([5, 6], dtype=int),
        historical_rows=_historical_rows(),
        estimator="first_order",
    )

    outputs = script.write_outputs(evaluation, tmp_path)

    assert outputs == [tmp_path / "summary.json", tmp_path / "per_row.csv"]
    assert "mean_objective" in (tmp_path / "summary.json").read_text(encoding="utf-8")
    csv_lines = (tmp_path / "per_row.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("csv_row_index,dummy_id,id,historical_u,policy_u")
    assert csv_lines[1].startswith("5,101,201,0.1,0.5")


def test_load_estimator_theta_reports_available_estimators() -> None:
    payload = {"estimators": {"first_order": {"theta": [1.0]}}}

    with pytest.raises(ValueError, match="Available: first_order"):
        script.load_estimator_theta(payload, "missing")


def test_reconstruct_run_row_indices_uses_full_eligible_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible = np.asarray([2, 5, 9], dtype=int)
    payload = {
        "config": {
            "n_samples": eligible.size,
            "seed": 7,
            "x_fixed_row_indices_shape": [eligible.size],
            "x_fixed_row_indices_head": eligible.tolist(),
            "x_fixed_row_indices_min": int(eligible.min()),
            "x_fixed_row_indices_max": int(eligible.max()),
        }
    }

    def fail_sample(*args, **kwargs):
        raise AssertionError("full eligible rows should not be reconstructed as a sample")

    monkeypatch.setattr(script, "eligible_csv_row_indices", lambda model_type: eligible.copy())
    monkeypatch.setattr(script, "sample_csv_row_indices", fail_sample)

    row_indices = script.reconstruct_run_row_indices(payload, "glm")

    np.testing.assert_array_equal(row_indices, eligible)


def test_reconstruct_run_row_indices_falls_back_to_seeded_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible = np.asarray([2, 5, 9], dtype=int)
    sampled = np.asarray([5, 2, 9], dtype=int)
    payload = {
        "config": {
            "n_samples": eligible.size,
            "seed": 7,
            "x_fixed_row_indices_shape": [sampled.size],
            "x_fixed_row_indices_head": sampled.tolist(),
            "x_fixed_row_indices_min": int(sampled.min()),
            "x_fixed_row_indices_max": int(sampled.max()),
        }
    }

    monkeypatch.setattr(script, "eligible_csv_row_indices", lambda model_type: eligible.copy())
    monkeypatch.setattr(script, "sample_csv_row_indices", lambda model_type, n_rows, seed: sampled.copy())

    row_indices = script.reconstruct_run_row_indices(payload, "glm")

    np.testing.assert_array_equal(row_indices, sampled)


def test_main_prints_theta_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    payload = {
        "config": {"state_dim": 19, "n_samples": 2, "seed": 7},
        "estimators": {"first_order": {"theta": [0.5]}},
    }
    monkeypatch.setattr(script, "load_summary_payload", lambda path: payload)
    monkeypatch.setattr(script, "reconstruct_run_row_indices", lambda payload, model_type: np.asarray([5, 6], dtype=int))
    monkeypatch.setattr(script, "build_config_for_saved_policy", lambda payload, row_indices, model_type: _config())
    monkeypatch.setattr(script, "load_historical_rows", lambda row_indices: _historical_rows())

    script.main(
        [
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--estimator",
            "first_order",
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-per-row",
        ]
    )

    output = capsys.readouterr().out
    assert "Theta used (first_order): [0.5]" in output
    assert "historical_acceptance = 1 - is_churn" in output
    assert (tmp_path / "out" / "summary.json").exists()
