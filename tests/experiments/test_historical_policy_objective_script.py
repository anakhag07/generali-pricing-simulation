from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_historical_policy_objective as script


class _ConstantPolicyObjective:
    def policy_value(self, theta: np.ndarray, x_batch: object) -> np.ndarray:
        return np.full(int(x_batch.shape[0]), float(theta[0]), dtype=float)


class _ModelObjective:
    premium_col = "premium"

    def policy_value(self, theta: np.ndarray, x_batch: object) -> np.ndarray:
        return np.full(int(x_batch.shape[0]), float(theta[0]), dtype=float)

    def _acceptance_proba(self, x_batch: pd.DataFrame, u_arr: np.ndarray) -> np.ndarray:
        return x_batch["acceptance"].to_numpy(dtype=float)

    def _loss_prediction(self, x_batch: pd.DataFrame) -> np.ndarray:
        return x_batch["loss"].to_numpy(dtype=float)

    def _premium_values(self, x_batch: pd.DataFrame) -> np.ndarray:
        return x_batch[self.premium_col].to_numpy(dtype=float)


def _config(n_rows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        objective=_ConstantPolicyObjective(),
        x_fixed=pd.DataFrame({"x": np.arange(n_rows, dtype=float)}),
    )


def _model_artifact() -> SimpleNamespace:
    x_by_split = {
        "train": pd.DataFrame(
            {
                "premium": [10.0, 20.0],
                "acceptance": [0.5, 1.0],
                "loss": [100.0, 200.0],
            }
        ),
        "test": pd.DataFrame(
            {
                "premium": [30.0],
                "acceptance": [0.25],
                "loss": [300.0],
            }
        ),
    }
    rows_by_split = {
        "train": np.asarray([5, 6], dtype=int),
        "test": np.asarray([9], dtype=int),
        "all": np.asarray([5, 6, 9], dtype=int),
    }

    def load_x(split: str = "all") -> pd.DataFrame:
        if split == "all":
            return pd.concat([x_by_split["train"], x_by_split["test"]], ignore_index=True)
        return x_by_split[split].copy()

    return SimpleNamespace(
        estimator="first_order",
        theta=np.asarray([0.5], dtype=float),
        objective=SimpleNamespace(model_type="glm"),
        build_objective=lambda: _ModelObjective(),
        load_x=load_x,
        row_indices=lambda split="all": rows_by_split[split].copy(),
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
    assert summary["objective_kind"] == "historical"
    assert summary["split"] == "all"
    assert summary["mean_objective"] == pytest.approx(42.5)
    assert summary["mean_historical_acceptance"] == pytest.approx(0.5)


def test_evaluate_model_policy_objective_uses_model_acceptance_and_loss() -> None:
    evaluation = script.evaluate_model_policy_objective(
        artifact=_model_artifact(),
        split="train",
    )

    np.testing.assert_allclose(evaluation.policy_u, [0.5, 0.5])
    np.testing.assert_allclose(evaluation.policy_revenue, [15.0, 30.0])
    np.testing.assert_allclose(evaluation.objective_contribution, [42.5, 170.0])

    summary = script.model_evaluation_summary(evaluation)
    assert summary["objective_kind"] == "model"
    assert summary["split"] == "train"
    assert summary["objective_value"] == pytest.approx(106.25)
    assert summary["mean_acceptance"] == pytest.approx(0.75)


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


def test_build_config_for_saved_policy_preserves_softmax_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "run": {"experiment_name": "real_data_glm_base"},
        "config": {
            "n_samples": 10,
            "objective": {
                "u_coef": -3.0,
                "policy_preprocessor": None,
                "policy": {
                    "type": "SoftmaxPolicy",
                    "action_low": -0.1,
                    "action_high": 0.2,
                    "feature_map": {"type": "IdentityFeatureMap"},
                },
            },
        },
    }

    def fake_get_config(preset, overrides):
        return SimpleNamespace(preset=preset, overrides=overrides)

    monkeypatch.setattr(script, "get_config", fake_get_config)

    cfg = script.build_config_for_saved_policy(payload, np.asarray([5, 6], dtype=int), "glm")

    assert cfg.overrides["n_samples"] == 2
    assert cfg.overrides["softmax_action_bounds"] == (-0.1, 0.2)


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


def test_reconstruct_run_row_indices_uses_resolved_data_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = np.asarray([5, 2, 9], dtype=int)
    captured: dict[str, int] = {}
    payload = {
        "config": {
            "n_samples": sampled.size,
            "seed": 7,
            "resolved_seed_setup": {"data_seed": 123, "split_seed": 456},
            "x_fixed_row_indices_shape": [sampled.size],
            "x_fixed_row_indices_head": sampled.tolist(),
            "x_fixed_row_indices_min": int(sampled.min()),
            "x_fixed_row_indices_max": int(sampled.max()),
        }
    }

    monkeypatch.setattr(script, "eligible_csv_row_indices", lambda model_type: np.asarray([0, 1], dtype=int))

    def sample(model_type, n_rows, seed):
        captured["seed"] = seed
        return sampled.copy()

    monkeypatch.setattr(script, "sample_csv_row_indices", sample)

    row_indices = script.reconstruct_run_row_indices(payload, "glm")

    assert captured["seed"] == 123
    np.testing.assert_array_equal(row_indices, sampled)


def test_split_run_row_indices_uses_resolved_split_seed() -> None:
    row_indices = np.asarray([10, 11, 12, 13, 14], dtype=int)
    payload = {
        "config": {
            "seed": 7,
            "test_fraction": 0.4,
            "resolved_seed_setup": {"data_seed": 123, "split_seed": 456},
        }
    }

    expected_positions = np.random.default_rng(456).permutation(row_indices.size).astype(int)[:2]
    test_rows = script.split_run_row_indices(payload, row_indices, "test")

    np.testing.assert_array_equal(test_rows, row_indices[expected_positions])


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


def test_main_can_load_policy_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    artifact = SimpleNamespace(
        estimator="finite_difference",
        theta=np.asarray([0.5], dtype=float),
        objective=SimpleNamespace(model_type="glm"),
        row_indices=lambda split: np.asarray([5, 6], dtype=int),
    )
    monkeypatch.setattr(script, "load_policy_artifact", lambda path: artifact)
    monkeypatch.setattr(script, "build_config_for_policy_artifact", lambda artifact, split="all": _config())
    monkeypatch.setattr(script, "load_historical_rows", lambda row_indices: _historical_rows())

    script.main(
        [
            "--policy-artifact",
            str(tmp_path / "policy.json"),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-per-row",
        ]
    )

    output = capsys.readouterr().out
    assert "Theta used (finite_difference): [0.5]" in output
    assert (tmp_path / "out" / "summary.json").exists()


def test_main_model_objective_loads_artifact_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    artifact = _model_artifact()
    monkeypatch.setattr(script, "load_policy_artifact", lambda path: artifact)

    script.main(
        [
            "--policy-artifact",
            str(tmp_path / "policy.json"),
            "--objective",
            "model",
            "--split",
            "train",
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-per-row",
        ]
    )

    output = capsys.readouterr().out
    assert "Mean model objective: 106.250000" in output
    summary = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert '"objective_kind": "model"' in summary
    assert '"split": "train"' in summary


def test_model_objective_requires_policy_artifact(tmp_path) -> None:
    with pytest.raises(SystemExit, match="requires --policy-artifact"):
        script.main(
            [
                "--summary-json",
                str(tmp_path / "summary.json"),
                "--objective",
                "model",
                "--skip-per-row",
            ]
        )


def test_split_run_row_indices_rejects_missing_test_split() -> None:
    payload = {"config": {"seed": 7, "test_fraction": 0.0}}

    with pytest.raises(ValueError, match="no test split"):
        script.split_run_row_indices(payload, np.asarray([1, 2, 3], dtype=int), "test")
