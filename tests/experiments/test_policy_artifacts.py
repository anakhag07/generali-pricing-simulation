from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json

import numpy as np
import pytest

from experiments.configs import get_config
from experiments.policy_artifacts import build_policy_artifact, load_policy_artifact
from experiments.policy_validation import policy_u_values
from experiments.reporting import JsonReporter, PolicyArtifactReporter, RunContext
from experiments.reporting.json_summary import _policy_artifact_paths
from experiments.run import run_experiment
from objective.noise import NoisyObjective, NoNoise


class _NamesResult:
    """Minimal result stub exposing only the estimator names iterated over."""

    def __init__(self, names) -> None:
        self.results = {name: None for name in names}


def _touch_policy_json(run_dir, name):
    policy_json = run_dir / "policies" / name / "policy.json"
    policy_json.parent.mkdir(parents=True, exist_ok=True)
    policy_json.write_text("{}", encoding="utf-8")
    return policy_json


def _run_context(run_dir):
    return RunContext(
        experiment_name="exp",
        run_id="rid",
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
        started_at=datetime(2026, 1, 1),
    )


def test_policy_artifact_paths_default_relative_to_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _touch_policy_json(run_dir, "first_order")

    paths = _policy_artifact_paths(_run_context(run_dir), _NamesResult(["first_order"]))

    # Default (summary_dir=None) is unchanged: relative to run_dir.
    assert paths == {"first_order": "policies/first_order/policy.json"}


def test_policy_artifact_paths_relative_to_summary_dir(tmp_path) -> None:
    # Seed-sweep layout: summary.json is written to the variant root, while the
    # artifacts live under a nested per-seed run_dir.
    variant_root = tmp_path / "variant"
    run_dir = variant_root / "runs" / "seed-1" / "ts"
    policy_json = _touch_policy_json(run_dir, "first_order")

    paths = _policy_artifact_paths(
        _run_context(run_dir), _NamesResult(["first_order"]), summary_dir=variant_root
    )

    assert paths == {"first_order": "runs/seed-1/ts/policies/first_order/policy.json"}
    # The recorded path must resolve against the summary's own directory.
    assert (variant_root / paths["first_order"]).resolve() == policy_json.resolve()


def test_policy_artifact_paths_skips_missing_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _touch_policy_json(run_dir, "first_order")

    paths = _policy_artifact_paths(
        _run_context(run_dir), _NamesResult(["first_order", "spsa"]), summary_dir=run_dir
    )

    assert paths == {"first_order": "policies/first_order/policy.json"}


@pytest.fixture(scope="module")
def glm_policy_result():
    config = get_config(
        "real_data_glm_base",
        overrides={
            "n_samples": 24,
            "train_fraction": 0.75,
            "test_fraction": 0.25,
            "policy_kind": "softmax",
            "softmax_action_bounds": (-0.1, 0.2),
            "initial_u": 0.0,
            "feature_order": "quadratic",
            "policy_preprocessing": "no_pca",
            "step_rule": "constant",
            "t_steps": 1,
            "n_grad_samples": 2,
            "enabled_estimators": ("first_order",),
            "grad_norm_tol": None,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )
    return run_experiment(config)


def test_policy_artifact_round_trip_predicts_same_train_u(tmp_path, glm_policy_result) -> None:
    artifact = build_policy_artifact(glm_policy_result, "first_order")
    policy_json = artifact.save(tmp_path / "first_order" / "policy.json")

    loaded = load_policy_artifact(policy_json)

    theta = glm_policy_result.results["first_order"].theta
    expected_u = policy_u_values(glm_policy_result.config.objective, theta, glm_policy_result.x_samples)
    np.testing.assert_allclose(loaded.predict_u(split="train"), expected_u)
    np.testing.assert_array_equal(loaded.row_indices("train"), glm_policy_result.train_row_indices)
    assert (policy_json.parent / "arrays.npz").exists()

    payload = json.loads(policy_json.read_text(encoding="utf-8"))
    preprocessing = payload["policy_input_preprocessing"]
    assert preprocessing["artifact_preprocessing"]["enabled"] is True
    assert preprocessing["policy_side_preprocessing"]["enabled"] is True
    assert payload["feature_map"]["type"] == "QuadraticFeatureMap"
    assert payload["policy_head"]["type"] == "SoftmaxPolicy"


def test_policy_artifact_unwraps_noisy_model_based_objective(glm_policy_result) -> None:
    noisy_result = replace(
        glm_policy_result,
        config=replace(
            glm_policy_result.config,
            objective=NoisyObjective(glm_policy_result.config.objective, NoNoise()),
        ),
    )

    artifact = build_policy_artifact(noisy_result, "first_order")

    assert artifact.objective.model_type == "glm"
    assert artifact.objective.loss_source == glm_policy_result.config.objective.loss_source


def test_policy_artifact_round_trip_matches_train_metrics(tmp_path, glm_policy_result) -> None:
    artifact = build_policy_artifact(glm_policy_result, "first_order")
    loaded = load_policy_artifact(artifact.save(tmp_path / "policy.json"))

    actual = loaded.evaluate(split="train")
    expected = glm_policy_result.train_metrics["first_order"]

    assert actual.n_samples == expected.n_samples
    assert actual.objective_value == pytest.approx(expected.objective_value)
    assert actual.objective_sum == pytest.approx(expected.objective_sum)
    assert actual.mean_u == pytest.approx(expected.mean_u)
    assert actual.mean_acceptance == pytest.approx(expected.mean_acceptance)
    assert actual.projected_loss == pytest.approx(expected.projected_loss)
    assert actual.projected_revenue == pytest.approx(expected.projected_revenue)


def test_policy_artifact_separates_preprocessing_from_feature_map(tmp_path, glm_policy_result) -> None:
    artifact = build_policy_artifact(glm_policy_result, "first_order")
    loaded = load_policy_artifact(artifact.save(tmp_path / "policy.json"))

    expected_z = glm_policy_result.config.objective._policy_features(glm_policy_result.x_samples)
    actual_z = loaded.policy_input_features(split="train")
    mapped = loaded.mapped_features(split="train")
    phi = loaded.policy_design_matrix(split="train")

    np.testing.assert_allclose(actual_z, expected_z)
    assert loaded.policy_input.policy_preprocessor is not None
    assert actual_z.shape[1] == loaded.policy_input.policy_preprocessor.output_dim_
    assert mapped.shape[1] > actual_z.shape[1]
    assert phi.shape[1] == loaded.theta.size
    assert phi.shape[1] == mapped.shape[1] + 1


def test_policy_artifact_reporter_writes_summary_paths(tmp_path, glm_policy_result) -> None:
    run_context = RunContext(
        experiment_name="real_data_glm_base",
        run_id="test-run",
        run_dir=tmp_path,
        plots_dir=tmp_path / "plots",
        started_at=datetime(2026, 1, 1),
    )

    PolicyArtifactReporter().on_end(run_context, glm_policy_result)
    JsonReporter().on_end(run_context, glm_policy_result)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["policy_artifacts"] == {
        "first_order": "policies/first_order/policy.json"
    }
    loaded = load_policy_artifact(tmp_path / summary["policy_artifacts"]["first_order"])
    np.testing.assert_allclose(
        loaded.predict_u(split="train"),
        policy_u_values(
            glm_policy_result.config.objective,
            glm_policy_result.results["first_order"].theta,
            glm_policy_result.x_samples,
        ),
    )


def test_xgb_logit_spline_policy_artifact_replays_id_bound_rows(tmp_path) -> None:
    config = get_config(
        "real_data_xgb_logit_spline_base",
        overrides={
            "n_samples": 8,
            "train_fraction": 0.75,
            "test_fraction": 0.25,
            "policy_kind": "constant",
            "initial_u": 0.08,
            "step_rule": "constant",
            "step_size": 1e-4,
            "t_steps": 1,
            "enabled_estimators": ("first_order",),
            "grad_norm_tol": None,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )
    result = run_experiment(config)
    artifact = build_policy_artifact(result, "first_order")
    loaded = load_policy_artifact(artifact.save(tmp_path / "policy.json"))

    assert loaded.objective.model_type == "xgb_logit_spline"
    assert "id" in loaded.load_x(split="train").columns
    actual = loaded.evaluate(split="train")
    expected = result.train_metrics["first_order"]
    assert actual.objective_value == pytest.approx(expected.objective_value)
    assert actual.mean_acceptance == pytest.approx(expected.mean_acceptance)


def test_monotone_policy_replay_records_independent_model_ids(tmp_path) -> None:
    config = get_config(
        "real_data_monotone_spline_glm_20260728_base",
        overrides={
            "n_samples": 8,
            "train_fraction": 1.0,
            "test_fraction": 0.0,
            "policy_kind": "constant",
            "initial_u": 0.08,
            "step_rule": "constant",
            "t_steps": 1,
            "enabled_estimators": ("first_order",),
            "grad_norm_tol": None,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
    )
    result = run_experiment(config)
    artifact = build_policy_artifact(result, "first_order")
    policy_json = artifact.save(tmp_path / "policy.json")
    loaded = load_policy_artifact(policy_json)
    payload = json.loads(policy_json.read_text(encoding="utf-8"))

    assert loaded.objective.model_type is None
    assert loaded.objective.acceptance_model_type == "xgb_monotone_spline_20260728"
    assert loaded.objective.loss_model_type == "glm_20260527"
    assert payload["schema_version"] == 2
    assert "id" in loaded.load_x(split="train").columns
    assert loaded.evaluate(split="train").objective_value == pytest.approx(
        result.train_metrics["first_order"].objective_value
    )
