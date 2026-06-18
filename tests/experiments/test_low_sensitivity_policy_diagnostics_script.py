from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.sensitivity_buckets import SensitivityBucket
from scripts import diagnose_low_sensitivity_policy_acceptance as script


class _FakeSoftmaxArtifact:
    def __init__(self) -> None:
        self.estimator = "first_order"
        self.theta = np.array([0.0, 1.0], dtype=float)
        self.objective = SimpleNamespace(model_type="glm", u_coef=-7.0)
        self.policy_head = SimpleNamespace(type="SoftmaxPolicy", action_low=-0.1, action_high=0.2)
        self.feature_map = SimpleNamespace(type="IdentityFeatureMap")
        self.policy_input = SimpleNamespace(
            policy_preprocessor=SimpleNamespace(output_feature_names_=("z",))
        )
        self.predict_calls = 0

    def row_indices(self, split: str) -> np.ndarray:
        assert split == "all"
        return np.array([10, 11, 12], dtype=int)

    def mapped_features(self, x_frame: pd.DataFrame) -> np.ndarray:
        return x_frame.loc[:, ["z"]].to_numpy(dtype=float)

    def policy_design_matrix(self, x_frame: pd.DataFrame) -> np.ndarray:
        mapped = self.mapped_features(x_frame)
        return np.column_stack([np.ones(mapped.shape[0], dtype=float), mapped])

    def predict_u(self, x_frame: pd.DataFrame, *, clip: bool = True) -> np.ndarray:
        del clip
        self.predict_calls += 1
        score = self.policy_design_matrix(x_frame) @ self.theta
        sigmoid = 1.0 / (1.0 + np.exp(-score))
        return -0.1 + 0.3 * sigmoid


def test_policy_outputs_match_softmax_formula() -> None:
    theta = np.array([0.2, 1.0, -2.0], dtype=float)
    features = np.array([[0.5, 0.25], [-1.0, 2.0]], dtype=float)

    feature_dot, policy_score, policy_sigmoid, policy_u = script._policy_outputs(theta, features)

    expected_dot = features @ theta[1:]
    expected_score = theta[0] + expected_dot
    expected_sigmoid = 1.0 / (1.0 + np.exp(-expected_score))
    np.testing.assert_allclose(feature_dot, expected_dot)
    np.testing.assert_allclose(policy_score, expected_score)
    np.testing.assert_allclose(policy_sigmoid, expected_sigmoid)
    np.testing.assert_allclose(policy_u, -0.5 + expected_sigmoid)


def test_artifact_policy_outputs_use_saved_predict_u_bounds() -> None:
    artifact = _FakeSoftmaxArtifact()
    x_frame = pd.DataFrame({"z": [0.0, 2.0]})

    names, mapped, feature_dot, policy_score, policy_sigmoid, policy_u = script._artifact_policy_outputs(
        artifact,
        x_frame,
    )

    expected_score = np.array([0.0, 2.0], dtype=float)
    expected_sigmoid = 1.0 / (1.0 + np.exp(-expected_score))
    np.testing.assert_allclose(mapped, np.array([[0.0], [2.0]], dtype=float))
    np.testing.assert_allclose(feature_dot, expected_score)
    np.testing.assert_allclose(policy_score, expected_score)
    np.testing.assert_allclose(policy_sigmoid, expected_sigmoid)
    np.testing.assert_allclose(policy_u, -0.1 + 0.3 * expected_sigmoid)
    assert names == ["z"]
    assert artifact.predict_calls == 1


def test_diagnostic_frame_includes_original_csv_row_mapping() -> None:
    bucket = SensitivityBucket(
        name="low",
        row_indices=np.array([0, 4], dtype=int),
        scores=np.array([0.1, 0.2], dtype=float),
    )
    frame = script._diagnostic_frame(
        bucket=bucket,
        policy_source="manual-theta",
        policy_estimator="manual",
        bucket_u_ref=0.1,
        bucket_row_source="eligible",
        policy_feature_names=("p1", "p2"),
        policy_features=np.array([[10.0, 20.0], [30.0, 40.0]]),
        policy_theta_coef=np.array([0.1, 0.2]),
        feature_dot=np.array([1.0, 2.0]),
        policy_score=np.array([0.5, 1.5]),
        policy_sigmoid=np.array([0.62, 0.82]),
        policy_u=np.array([0.12, 0.32]),
        acceptance_feature_names=("a1", "a2"),
        acceptance_features=np.array([[2.0, 4.0], [6.0, 8.0]]),
        acceptance_beta_x=np.array([0.5, -1.0]),
        acceptance_base_logit=np.array([2.0, 3.0]),
        beta_u=-4.0,
        acceptance_logit=np.array([1.52, 1.72]),
        acceptance_probability=np.array([0.82, 0.85]),
    )

    assert frame["row_index"].tolist() == [0, 4]
    assert frame["csv_line_number"].tolist() == [2, 6]
    assert frame["bucket_u_ref"].tolist() == [0.1, 0.1]
    assert frame["bucket_row_source"].tolist() == ["eligible", "eligible"]
    assert frame["policy_source"].tolist() == ["manual-theta", "manual-theta"]
    assert frame["acceptance_u_term"].tolist() == [-0.48, -1.28]
    assert frame["policy_feature_00_p1"].tolist() == [10.0, 30.0]
    assert frame["policy_contribution_01_p2"].tolist() == [4.0, 8.0]
    assert frame["acceptance_feature_00_a1"].tolist() == [2.0, 6.0]
    assert frame["acceptance_contribution_01_a2"].tolist() == [-4.0, -8.0]


def test_resolve_bucket_names_supports_all_and_deduplicates() -> None:
    assert script._resolve_bucket_names(["all"]) == ("low", "medium", "high")
    assert script._resolve_bucket_names(["medium", "high", "medium"]) == ("medium", "high")


def test_artifact_bucket_row_source_requires_policy_artifact() -> None:
    with pytest.raises(ValueError, match="require --policy-artifact"):
        script._bucket_source_row_indices("artifact-all", None)


def test_run_diagnostics_replays_artifact_and_bucket_options(monkeypatch, tmp_path) -> None:
    artifact = _FakeSoftmaxArtifact()
    calls: dict[str, object] = {}
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    def fake_load_policy_artifact(path):
        calls["artifact_path"] = path
        return artifact

    monkeypatch.setattr(script, "load_policy_artifact", fake_load_policy_artifact)
    monkeypatch.setattr(
        script,
        "_fit_full_data_policy_preprocessor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manual preprocessor should not be fit")),
    )
    acceptance_model = SimpleNamespace(x_feature_cols=("x",), model_frame=lambda frame: frame)
    coeffs = {
        "x_feature_names": ["x"],
        "x_coef": np.array([0.01], dtype=float),
        "intercept": 0.5,
        "u_coef": -4.0,
    }
    monkeypatch.setattr(script, "load_model_artifacts", lambda model_type: (acceptance_model, None))
    monkeypatch.setattr(script, "extract_glm_acceptance_coefficients", lambda model: coeffs)
    monkeypatch.setattr(script, "median_observed_u", lambda model_type: 0.05)

    def fake_build_buckets(*, u_ref=None, row_indices=None):
        calls["bucket_u_ref"] = u_ref
        calls["bucket_row_indices"] = np.asarray(row_indices, dtype=int)
        return (
            SensitivityBucket(
                name="low",
                row_indices=np.array([10, 11], dtype=int),
                scores=np.array([0.1, 0.2], dtype=float),
            ),
        )

    monkeypatch.setattr(script, "build_glm_sensitivity_buckets", fake_build_buckets)

    def fake_load_x_frame(model_type, row_indices=None):
        del model_type
        rows = np.asarray(row_indices, dtype=float)
        return pd.DataFrame({"x": rows, "z": rows / 10.0})

    monkeypatch.setattr(script, "load_x_frame", fake_load_x_frame)

    output_dir = tmp_path / "out"
    args = script._parse_args(
        [
            "--policy-artifact",
            str(artifact_dir),
            "--bucket",
            "low",
            "--bucket-row-source",
            "artifact-all",
            "--bucket-u-ref",
            "0.2",
            "--output-dir",
            str(output_dir),
            "--bins",
            "2",
            "--preview-rows",
            "0",
        ]
    )

    script.run_diagnostics(args)

    assert calls["artifact_path"] == artifact_dir / "policy.json"
    assert calls["bucket_u_ref"] == 0.2
    np.testing.assert_array_equal(calls["bucket_row_indices"], np.array([10, 11, 12], dtype=int))
    assert artifact.predict_calls == 1
    frame = pd.read_csv(output_dir / "low_sensitivity_policy_acceptance_diagnostics.csv")
    assert frame["policy_estimator"].tolist() == ["first_order", "first_order"]
    assert frame["bucket_row_source"].tolist() == ["artifact-all", "artifact-all"]
    assert frame["bucket_u_ref"].tolist() == [0.2, 0.2]
    assert frame["acceptance_beta_u"].tolist() == [-7.0, -7.0]
    expected_u = -0.1 + 0.3 / (1.0 + np.exp(-np.array([1.0, 1.1])))
    np.testing.assert_allclose(frame["policy_u"].to_numpy(dtype=float), expected_u)


def test_acceptance_outputs_match_sigmoid_formula() -> None:
    base_logit = np.array([1.0, -2.0], dtype=float)
    policy_u = np.array([0.25, -0.5], dtype=float)
    beta_u = -4.0

    acceptance_logit, acceptance_probability = script._acceptance_outputs(
        base_logit,
        beta_u,
        policy_u,
    )

    expected_logit = base_logit + beta_u * policy_u
    expected_probability = 1.0 / (1.0 + np.exp(-expected_logit))
    np.testing.assert_allclose(acceptance_logit, expected_logit)
    np.testing.assert_allclose(acceptance_probability, expected_probability)


def test_acceptance_terms_match_glm_predict_proba_on_real_rows() -> None:
    from data.loader import extract_glm_acceptance_coefficients, load_model_artifacts, load_x_frame

    acceptance_model, _ = load_model_artifacts("glm")
    coeffs = extract_glm_acceptance_coefficients(acceptance_model)
    x_frame = load_x_frame("glm", row_indices=np.array([213758, 276076, 294556], dtype=int))
    u_values = np.array([-0.2, 0.0, 0.15], dtype=float)

    base_logit, _, _, _ = script._acceptance_base_terms(acceptance_model, x_frame, coeffs)
    _, actual_probability = script._acceptance_outputs(base_logit, float(coeffs["u_coef"]), u_values)

    raw_frame = x_frame.loc[:, list(acceptance_model.x_feature_cols)].copy()
    raw_frame["U"] = u_values
    model_frame = acceptance_model.model_frame(raw_frame)
    expected_probability = acceptance_model.model.predict_proba(model_frame)[:, 1]

    np.testing.assert_allclose(actual_probability, expected_probability, atol=1e-12)
