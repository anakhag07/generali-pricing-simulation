import numpy as np

from experiments.sensitivity_buckets import SensitivityBucket
from scripts import diagnose_low_sensitivity_policy_acceptance as script


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


def test_diagnostic_frame_includes_original_csv_row_mapping() -> None:
    bucket = SensitivityBucket(
        name="low",
        row_indices=np.array([0, 4], dtype=int),
        scores=np.array([0.1, 0.2], dtype=float),
    )
    frame = script._diagnostic_frame(
        bucket=bucket,
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
    assert frame["acceptance_u_term"].tolist() == [-0.48, -1.28]
    assert frame["policy_feature_00_p1"].tolist() == [10.0, 30.0]
    assert frame["policy_contribution_01_p2"].tolist() == [4.0, 8.0]
    assert frame["acceptance_feature_00_a1"].tolist() == [2.0, 6.0]
    assert frame["acceptance_contribution_01_a2"].tolist() == [-4.0, -8.0]


def test_resolve_bucket_names_supports_all_and_deduplicates() -> None:
    assert script._resolve_bucket_names(["all"]) == ("low", "medium", "high")
    assert script._resolve_bucket_names(["medium", "high", "medium"]) == ("medium", "high")


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
