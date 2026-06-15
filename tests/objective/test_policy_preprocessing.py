"""Tests for policy-side preprocessing utilities."""

import numpy as np
import pytest

from objective.policy_preprocessing import (
    PolicyFeaturePreprocessor,
    fit_policy_feature_preprocessor,
    make_policy_features,
)


def _make_x(n: int = 200, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 4))
    base[:, 1] = 2.0 * base[:, 0] + 0.1 * rng.normal(size=n)
    base[:, 2] = 5.0 + 3.0 * base[:, 2]
    base[:, 3] = -2.0 + 0.5 * base[:, 3]
    return base


def test_fit_transform_is_deterministic() -> None:
    x = _make_x()
    preprocessor = fit_policy_feature_preprocessor(x, pca_dim=2)

    out_1 = preprocessor.transform(x)
    out_2 = make_policy_features(x, preprocessor)

    assert out_1.shape == (x.shape[0], 2)
    np.testing.assert_allclose(out_1, out_2)


def test_full_sphering_keeps_input_dimension() -> None:
    x = _make_x(n=500)
    out = PolicyFeaturePreprocessor(pca_dim=None).fit_transform(x)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.cov(out, rowvar=False), np.eye(x.shape[1]), atol=0.05)


def test_pca_dim_controls_output_width() -> None:
    x = _make_x()
    preprocessor = fit_policy_feature_preprocessor(x, pca_dim=3)

    assert preprocessor.output_dim_ == 3
    assert preprocessor.transform(x).shape == (x.shape[0], 3)
    assert preprocessor.output_feature_names_ == ["policy_pc1", "policy_pc2", "policy_pc3"]


def test_standardize_without_sphere_preserves_dimension() -> None:
    x = _make_x()
    out = PolicyFeaturePreprocessor(sphere=False, pca_dim=None).fit_transform(x)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-10)


def test_invalid_pca_dim_raises() -> None:
    x = _make_x()
    with pytest.raises(ValueError, match="exceeds input dimension"):
        fit_policy_feature_preprocessor(x, pca_dim=x.shape[1] + 1)


def test_transform_requires_fit() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        PolicyFeaturePreprocessor().transform(np.ones((2, 2)))


def test_to_state_from_state_restores_transform_exactly() -> None:
    x = _make_x()
    preprocessor = fit_policy_feature_preprocessor(x, pca_dim=2)

    state = preprocessor.to_state()
    restored = PolicyFeaturePreprocessor.from_state(
        state["metadata"],
        state["arrays"],
    )

    np.testing.assert_allclose(restored.transform(x), preprocessor.transform(x))
    assert restored.to_dict() == preprocessor.to_dict()
