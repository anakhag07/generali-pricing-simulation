"""Tests for src/data/feature_processor.py :: FeatureProcessor."""

import numpy as np
import pandas as pd
import pytest

from data.feature_processor import FeatureProcessor


def _make_numeric_df(n=100, seed=42):
    """Small synthetic numeric DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(5.0, 2.0, n),
        "b": rng.normal(-1.0, 0.5, n),
        "c": rng.normal(0.0, 1.0, n),
    })


def _make_mixed_df(n=100, seed=42):
    """Small synthetic DataFrame with numeric + categorical columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(3, 2, n),
        "color": rng.choice(["red", "green", "blue"], n),
    })


# -- Centering ------------------------------------------------------------------


def test_centering():
    """Transformed numeric features have approximately zero mean."""
    df = _make_numeric_df()
    fp = FeatureProcessor(numeric_cols=["a", "b", "c"], categorical_cols=[])
    out = fp.fit_transform(df)
    means = out.values.mean(axis=0)
    np.testing.assert_allclose(means, 0.0, atol=1e-10)


# -- Sphering -------------------------------------------------------------------


def test_sphering_identity_covariance():
    """After sphering (no PCA), cov(X_out) is approximately identity."""
    df = _make_numeric_df(n=500)
    fp = FeatureProcessor(numeric_cols=["a", "b", "c"], categorical_cols=[], use_pca=False)
    out = fp.fit_transform(df)
    cov = np.cov(out.values, rowvar=False)
    np.testing.assert_allclose(cov, np.eye(3), atol=0.15)


# -- PCA ------------------------------------------------------------------------


def test_pca_reduces_dimensions():
    """With n_components=2, output has 2 numeric columns."""
    df = _make_numeric_df()
    fp = FeatureProcessor(
        numeric_cols=["a", "b", "c"], categorical_cols=[],
        use_pca=True, n_components=2,
    )
    out = fp.fit_transform(df)
    assert out.shape[1] == 2


def test_pca_inverse_transform_matches_manual():
    """inverse_transform computes X_out @ V_k^T + mu correctly."""
    df = _make_numeric_df(n=100)
    fp = FeatureProcessor(
        numeric_cols=["a", "b", "c"], categorical_cols=[],
        use_pca=True, n_components=2,
    )
    out = fp.fit_transform(df)
    recovered = fp.inverse_transform_numeric(out.values)

    # Manual computation: X_out @ V_k^T + mu
    V_k = fp.pca_components_
    mu = fp.numeric_means_.values
    expected = out.values @ V_k.T + mu
    np.testing.assert_allclose(recovered, expected, atol=1e-10)
    assert recovered.shape == (100, 3)


def test_pca_inverse_without_pca_raises():
    """Raises NotImplementedError when use_pca=False."""
    df = _make_numeric_df()
    fp = FeatureProcessor(numeric_cols=["a", "b", "c"], categorical_cols=[], use_pca=False)
    fp.fit(df)
    out = fp.transform(df)
    with pytest.raises(NotImplementedError):
        fp.inverse_transform_numeric(out.values)


# -- Categorical encoding -------------------------------------------------------


def test_categorical_encoding():
    """Categories are label-encoded and normalized by category count."""
    df = _make_mixed_df()
    fp = FeatureProcessor(
        numeric_cols=["x1", "x2"], categorical_cols=["color"],
    )
    out = fp.fit_transform(df)
    cat_col = out.iloc[:, -1]  # last column is the encoded categorical
    # All values should be in [0, 1] (label / n_categories)
    assert cat_col.min() >= 0.0
    assert cat_col.max() <= 1.0
    # Number of unique encoded values should match number of categories
    assert cat_col.nunique() == 3


# -- Consistency ----------------------------------------------------------------


def test_fit_transform_equals_fit_then_transform():
    """fit_transform matches separate fit + transform."""
    df = _make_numeric_df()
    fp1 = FeatureProcessor(numeric_cols=["a", "b", "c"], categorical_cols=[])
    out1 = fp1.fit_transform(df)

    fp2 = FeatureProcessor(numeric_cols=["a", "b", "c"], categorical_cols=[])
    fp2.fit(df)
    out2 = fp2.transform(df)

    pd.testing.assert_frame_equal(out1, out2)
