"""Policy-side feature preprocessing independent of black-box artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyFeaturePreprocessor:
    """Fit-once standardization, whitening, and optional PCA for policy inputs."""

    standardize: bool = True
    sphere: bool = True
    pca_dim: int | None = None
    regularization: float = 1e-6

    def __post_init__(self) -> None:
        if self.pca_dim is not None and int(self.pca_dim) <= 0:
            raise ValueError("pca_dim must be positive or None.")
        if float(self.regularization) <= 0.0:
            raise ValueError("regularization must be positive.")
        self.pca_dim = None if self.pca_dim is None else int(self.pca_dim)
        self.regularization = float(self.regularization)

    def fit(self, x_raw: np.ndarray) -> "PolicyFeaturePreprocessor":
        """Fit preprocessing statistics on a fixed raw policy-state matrix."""
        x_arr = _as_2d_float_array(x_raw)
        n_features = x_arr.shape[1]
        if self.pca_dim is not None and self.pca_dim > n_features:
            raise ValueError(
                f"pca_dim={self.pca_dim} exceeds input dimension {n_features}."
            )

        self.input_dim_ = n_features
        self.mean_ = x_arr.mean(axis=0) if self.standardize else np.zeros(n_features)
        centered = x_arr - self.mean_
        scale = centered.std(axis=0) if self.standardize else np.ones(n_features)
        self.scale_ = np.where(scale > 0.0, scale, 1.0)
        normalized = centered / self.scale_

        if self.sphere or self.pca_dim is not None:
            cov = np.cov(normalized, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], self.regularization)
            eigvecs = eigvecs[:, order]
        else:
            eigvals = np.ones(n_features, dtype=float)
            eigvecs = np.eye(n_features, dtype=float)

        self.eigenvalues_ = eigvals
        total_var = float(eigvals.sum())
        self.explained_variance_ratio_ = eigvals / total_var if total_var > 0.0 else eigvals
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        n_keep = n_features if self.pca_dim is None else self.pca_dim
        self.output_dim_ = n_keep
        components = eigvecs[:, :n_keep]
        if self.sphere:
            scale_keep = 1.0 / np.sqrt(eigvals[:n_keep])
            self.transform_matrix_ = components @ np.diag(scale_keep)
        else:
            self.transform_matrix_ = components
        self.output_feature_names_ = [f"policy_pc{i + 1}" for i in range(n_keep)]
        return self

    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to raw policy-state rows."""
        if not hasattr(self, "transform_matrix_"):
            raise ValueError("PolicyFeaturePreprocessor is not fitted. Call fit() first.")
        x_arr = _as_2d_float_array(x_raw)
        if x_arr.shape[1] != self.input_dim_:
            raise ValueError(
                f"Expected {self.input_dim_} input columns, got {x_arr.shape[1]}."
            )
        normalized = (x_arr - self.mean_) / self.scale_
        return np.asarray(normalized @ self.transform_matrix_, dtype=float)

    def fit_transform(self, x_raw: np.ndarray) -> np.ndarray:
        """Fit and transform the same raw policy-state matrix."""
        return self.fit(x_raw).transform(x_raw)

    def to_dict(self) -> dict[str, object]:
        """Serialize preprocessing settings and fitted dimensionality metadata."""
        return {
            "standardize": bool(self.standardize),
            "sphere": bool(self.sphere),
            "pca_dim": self.pca_dim,
            "regularization": float(self.regularization),
            "input_dim": int(getattr(self, "input_dim_", 0)),
            "output_dim": int(getattr(self, "output_dim_", 0)),
            "output_feature_names": list(getattr(self, "output_feature_names_", ())),
        }


def fit_policy_feature_preprocessor(
    x_raw: np.ndarray,
    *,
    standardize: bool = True,
    sphere: bool = True,
    pca_dim: int | None = None,
    regularization: float = 1e-6,
) -> PolicyFeaturePreprocessor:
    """Return a fitted policy-side feature preprocessor."""
    return PolicyFeaturePreprocessor(
        standardize=standardize,
        sphere=sphere,
        pca_dim=pca_dim,
        regularization=regularization,
    ).fit(x_raw)


def make_policy_features(
    x_raw: np.ndarray,
    preprocessor: PolicyFeaturePreprocessor,
) -> np.ndarray:
    """Transform raw policy-state rows with a fitted policy preprocessor."""
    return preprocessor.transform(x_raw)


def _as_2d_float_array(x_raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(x_raw, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x_raw must be a 2D array.")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("x_raw must have at least one row and one column.")
    if not np.isfinite(arr).all():
        raise ValueError("x_raw must contain only finite values.")
    return arr


__all__ = [
    "PolicyFeaturePreprocessor",
    "fit_policy_feature_preprocessor",
    "make_policy_features",
]
