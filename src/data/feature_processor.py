"""Feature preprocessing utilities used by bundled model artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureProcessor:
    """Center numeric features and optionally whiten them with PCA."""

    def __init__(
        self,
        numeric_cols=None,
        categorical_cols=None,
        regularization: float = 1e-6,
        missing_category: str = "__MISSING__",
        use_pca: bool = False,
        n_components=None,
        explained_variance_threshold=None,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.regularization = regularization
        self.missing_category = missing_category
        self.use_pca = use_pca
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold

    def fit(self, X: pd.DataFrame) -> "FeatureProcessor":
        X = X.copy()

        if self.numeric_cols is None:
            self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols_ = list(self.numeric_cols)

        if self.categorical_cols is None:
            self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]
        else:
            self.categorical_cols_ = list(self.categorical_cols)

        if self.numeric_cols_:
            X_num = X[self.numeric_cols_].astype(float)
            self.numeric_means_ = X_num.mean(axis=0)
            X_num_centered = X_num - self.numeric_means_

            cov = np.cov(X_num_centered.values, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])

            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            eigvals = np.maximum(eigvals, self.regularization)

            self.eigenvalues_ = eigvals
            total_var = eigvals.sum()
            self.explained_variance_ratio_ = eigvals / total_var if total_var > 0 else eigvals
            self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

            if self.use_pca:
                n_features = len(eigvals)
                if self.explained_variance_threshold is not None:
                    n_keep = np.searchsorted(
                        self.cumulative_variance_ratio_,
                        self.explained_variance_threshold,
                    ) + 1
                    n_keep = min(n_keep, n_features)
                elif self.n_components is not None:
                    if isinstance(self.n_components, float) and 0 < self.n_components < 1:
                        n_keep = np.searchsorted(
                            self.cumulative_variance_ratio_,
                            self.n_components,
                        ) + 1
                        n_keep = min(n_keep, n_features)
                    else:
                        n_keep = min(int(self.n_components), n_features)
                else:
                    n_keep = n_features

                self.n_components_ = n_keep
                eigvals_keep = eigvals[:n_keep]
                eigvecs_keep = eigvecs[:, :n_keep]
                self.pca_components_ = eigvecs_keep
                self.whitening_scale_ = 1.0 / np.sqrt(eigvals_keep)
                self.sphering_matrix_ = eigvecs_keep @ np.diag(self.whitening_scale_)
                self.numeric_feature_names_ = [f"PC{i + 1}" for i in range(n_keep)]
            else:
                self.n_components_ = len(eigvals)
                self.sphering_matrix_ = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                self.numeric_feature_names_ = [f"sphered__{col}" for col in self.numeric_cols_]
        else:
            self.numeric_means_ = pd.Series(dtype=float)
            self.sphering_matrix_ = np.empty((0, 0))
            self.n_components_ = 0
            self.numeric_feature_names_ = []

        self.cat_mapping_ = {}
        self.cat_denominator_ = {}
        self.encoded_cat_feature_names_ = []

        for col in self.categorical_cols_:
            col_values = X[col].fillna(self.missing_category).astype(str)
            categories = pd.Index(col_values.unique())
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            self.cat_mapping_[col] = mapping
            self.cat_denominator_[col] = max(len(mapping), 1)
            self.encoded_cat_feature_names_.append(f"label_cont__{col}")

        self.output_feature_names_ = [
            *self.numeric_feature_names_,
            *self.encoded_cat_feature_names_,
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.numeric_cols_:
            X_num = X[self.numeric_cols_].astype(float)
            X_num_centered = X_num - self.numeric_means_
            X_num_transformed = X_num_centered.values @ self.sphering_matrix_
        else:
            X_num_transformed = np.empty((len(X), 0))

        cat_arrays = []
        for col in self.categorical_cols_:
            values = X[col].fillna(self.missing_category).astype(str)
            mapping = self.cat_mapping_[col]
            unknown_code = len(mapping)
            denom = self.cat_denominator_[col]
            label_codes = values.map(mapping).fillna(unknown_code).astype(float)
            cat_arrays.append((label_codes / denom).to_numpy().reshape(-1, 1))

        if cat_arrays:
            X_cat_encoded = np.hstack(cat_arrays)
        else:
            X_cat_encoded = np.empty((len(X), 0))

        X_out = np.hstack([X_num_transformed, X_cat_encoded])
        return pd.DataFrame(X_out, index=X.index, columns=self.output_feature_names_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def get_explained_variance_info(self) -> dict[str, object]:
        if not hasattr(self, "eigenvalues_"):
            raise ValueError("Model not fitted yet. Call fit() first.")
        return {
            "eigenvalues": self.eigenvalues_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "cumulative_variance_ratio": self.cumulative_variance_ratio_,
            "n_components_kept": self.n_components_,
        }

    def inverse_transform_numeric(self, X_transformed: np.ndarray) -> np.ndarray:
        if not self.use_pca:
            raise NotImplementedError("Inverse transform only supports use_pca=True.")
        if not self.numeric_cols_:
            return np.empty((len(X_transformed), 0))
        X_centered = X_transformed @ self.pca_components_.T
        return X_centered + self.numeric_means_.values


__all__ = ["FeatureProcessor"]
