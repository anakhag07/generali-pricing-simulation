"""Plot a t-SNE embedding of sampled GLM real-data feature rows."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_GLM,
    LOSS_FEATURE_COLS,
    dataset_csv_path,
    load_observed_u_array,
    load_x_array,
    sample_csv_row_indices,
)


FEATURE_SETS = {
    "glm": tuple(FEATURE_COLS_GLM),
    "acceptance": tuple(ACCEPTANCE_STATE_COLS),
    "loss": tuple(LOSS_FEATURE_COLS),
}

DEFAULT_COLOR_COLUMNS = (
    "cluster",
    "F_acc",
    "observed_u",
    "X_policy_premium",
    "X_age",
    "X_bonus_malus_rating",
    "X_policy_tenure",
)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    feature_cols = FEATURE_SETS[args.feature_set]
    row_indices = sample_csv_row_indices("glm", n_rows=args.n_rows, seed=args.seed)
    x_all = load_x_array("glm", row_indices=row_indices)
    observed_u = load_observed_u_array("glm", row_indices=row_indices)
    prediction_frame = load_glm_prediction_frame(row_indices)

    all_feature_frame = pd.DataFrame(x_all, columns=FEATURE_COLS_GLM)
    feature_frame = all_feature_frame.loc[:, list(feature_cols)]
    embedding = compute_tsne_embedding(
        feature_frame.to_numpy(dtype=float),
        seed=args.seed,
        pca_dim=args.pca_dim,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
    )
    clusters = compute_clusters(
        feature_frame.to_numpy(dtype=float),
        n_clusters=args.n_clusters,
        seed=args.seed,
    )

    output_dir = _output_dir(args.output_root, args.feature_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_frame = pd.DataFrame(
        {
            "row_index": row_indices.astype(int),
            "tsne_1": embedding[:, 0],
            "tsne_2": embedding[:, 1],
            "F_acc": prediction_frame["prob_acceptance"].to_numpy(dtype=float),
            "churn_prediction": prediction_frame["churn_prediction"].to_numpy(dtype=float),
            "observed_u": observed_u,
        }
    )
    if clusters is not None:
        output_frame["cluster"] = clusters.astype(int)
    output_frame = pd.concat([output_frame, all_feature_frame.reset_index(drop=True)], axis=1)
    output_frame.to_csv(output_dir / "embedding.csv", index=False)

    for column in _color_columns(args.color_by, output_frame.columns):
        plot_embedding(
            output_frame,
            color_column=column,
            output_path=output_dir / f"tsne_by_{_safe_filename(column)}.png",
        )

    print(f"Wrote GLM t-SNE diagnostics to {output_dir}")


def compute_tsne_embedding(
    x_values: np.ndarray,
    *,
    seed: int,
    pca_dim: int,
    perplexity: float,
    max_iter: int,
    n_jobs: int | None,
) -> np.ndarray:
    """Standardize, optionally PCA-reduce, then compute a 2D t-SNE embedding."""
    x_arr = _as_2d(x_values)
    if perplexity >= x_arr.shape[0]:
        raise ValueError("perplexity must be smaller than n_rows.")
    standardized = StandardScaler().fit_transform(x_arr)
    pca_width = min(int(pca_dim), standardized.shape[1])
    if pca_width > 0 and pca_width < standardized.shape[1]:
        tsne_input = PCA(n_components=pca_width, random_state=seed).fit_transform(standardized)
    else:
        tsne_input = standardized
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        max_iter=int(max_iter),
        random_state=int(seed),
        n_jobs=n_jobs,
    )
    return np.asarray(tsne.fit_transform(tsne_input), dtype=float)


def load_glm_prediction_frame(row_indices: np.ndarray) -> pd.DataFrame:
    """Load saved GLM out-of-fold churn and acceptance predictions for sampled rows."""
    predictions = pd.read_csv(
        dataset_csv_path(),
        sep=";",
        usecols=["churn_prediction", "prob_acceptance"],
    )
    return predictions.iloc[np.asarray(row_indices, dtype=int)].reset_index(drop=True)


def compute_clusters(
    x_values: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
) -> np.ndarray | None:
    """Cluster standardized feature rows with KMeans for coloring the embedding."""
    if int(n_clusters) <= 0:
        return None
    x_arr = _as_2d(x_values)
    if int(n_clusters) > x_arr.shape[0]:
        raise ValueError("n_clusters must be <= n_rows.")
    standardized = StandardScaler().fit_transform(x_arr)
    return KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto").fit_predict(standardized)


def plot_embedding(
    frame: pd.DataFrame,
    *,
    color_column: str,
    output_path: Path,
) -> None:
    """Write one t-SNE scatter plot colored by a column in the embedding frame."""
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.2))
    values = frame[color_column]
    scatter = ax.scatter(
        frame["tsne_1"],
        frame["tsne_2"],
        c=values,
        s=8.0,
        alpha=0.72,
        cmap="tab10" if color_column == "cluster" else "viridis",
        linewidths=0.0,
    )
    ax.set_title(f"GLM data t-SNE colored by {color_column}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="glm")
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/data-tsne")
    parser.add_argument("--color-by", nargs="*", default=list(DEFAULT_COLOR_COLUMNS))
    return parser


def _color_columns(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    available_set = set(available)
    return [column for column in requested if column in available_set]


def _output_dir(output_root: str, feature_set: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / "glm" / feature_set / timestamp


def _safe_filename(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_")


def _as_2d(x_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(x_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x_values must be a 2D array.")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError("x_values must have at least two rows and one column.")
    if not np.isfinite(arr).all():
        raise ValueError("x_values must contain only finite values.")
    return arr


if __name__ == "__main__":
    main()
