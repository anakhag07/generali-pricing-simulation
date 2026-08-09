from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import plot_model_feature_correlations as correlations


def _importance_rows() -> pd.DataFrame:
    rows = []
    for target in ("acceptance", "loss"):
        for model_index, model in enumerate(correlations.MODEL_ORDER):
            for rank, feature in enumerate(("X_numeric_a", "X_category", "X_numeric_b"), 1):
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "feature": feature,
                        "importance_mean": 4 - rank + 0.01 * model_index,
                        "rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def test_render_correlations_writes_three_matplotlib_plots(
    tmp_path: Path, monkeypatch
) -> None:
    _importance_rows().to_csv(tmp_path / "feature_importance.csv", index=False)
    frame = pd.DataFrame(
        {
            "X_numeric_a": [1.0, 2.0, 3.0, 4.0],
            "X_category": ["a", "b", "a", "b"],
            "X_numeric_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    monkeypatch.setattr(correlations, "_sample_rows", lambda n_rows, seed: np.arange(4))
    monkeypatch.setattr(
        correlations, "load_x_frame", lambda model_type, row_indices: frame.copy()
    )
    monkeypatch.setattr(
        correlations,
        "load_observed_u_array",
        lambda model_type, row_indices: np.asarray([0.0, 0.04, 0.08, 0.12]),
    )
    monkeypatch.setattr(
        correlations,
        "load_observed_loss_array",
        lambda model_type, row_indices: np.asarray([10.0, 8.0, 6.0, 4.0]),
    )
    monkeypatch.setattr(
        correlations,
        "_observed_acceptance",
        lambda row_indices: np.asarray([1.0, 1.0, 0.0, 0.0]),
    )

    metadata = correlations.render_correlations(
        tmp_path, sample_n_rows=4, sample_seed=0, top_k=3
    )

    assert metadata["excluded_categorical_or_code_features"]["acceptance"] == [
        "X_category"
    ]
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "acceptance_top_feature_spearman.png",
        "loss_top_feature_spearman.png",
        "feature_importance_rank_correlations.png",
    }
    assert (tmp_path / "correlation_analysis.json").exists()
    assert (tmp_path / "acceptance_top_feature_spearman.csv").exists()
    assert (tmp_path / "loss_top_feature_spearman.csv").exists()
    assert (tmp_path / "feature_importance_rank_correlations.csv").exists()
