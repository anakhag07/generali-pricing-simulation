from __future__ import annotations

import numpy as np
import pandas as pd

import reporting.exact_spline_cache as cache_mod
from reporting.exact_spline_cache import load_or_build_exact_spline_response_grid


def test_response_cache_reuses_only_matching_recipe(monkeypatch, tmp_path) -> None:
    calls: list[np.ndarray] = []

    monkeypatch.setattr(
        cache_mod,
        "spline_anchor_weights",
        lambda rows: np.asarray([0.25, 0.75]),
    )

    def acceptance(artifact, frame, actions, weights, *, n_jobs):
        del artifact, weights, n_jobs
        calls.append(np.asarray(actions).copy())
        return np.full((len(frame), len(actions)), 0.5)

    monkeypatch.setattr(cache_mod, "exact_spline_acceptance_matrix", acceptance)
    monkeypatch.setattr(
        cache_mod,
        "predict_loss",
        lambda artifact, frame: np.full(len(frame), 20.0),
    )
    frame = pd.DataFrame({"X_policy_premium": [100.0, 120.0]})
    kwargs = {
        "cache_path": tmp_path / "responses.npz",
        "frame": frame,
        "row_indices": [1, 2],
        "eligible_rows": [1, 2, 3],
        "acceptance_artifact": object(),
        "loss_artifact": object(),
        "n_jobs": 1,
    }

    first = load_or_build_exact_spline_response_grid(
        **kwargs, action_grid=[0.0, 0.1]
    )
    second = load_or_build_exact_spline_response_grid(
        **kwargs, action_grid=[0.0, 0.1]
    )
    load_or_build_exact_spline_response_grid(
        **kwargs, action_grid=[0.0, 0.2]
    )

    assert len(calls) == 2
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
