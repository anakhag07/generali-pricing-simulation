"""Tests for the GLM sensitivity bucket experiment script."""

from __future__ import annotations

import numpy as np


def test_sensitivity_bucket_script_uses_bucket_row_counts() -> None:
    import scripts.run_glm_sensitivity_bucket_experiment as script

    assert "n_samples" not in script.RUN_OVERRIDES
    assert script.RUN_OVERRIDES["enabled_estimators"] == (
        "first_order",
        "finite_difference",
        "stein_difference",
    )
    assert script.RUN_OVERRIDES["plot"] is True


def test_summary_reports_mean_and_quantiles() -> None:
    import scripts.run_glm_sensitivity_bucket_experiment as script

    summary = script._summary("score", np.array([1.0, 2.0, 3.0, 4.0]))

    assert summary["score_mean"] == 2.5
    assert summary["score_q50"] == 2.5
    assert summary["score_q05"] < summary["score_q95"]
