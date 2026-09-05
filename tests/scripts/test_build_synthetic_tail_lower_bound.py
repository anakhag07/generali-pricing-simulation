"""Tests for the explicitly synthetic support-tail construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import build_synthetic_tail_lower_bound as script


def _frame() -> pd.DataFrame:
    u = np.linspace(-0.1, 0.2, 301)
    mean = 140.0 + 100.0 * u
    width = 2.0 + 10.0 * np.square(u - 0.08)
    return pd.DataFrame(
        {
            "u": u,
            "smoothed_mean_profit": mean,
            "smoothed_support_half_width": width,
            "support_cloud_lower_profit": mean - width,
            "support_cloud_upper_profit": mean + width,
        }
    )


def test_synthetic_tail_preserves_pre_cutoff_and_hits_target() -> None:
    source = _frame()

    result, slope = script.synthetic_tail_lower_bound(
        source,
        cutoff=0.12,
        target_u=0.2,
        target_lower_profit=120.0,
    )

    before = result["u"] <= 0.12
    np.testing.assert_allclose(
        result.loc[before, "support_cloud_lower_profit"],
        source.loc[before, "support_cloud_lower_profit"],
    )
    endpoint = result.loc[np.isclose(result["u"], 0.2)].iloc[0]
    assert np.isclose(endpoint["support_cloud_lower_profit"], 120.0)
    assert np.isclose(
        endpoint["optimization_support_penalty"],
        endpoint["smoothed_mean_profit"] - 120.0,
    )
    assert slope > 0.0


def test_synthetic_tail_does_not_modify_mean_or_upper_envelope() -> None:
    source = _frame()

    result, _ = script.synthetic_tail_lower_bound(
        source,
        cutoff=0.12,
        target_u=0.2,
        target_lower_profit=120.0,
    )

    np.testing.assert_allclose(
        result["smoothed_mean_profit"], source["smoothed_mean_profit"]
    )
    np.testing.assert_allclose(
        result["support_cloud_upper_profit"],
        source["support_cloud_upper_profit"],
    )
