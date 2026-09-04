from __future__ import annotations

import numpy as np

from scripts.plot_customer_coverage_envelope_slides import (
    EXPLORATORY_U_GRID,
    MAD_CLOUD_MULTIPLIER,
    MAD_TO_NORMAL_STD,
    U_GRID,
    _mad_cloud_half_width,
    _mad_dispersion,
    _marginal_action_effective_sample_size,
    _minimize_xgboost_objective,
    _repo_spline_minimize,
    _support_weighted_band_half_width,
    _within_customer_change_summary,
)


def test_mad_cloud_uses_requested_scale_on_primary_domain() -> None:
    customer_mad = np.full(EXPLORATORY_U_GRID.shape, 5.0)

    half_width = _mad_cloud_half_width(
        {
            "exploratory_u": EXPLORATORY_U_GRID,
            "exploratory_customer_mad": customer_mad,
        }
    )

    assert half_width.shape == U_GRID.shape
    assert np.allclose(half_width, MAD_CLOUD_MULTIPLIER * 5.0)


def test_mad_dispersion_is_robust_to_customer_outlier() -> None:
    customer_profit = np.asarray(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [100.0, 13.0],
        ]
    )

    mad, robust_std = _mad_dispersion(customer_profit)

    assert np.allclose(mad, [1.0, 1.0])
    assert np.allclose(robust_std, MAD_TO_NORMAL_STD * mad)
    assert np.std(customer_profit[:, 0], ddof=1) > robust_std[0]


def test_marginal_support_band_widens_away_from_historical_actions() -> None:
    historical_u = np.asarray([0.09, 0.095, 0.10, 0.105, 0.11])
    u = np.asarray([0.0, 0.10, 0.16])

    ess = _marginal_action_effective_sample_size(
        historical_u,
        u,
        bandwidth=0.01,
    )
    relative_ess, relative_risk, half_width = _support_weighted_band_half_width(
        ess,
        max_half_width=10.0,
    )

    assert relative_ess[1] == 1.0
    assert relative_risk[1] == 1.0
    assert half_width[1] == 0.0
    assert half_width[0] > half_width[1]
    assert half_width[2] > half_width[1]
    assert np.max(half_width) == 10.0


def test_within_customer_summary_removes_customer_profit_levels() -> None:
    baseline = np.asarray([100.0, 200.0, 400.0, 800.0])
    changes = np.asarray(
        [
            [-10.0, 0.0, 10.0],
            [-5.0, 0.0, 20.0],
            [5.0, 0.0, 30.0],
            [10.0, 0.0, 40.0],
        ]
    )

    summary = _within_customer_change_summary(
        baseline[:, None] + changes,
        baseline,
    )

    assert np.allclose(summary["median"], [0.0, 0.0, 25.0])
    assert summary["robust_std"][1] == 0.0
    assert summary["robust_std"][2] > summary["robust_std"][1]


def test_repo_spline_minimizer_finds_smooth_dispersion_baseline_off_grid() -> None:
    u = np.asarray([0.0, 0.04, 0.0875, 0.12, 0.16])
    dispersion = (u - 0.0875) ** 2

    solution = _repo_spline_minimize(
        u,
        dispersion,
        start_u=0.0875,
    )

    assert solution["success"] is True
    assert abs(float(solution["u"]) - 0.0875) < 3e-4
    assert abs(float(solution["minimized_value"])) < 1e-6


def test_displayed_profit_solution_comes_from_repo_minimizer() -> None:
    u = np.linspace(0.0, 0.16, 17)
    xgboost_objective = (u - 0.0734) ** 2

    solution = _minimize_xgboost_objective(u, xgboost_objective)

    assert solution["success"] is True
    assert abs(float(solution["u"]) - 0.0734) < 0.002
    assert float(solution["minimized_value"]) <= 1e-5
    assert np.isclose(
        float(solution["plotted_profit_value"]),
        -float(solution["minimized_value"]),
    )
    assert not np.any(np.isclose(float(solution["u"]), u, atol=1e-7))


def test_uncertainty_penalty_moves_optimizer_toward_supported_region() -> None:
    u = np.linspace(0.0, 0.16, 161)
    xgboost_objective = -160.0 + 1_200.0 * (u - 0.13) ** 2
    width = 10.0 * (1.0 - np.exp(-0.5 * ((u - 0.09) / 0.025) ** 2))

    profit_solution = _minimize_xgboost_objective(u, xgboost_objective)
    adjusted_solution = _minimize_xgboost_objective(
        u,
        xgboost_objective + width,
    )

    assert float(adjusted_solution["u"]) < float(profit_solution["u"]) - 0.01
    assert abs(float(adjusted_solution["u"]) - 0.09) < abs(
        float(profit_solution["u"]) - 0.09
    )
