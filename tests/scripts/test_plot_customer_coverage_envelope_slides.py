from __future__ import annotations

import numpy as np

from scripts.plot_customer_coverage_envelope_slides import (
    MAD_TO_NORMAL_STD,
    _mad_dispersion,
    _minimize_xgboost_objective,
)


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
